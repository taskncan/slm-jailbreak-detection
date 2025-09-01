#!/usr/bin/env python3
"""
SLM Ensemble (SBERT + BOW) — Dual Threshold (Single vs. Multi Turn)
Train with Optuna, fold-local centroids, handcrafted features, and
optional Mahalanobis outlier score (benign cluster).
"""

from __future__ import annotations
import gc, json, re
from glob import glob
from pathlib import Path

import joblib
import numpy as np
import optuna
import pandas as pd
import torch
import xgboost as xgb
from scipy import sparse as sp
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, brier_score_loss, confusion_matrix, f1_score,
    precision_recall_curve, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.covariance import LedoitWolf

from features import build_feature_vector
from helpers import is_single_turn_text, extract_last_user_and_history

# =========================
# GPU helpers
# =========================
try:
    import cupy as cp  # optional
except Exception:
    cp = None

def to_gpu(x):  return cp.asarray(x) if (cp is not None) else x
def to_cpu(x):  return cp.asnumpy(x) if (cp is not None and isinstance(x, cp.ndarray)) else x

def get_device() -> str:
    if torch.cuda.is_available(): return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available(): return "mps"
    return "cpu"

DEVICE = get_device()
XGB_TREE_METHOD = "hist"
XGB_DEVICE = "cuda" if DEVICE == "cuda" else "cpu"
print(f"[INFO] Detected device: {DEVICE}  (xgb device = {XGB_DEVICE})")

def xgb_proba_gpu_safe(clf: xgb.XGBClassifier, X) -> np.ndarray:
    X_dev = to_gpu(X); p = clf.predict_proba(X_dev)[:, 1]
    return to_cpu(p).astype("float32")

# =========================
# Config
# =========================
TS = pd.Timestamp.now().strftime("%Y%m%d_%H%M")
SEED = 42
NFOLD = 5
EARLY_STOPPING = 20
WINDOW = 3
TARGET_RECALL = None
USE_OUTLIER = False   

ROOT = Path(".")
RAW = ROOT / "data" / "raw"
REGDIR = ROOT / "detector" / "model_registry"
MODELDIR = REGDIR / "models"
METRICSDIR = REGDIR / "metrics"
ENCODESDIR = REGDIR / "encodes"
for d in (REGDIR, MODELDIR, METRICSDIR, ENCODESDIR):
    d.mkdir(parents=True, exist_ok=True)

SBERT_NAME = "sentence-transformers/all-distilroberta-v1"
model_out_path   = MODELDIR   / "model-ensemble-dualthr.pkl"
metrics_out_path = METRICSDIR / "metrics-ensemble-dualthr.json"

# =========================
# Data
# =========================
csvs = sorted(glob(str(RAW / "train_*.csv")))
if not csvs:
    raise RuntimeError("[ERR] No training CSVs found in data/raw/")
used_csv = Path(csvs[-1])
print(f"[INFO] Using training CSV: {used_csv.name}")

df = pd.read_csv(used_csv)
texts  = df["text"].astype(str).to_numpy()
labels = df["label"].to_numpy().astype(int)

print("[INFO] Preparing sliding-window strings…")
prompts_text, histories_text = [], []
for t in texts:
    last_user, hist = extract_last_user_and_history(t, window=WINDOW)
    prompts_text.append(last_user)
    histories_text.append(hist if isinstance(hist, list) else [hist] if hist else [])

# =========================
# SBERT encodings (cached, no leakage)
# =========================
print("[INFO] Encoding SBERT embeddings (cached)…")
sbert = SentenceTransformer(SBERT_NAME, device=DEVICE)

def encode_or_load(cache_name: str, encoder, arr) -> np.ndarray:
    path = ENCODESDIR / cache_name
    if path.exists(): return np.load(path)
    vec = encoder.encode(arr, batch_size=128, normalize_embeddings=True).astype("float32")
    np.save(path, vec); return vec

csv_tag = used_csv.stem
E_PROMPT_S  = encode_or_load(f"cache_E_PROMPT_S__{csv_tag}__W{WINDOW}.npy",  sbert, prompts_text)
E_HISTORY_S = encode_or_load(f"cache_E_HISTORY_S__{csv_tag}__W{WINDOW}.npy",
                             sbert, [" ".join(x) if isinstance(x, list) else x for x in histories_text])

# =========================
# Outlier scorer (Mahalanobis vs. benign) — optional
# =========================
class MahalanobisOutlier:
    def __init__(self, eps: float = 1e-6):
        self.eps = eps
        self.mu_: np.ndarray | None = None
        self.prec_: np.ndarray | None = None  # inverse covariance

    def fit(self, X_benign: np.ndarray):
        X = np.asarray(X_benign, dtype="float64")
        self.mu_ = X.mean(axis=0)
        lw = LedoitWolf().fit(X)  # shrinkage → stable covariance
        cov = lw.covariance_
        cov = cov + self.eps * np.eye(cov.shape[0], dtype=cov.dtype)  # ridge
        self.prec_ = np.linalg.inv(cov)
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype="float64")
        d = X - self.mu_
        d2 = np.einsum("ij,jk,ik->i", d, self.prec_, d)  # squared Mahalanobis
        return d2.astype("float32")

    def to_payload(self):
        return dict(
            outlier_type="mahalanobis_benign",
            outlier_mu=self.mu_.astype("float32"),
            outlier_precision=self.prec_.astype("float32"),
        )

# =========================
# Feature matrix builder
# =========================
def build_feature_matrix(texts_, embs, benign_c, attack_c, history_embs,
                         prompts_txt, histories_txt, outlier_scores=None) -> np.ndarray:
    feats = []
    for i, (txt, emb, hist_emb, ptxt, htxt) in enumerate(
        zip(texts_, embs, history_embs, prompts_txt, histories_txt)
    ):
        vec = build_feature_vector(
            txt, emb, benign_c, attack_c, hist_emb, ptxt, htxt,
            outlier_score=(outlier_scores[i] if outlier_scores is not None else 0.0),
        )
        feats.append(vec)
    return np.vstack(feats).astype("float32")

def best_f1_from_probs(y_true: np.ndarray, probs: np.ndarray) -> float:
    prec, rec, _thr = precision_recall_curve(y_true, probs)
    f1 = 2 * prec * rec / (prec + rec + 1e-9)
    return float(np.max(f1))

def pick_threshold(y_true: np.ndarray, probs: np.ndarray, target_recall: float | None = None, beta: float = 1.0):
    pos = int(np.sum(y_true))
    if pos == 0 or pos == len(y_true): return 0.5, "fallback"
    prec, rec, thr = precision_recall_curve(y_true, probs)
    if target_recall is not None:
        ok = np.where(rec >= target_recall)[0]
        if len(ok):
            idx = ok[np.argmax(prec[ok])]
            return (thr[idx] if idx < len(thr) else 0.5), "recall_target"
    fbeta = (1 + beta**2) * prec * rec / (beta**2 * prec + rec + 1e-9)
    idx = np.argmax(fbeta)
    return (thr[idx] if idx < len(thr) else 0.5), f"F{beta}"

# =========================
# Cross-validated training
# =========================
skf = StratifiedKFold(n_splits=NFOLD, shuffle=True, random_state=SEED)
val_metrics = []
thr_single_all, thr_multi_all = [], []
best_overall_f1, best_model, best_outlier_payload = 0.0, None, None

for fold, (tr_idx, va_idx) in enumerate(skf.split(texts, labels), 1):
    print(f"\n[INFO] Fold {fold}/{NFOLD} -------------------")
    Xt, Xv = texts[tr_idx], texts[va_idx]
    yt, yv = labels[tr_idx], labels[va_idx]

    Ep_tr, Ep_val = E_PROMPT_S[tr_idx],  E_PROMPT_S[va_idx]
    Eh_tr, Eh_val = E_HISTORY_S[tr_idx], E_HISTORY_S[va_idx]

    # ---- Fold-local centroids (no leakage)
    benign_c = Ep_tr[yt == 0].mean(axis=0).astype("float32")
    attack_c = Ep_tr[yt == 1].mean(axis=0).astype("float32")

    # ---- Outlier scores (optional)
    if USE_OUTLIER:
        outlier = MahalanobisOutlier().fit(Ep_tr[yt == 0])
        out_tr = outlier.score(Ep_tr)
        out_val = outlier.score(Ep_val)
    else:
        outlier = None
        out_tr  = np.zeros(len(Ep_tr),  dtype="float32")
        out_val = np.zeros(len(Ep_val), dtype="float32")

    # ---- TF-IDF (fit on Xt only)
    vec_char = TfidfVectorizer(analyzer="char", ngram_range=(2, 6), max_features=5000)
    vec_word = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), max_features=3000)
    vec_char.fit(Xt); vec_word.fit(Xt)

    Xc_tr = vec_char.transform(Xt); Xw_tr = vec_word.transform(Xt)
    X_bow_tr = sp.hstack([Xc_tr, Xw_tr]).tocsr()
    Xc_val = vec_char.transform(Xv); Xw_val = vec_word.transform(Xv)
    X_bow_val = sp.hstack([Xc_val, Xw_val]).tocsr()

    print("[INFO] Building handcrafted feature matrices…")
    H_tr = build_feature_matrix(
        Xt, Ep_tr, benign_c, attack_c, Eh_tr,
        [prompts_text[i] for i in tr_idx], [histories_text[i] for i in tr_idx],
        outlier_scores=out_tr,
    )
    H_val = build_feature_matrix(
        Xv, Ep_val, benign_c, attack_c, Eh_val,
        [prompts_text[i] for i in va_idx], [histories_text[i] for i in va_idx],
        outlier_scores=out_val,
    )

    # ---- Optuna: BOW LR
    def tune_lr(trial):
        params = {"C": trial.suggest_float("C", 0.01, 10.0, log=True),
                  "max_iter": 400, "class_weight": "balanced", "random_state": SEED}
        m = LogisticRegression(**params).fit(X_bow_tr, yt)
        probs = m.predict_proba(X_bow_val)[:, 1]
        return best_f1_from_probs(yv, probs)
    study_lr = optuna.create_study(direction="maximize")
    study_lr.optimize(tune_lr, n_trials=20, show_progress_bar=False)
    best_lr_params = study_lr.best_params

    # ---- Optuna: XGB on SBERT embeddings
    def tune_xgb_emb(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 600, step=100),
            "max_depth": trial.suggest_int("max_depth", 2, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 5.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 10.0),
            "random_state": SEED, "n_jobs": -1, "device": XGB_DEVICE,
            "tree_method": XGB_TREE_METHOD, "eval_metric": "logloss",
            "early_stopping_rounds": EARLY_STOPPING,
        }
        m = xgb.XGBClassifier(**params)
        m.fit(to_gpu(Ep_tr), yt, eval_set=[(to_gpu(Ep_val), yv)], verbose=False)
        probs = xgb_proba_gpu_safe(m, Ep_val)
        return best_f1_from_probs(yv, probs)
    study_emb = optuna.create_study(direction="maximize")
    study_emb.optimize(tune_xgb_emb, n_trials=25, show_progress_bar=False)
    best_xgb_emb_params = study_emb.best_params

    # ---- Optuna: XGB on handcrafted features
    def tune_xgb_feat(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 600, step=100),
            "max_depth": trial.suggest_int("max_depth", 2, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 5.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 10.0),
            "random_state": SEED, "n_jobs": -1, "device": XGB_DEVICE,
            "tree_method": XGB_TREE_METHOD, "eval_metric": "logloss",
            "early_stopping_rounds": EARLY_STOPPING,
        }
        m = xgb.XGBClassifier(**params)
        m.fit(to_gpu(H_tr), yt, eval_set=[(to_gpu(H_val), yv)], verbose=False)
        probs = xgb_proba_gpu_safe(m, H_val)
        return best_f1_from_probs(yv, probs)
    study_feat = optuna.create_study(direction="maximize")
    study_feat.optimize(tune_xgb_feat, n_trials=25, show_progress_bar=False)
    best_xgb_feat_params = study_feat.best_params

    # ---- Train base models with best params
    bow_lr = LogisticRegression(max_iter=400, class_weight="balanced",
                                random_state=SEED, **best_lr_params).fit(X_bow_tr, yt)
    xgb_emb = xgb.XGBClassifier(**best_xgb_emb_params, random_state=SEED, n_jobs=-1,
                                device=XGB_DEVICE, tree_method=XGB_TREE_METHOD,
                                eval_metric="logloss", early_stopping_rounds=EARLY_STOPPING)
    xgb_emb.fit(to_gpu(Ep_tr), yt, eval_set=[(to_gpu(Ep_val), yv)], verbose=False)

    xgb_feat = xgb.XGBClassifier(**best_xgb_feat_params, random_state=SEED, n_jobs=-1,
                                 device=XGB_DEVICE, tree_method=XGB_TREE_METHOD,
                                 eval_metric="logloss", early_stopping_rounds=EARLY_STOPPING)
    xgb_feat.fit(to_gpu(H_tr), yt, eval_set=[(to_gpu(H_val), yv)], verbose=False)

    # ---- Ensemble inputs (3 channels)
    P_tr = np.column_stack([
        bow_lr.predict_proba(X_bow_tr)[:, 1].astype("float32"),
        xgb_proba_gpu_safe(xgb_emb,  Ep_tr),
        xgb_proba_gpu_safe(xgb_feat, H_tr),
    ]).astype("float32")
    P_val = np.column_stack([
        bow_lr.predict_proba(X_bow_val)[:, 1].astype("float32"),
        xgb_proba_gpu_safe(xgb_emb,  Ep_val),
        xgb_proba_gpu_safe(xgb_feat, H_val),
    ]).astype("float32")

    # ---- Optuna: ensemble head
    def tune_ens(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 400, step=100),
            "max_depth": trial.suggest_int("max_depth", 2, 6),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 3.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 5.0),
            "random_state": SEED, "n_jobs": -1, "device": XGB_DEVICE,
            "tree_method": XGB_TREE_METHOD, "eval_metric": "logloss",
            "early_stopping_rounds": EARLY_STOPPING,
        }
        m = xgb.XGBClassifier(**params)
        m.fit(to_gpu(P_tr), yt, eval_set=[(to_gpu(P_val), yv)], verbose=False)
        probs = xgb_proba_gpu_safe(m, P_val)
        return best_f1_from_probs(yv, probs)
    study_ens = optuna.create_study(direction="maximize")
    study_ens.optimize(tune_ens, n_trials=20, show_progress_bar=False)
    best_ens_params = study_ens.best_params

    ensemble = xgb.XGBClassifier(**best_ens_params, random_state=SEED, n_jobs=-1,
                                 device=XGB_DEVICE, tree_method=XGB_TREE_METHOD,
                                 eval_metric="logloss", early_stopping_rounds=EARLY_STOPPING)
    ensemble.fit(to_gpu(P_tr), yt, eval_set=[(to_gpu(P_val), yv)], verbose=False)

    # ---- Dual thresholds on validation split
    probs_val = xgb_proba_gpu_safe(ensemble, P_val)
    single_mask = np.array([is_single_turn_text(x) for x in Xv])
    multi_mask  = ~single_mask

    thr_single, rule_s = (pick_threshold(yv[single_mask], probs_val[single_mask], TARGET_RECALL, beta=1.0)
                          if single_mask.any() else (0.5, "fallback"))
    thr_multi,  rule_m = (pick_threshold(yv[multi_mask],  probs_val[multi_mask],  TARGET_RECALL, beta=1.0)
                          if multi_mask.any() else (0.5, "fallback"))

    thr_applied = np.where(single_mask, thr_single, thr_multi).astype("float32")
    preds_val = (probs_val >= thr_applied).astype(int)

    # ---- Metrics (overall)
    metrics = dict(
        roc_auc=roc_auc_score(yv, probs_val),
        pr_auc=average_precision_score(yv, probs_val),
        brier=brier_score_loss(yv, probs_val),
        f1=f1_score(yv, preds_val),
        precision=precision_score(yv, preds_val),
        recall=recall_score(yv, preds_val),
        confusion_matrix=confusion_matrix(yv, preds_val).tolist(),
        thr_single=float(thr_single), thr_multi=float(thr_multi),
        thr_rule_single=rule_s, thr_rule_multi=rule_m,
    )
    val_metrics.append(metrics)

    print(f"  [Fold {fold}] F1={metrics['f1']:.4f}  ROC_AUC={metrics['roc_auc']:.4f}  "
          f"PR_AUC={metrics['pr_auc']:.4f}  thr_single={thr_single:.4f} ({rule_s})  "
          f"thr_multi={thr_multi:.4f} ({rule_m})")

    # Keep best fold artefacts (and its outlier payload)
    if metrics["f1"] > best_overall_f1:
        best_overall_f1 = metrics["f1"]
        best_model = dict(
            vec_char=vec_char,
            vec_word=vec_word,
            bow_lr=bow_lr,
            sbert_name=SBERT_NAME,
            xgb_emb=xgb_emb,
            xgb_feat_s=xgb_feat,
            ensemble_clf=ensemble,
            best_threshold_single=float(thr_single),
            best_threshold_multi=float(thr_multi),
            benign_c=benign_c,
            attack_c=attack_c,
            window=WINDOW,
            use_outlier=USE_OUTLIER,
        )
        if USE_OUTLIER and outlier is not None:
            best_outlier_payload = outlier.to_payload()
        else:
            best_outlier_payload = None

    # Cleanup
    del vec_char, vec_word
    del Xc_tr, Xw_tr, X_bow_tr, Xc_val, Xw_val, X_bow_val
    del H_tr, H_val, Ep_tr, Ep_val, Eh_tr, Eh_val
    del bow_lr, xgb_emb, xgb_feat, ensemble
    gc.collect()

# =========================
# Save artefacts & metrics
# =========================
if best_model is not None and USE_OUTLIER and best_outlier_payload is not None:
    best_model.update(best_outlier_payload)

joblib.dump(best_model, model_out_path)
print(f"\n✓ Best model saved to: {model_out_path}")

metrics_summary = {
    "folds": val_metrics,
    "mean_f1": float(np.mean([m["f1"] for m in val_metrics])),
    "mean_roc_auc": float(np.mean([m["roc_auc"] for m in val_metrics])),
    "mean_pr_auc": float(np.mean([m["pr_auc"] for m in val_metrics])),
    "mean_brier": float(np.mean([m["brier"] for m in val_metrics])),
    "csv_used": used_csv.name,
    "window": WINDOW,
    "device": DEVICE,
    "use_outlier": USE_OUTLIER,
    "mean_thr_single": float(np.mean([m["thr_single"] for m in val_metrics])),
    "mean_thr_multi": float(np.mean([m["thr_multi"]  for m in val_metrics])),
}
with open(metrics_out_path, "w") as f:
    json.dump(metrics_summary, f, indent=2)
print(f"✓ Metrics summary saved to: {metrics_out_path}")