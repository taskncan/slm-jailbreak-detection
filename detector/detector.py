# detector/detector.py
from __future__ import annotations
from pathlib import Path
from typing import Iterable, List
import os
import re
import joblib
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from scipy import sparse as sp

from detector.features import build_feature_vector
from detector.helpers import (
    is_single_turn_text, is_single_turn_batch, extract_last_user_and_history
)

# ---------------- CuPy (yalnızca CUDA varsa) ----------------
try:
    import cupy as cp
    try:
        _cupy_ok = cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        _cupy_ok = False
except Exception:
    cp = None
    _cupy_ok = False

def to_gpu(x):
    return cp.asarray(x) if (cp is not None and _cupy_ok) else x

def to_cpu(x):
    return cp.asnumpy(x) if (cp is not None and _cupy_ok and isinstance(x, cp.ndarray)) else x

# ---------------- Device Selection ----------------
def _select_device() -> str:
    if os.getenv("DISABLE_MPS", "0") == "1":
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

DEVICE = _select_device()
print(f"[INFO] Detected device: {DEVICE}")

# ---------------- Artifact Loading ----------------
ROOT   = Path(__file__).resolve().parents[1]
REGDIR = ROOT / "detector" / "model_registry" / "models"
MODEL  = "model-ensemble-dualthr.pkl"
ART    = joblib.load(sorted(REGDIR.glob(MODEL))[-1])

_vec_c       = ART["vec_char"]
_vec_w       = ART["vec_word"]
_bow_lr      = ART["bow_lr"]
_xgb_emb     = ART["xgb_emb"]
_xgb_feat_s  = ART["xgb_feat_s"]
_ensemble    = ART["ensemble_clf"]
_sbert_name  = ART["sbert_name"]

benign_c = np.asarray(ART["benign_c"], dtype="float32").ravel()
attack_c = np.asarray(ART["attack_c"], dtype="float32").ravel()
WINDOW   = int(ART.get("window", 3))

# Thresholds
THR_SINGLE = 0.15
THR_MULTI  = float(ART.get("best_threshold_multi", 0.5))

# Outlier payload (optional)
OUTLIER_USED = bool(ART.get("use_outlier", False))
_OUT_MU      = ART.get("outlier_mu", None)
_OUT_PREC    = ART.get("outlier_precision", None)
if OUTLIER_USED and _OUT_MU is not None and _OUT_PREC is not None:
    OUT_MU   = np.asarray(_OUT_MU, dtype="float32").ravel()
    OUT_PREC = np.asarray(_OUT_PREC, dtype="float32")
else:
    OUTLIER_USED = False
    OUT_MU = OUT_PREC = None

# ---------------- SBERT (lazy) ----------------
_sbert = None
_SBER_BATCH = int(os.getenv("SBER_BATCH", "32"))  # Mac'te küçük tut
def _get_sbert():
    global _sbert
    if _sbert is None:
        _sbert = SentenceTransformer(_sbert_name, device=DEVICE)
    return _sbert

def _prepare_prompt_and_history(texts: List[str], window: int):
    """Last-user prompt ve history çıkar, SBERT ile encode et."""
    sbert = _get_sbert()
    prompts_txt, histories_txt = [], []
    for t in texts:
        p, h = extract_last_user_and_history(t, window)
        prompts_txt.append(p)
        histories_txt.append(h)
    with torch.inference_mode():
        E_prompt  = sbert.encode(prompts_txt,  batch_size=_SBER_BATCH, normalize_embeddings=True).astype("float32")
        E_history = sbert.encode(histories_txt, batch_size=_SBER_BATCH, normalize_embeddings=True).astype("float32")
    return prompts_txt, histories_txt, E_prompt, E_history

# ---------------- Outlier scores ----------------
def _outlier_scores(emb_prompt: np.ndarray) -> np.ndarray:
    if not OUTLIER_USED or OUT_MU is None or OUT_PREC is None:
        return np.zeros(emb_prompt.shape[0], dtype="float32")
    d = emb_prompt.astype("float32") - OUT_MU
    return np.einsum("ij,jk,ik->i", d, OUT_PREC, d).astype("float32")

# ---------------- Feature Extractors ----------------
def sp_hstack_safe(Xc, Xw):
    return sp.hstack([Xc, Xw]).tocsr()

def _bow_lr_prob(batch_texts: List[str]) -> np.ndarray:
    Xc = _vec_c.transform(batch_texts)
    Xw = _vec_w.transform(batch_texts)
    X_bow = sp_hstack_safe(Xc, Xw)
    return _bow_lr.predict_proba(X_bow)[:, 1].astype("float32")

def _sbert_emb_prob(E_prompt: np.ndarray) -> np.ndarray:
    probs = _xgb_emb.predict_proba(to_gpu(E_prompt))[:, 1]
    return to_cpu(probs).astype("float32")

def _sbert_feat_prob(raw_texts, E_prompt, E_history, prompts_txt, histories_txt, out_scores):
    H = np.vstack([
        build_feature_vector(
            txt, emb_p, benign_c, attack_c, emb_h,
            ptxt,
            ([htxt] if isinstance(htxt, str) and htxt else (htxt if isinstance(htxt, list) else [])),
            outlier_score=float(out_scores[i]),
        )
        for i, (txt, emb_p, emb_h, ptxt, htxt) in enumerate(
            zip(raw_texts, E_prompt, E_history, prompts_txt, histories_txt)
        )
    ]).astype("float32")
    probs = _xgb_feat_s.predict_proba(to_gpu(H))[:, 1]
    return to_cpu(probs).astype("float32")

# ---------------- Public API ----------------
class JailbreakDetector:
    """
    Dual-threshold inference:
      - single-turn → THR_SINGLE
      - multi-turn  → THR_MULTI

    Performance options:
      - chunk_size: splits large lists into chunks for lower RAM usage
      - fast_only_bow: uses only the BOW-LR path (very fast, lower accuracy)
    """
    def __init__(self, chunk_size: int | None = None, fast_only_bow: bool = False):
        self.threshold_single: float = THR_SINGLE
        self.threshold_multi: float  = THR_MULTI
        self.window: int = WINDOW
        self.chunk_size = chunk_size or (int(os.getenv("DET_CHUNK", "0")) or None)
        self.fast_only_bow = fast_only_bow or (os.getenv("FAST_ONLY_BOW", "0") == "1")

    def is_single_turn_batch(self, texts: Iterable[str]) -> np.ndarray:
        return is_single_turn_batch(list(texts))

    def _predict_block(self, texts: List[str]) -> np.ndarray:
        p_bow = _bow_lr_prob(texts)
        pr_txt, hi_txt, E_p, E_h = _prepare_prompt_and_history(texts, self.window)
        out_scores = _outlier_scores(E_p)
        p_emb  = _sbert_emb_prob(E_p)
        p_feat = _sbert_feat_prob(texts, E_p, E_h, pr_txt, hi_txt, out_scores)
        P = np.column_stack([p_bow, p_emb, p_feat]).astype("float32")
        probs = _ensemble.predict_proba(to_gpu(P))[:, 1]
        return to_cpu(probs).astype("float32")

    def predict_proba(self, texts: Iterable[str]) -> np.ndarray:
        texts = list(texts)
        if not self.chunk_size or len(texts) <= self.chunk_size:
            return self._predict_block(texts)
        out = np.empty(len(texts), dtype="float32")
        cs = self.chunk_size
        for i in range(0, len(texts), cs):
            j = min(i + cs, len(texts))
            out[i:j] = self._predict_block(texts[i:j])
        return out

    def predict(self, texts: Iterable[str]) -> np.ndarray:
        texts = list(texts)
        probs = self.predict_proba(texts)
        mask_single = is_single_turn_batch(texts)
        thr = np.where(mask_single, self.threshold_single, self.threshold_multi)
        return (probs >= thr).astype(int)

    def predict_with_details(self, texts: Iterable[str]):
        texts = list(texts)
        probs = self.predict_proba(texts)
        mask_single = is_single_turn_batch(texts)
        thr = np.where(mask_single, self.threshold_single, self.threshold_multi)
        preds = (probs >= thr).astype(int)
        return {
            "probs": probs,
            "preds": preds,
            "mask_single": mask_single,
            "thr_used": thr.astype("float32"),
            "thr_single": float(self.threshold_single),
            "thr_multi": float(self.threshold_multi),
            "outlier_used": OUTLIER_USED,
        }