#!/usr/bin/env python3
"""
Evaluate the JailbreakDetector model on a given CSV file (dual-threshold aware).

Run:
  # Mac'te önerilen:
  DISABLE_MPS=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 VECLIB_MAXIMUM_THREADS=4 \
  SBER_BATCH=32 DET_CHUNK=256 \
  python scripts/02_evaluate_detector.py data/raw/ood_test.csv

Outputs:
  • results/evaluation-<timestamp>/metrics.json
  • results/evaluation-<timestamp>/preds.csv
  • results/evaluation-<timestamp>/resource_stats.json
"""
from __future__ import annotations
import os, sys, json, time
from pathlib import Path
from collections import defaultdict
from datetime import datetime

import psutil
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    confusion_matrix, classification_report, brier_score_loss,
)

# (İlk satırlarda thread limitlerini set etmek iyi)
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "4")

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from detector.detector import JailbreakDetector
from detector.helpers import is_single_turn_batch, classify_turn_type

RAW_DIR = ROOT / "data" / "raw"

# ---------- CSV seçimi ----------
if len(sys.argv) > 1:
    CSV_PATH = Path(sys.argv[1])
else:
    cands = sorted(RAW_DIR.glob("ood_test_*.csv"))
    if not cands:
        sys.exit("[ERR] no CSV like data/raw/ood_test_*.csv found")
    CSV_PATH = cands[-1]
print(f"[INFO] evaluating → {CSV_PATH.relative_to(ROOT) if CSV_PATH.is_absolute() else CSV_PATH}")

# ---------- Veri ----------
df      = pd.read_csv(CSV_PATH)
texts   = df["text"].astype(str).tolist()
y_true  = df["label"].to_numpy(dtype=int)
atk_fn  = df.get("attack_fn",    pd.Series(["?"]*len(df))).astype(str).tolist()
atk_lv  = df.get("attack_level", pd.Series([-1]*len(df))).astype(int).tolist()

# ---------- Model ----------
chunk_env = int(os.getenv("DET_CHUNK", "0")) or None
fast_bow  = os.getenv("FAST_ONLY_BOW", "0") == "1"
det = JailbreakDetector(chunk_size=chunk_env, fast_only_bow=fast_bow)

THR_SINGLE = float(getattr(det, "threshold_single", 0.15))
THR_MULTI  = float(getattr(det, "threshold_multi",  0.5))
print(f"[INFO] thresholds → single: {THR_SINGLE}, multi: {THR_MULTI}")
is_single_mask = det.is_single_turn_batch(texts).astype(int)
turn_type = np.where(is_single_mask == 1, "single", "multi")
print("single_count =", int(is_single_mask.sum()), "/", len(texts))

# ---------- Tahmin ----------
t0 = time.perf_counter()
y_prob = det.predict_proba(texts)  
t_infer = time.perf_counter() - t0

thr_used = np.where(is_single_mask == 1, THR_SINGLE, THR_MULTI).astype("float32")
y_pred   = (y_prob >= thr_used).astype(int)

tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

# ---------- Metrikler ----------
roc   = roc_auc_score(y_true, y_prob)
prc   = average_precision_score(y_true, y_prob)
f1    = 2 * tp / max(1, (2 * tp + fp + fn))
acc   = (tp + tn) / (tp + fp + fn + tn)
brier = brier_score_loss(y_true, y_prob)

model_metrics = dict(
    CSV        = CSV_PATH.name,
    Samples    = int(len(df)),
    Threshold_single = float(THR_SINGLE),
    Threshold_multi  = float(THR_MULTI),
    ROC_AUC    = float(roc),
    PR_AUC     = float(prc),
    Accuracy   = float(acc),
    F1         = float(f1),
    Recall     = float(tp / max(1, tp + fn)),
    Precision  = float(tp / max(1, tp + fp)),
    Brier      = float(brier),
    TP = int(tp), FP = int(fp), FN = int(fn), TN = int(tn),
)

w = max(len(k) for k in model_metrics)
print("\n=== headline metrics ===")
for k, v in model_metrics.items():
    print(f"{k:<{w}} : {v:.4f}" if isinstance(v, float) else f"{k:<{w}} : {v}")

# ---------- Kırılımlar ----------
def grouped(keys, title: str):
    bucket = defaultdict(list)
    for yt, yp, k in zip(y_true, y_pred, keys):
        bucket[k].append((yt, yp))
    rows = []
    for k, pairs in bucket.items():
        ys = np.fromiter((p[0] for p in pairs), dtype=int)
        ps = np.fromiter((p[1] for p in pairs), dtype=int)
        tp = ((ys == 1) & (ps == 1)).sum()
        fp = ((ys == 0) & (ps == 1)).sum()
        tn = ((ys == 0) & (ps == 0)).sum()
        fn = ((ys == 1) & (ps == 0)).sum()
        tpr = tp / (tp + fn) if tp + fn else np.nan
        fpr = fp / (fp + tn) if fp + tn else np.nan
        prec = tp / (tp + fp) if tp + fp else np.nan
        rows.append(dict(group=k, support=len(ys), TPR=tpr, FPR=fpr, Precision=prec))
    print(f"\n[{title} breakdown]")
    print(pd.DataFrame(rows).sort_values("group").to_string(index=False))
    return rows

attack_metrics = grouped(atk_fn, "Attack-fn")
level_metrics  = grouped(atk_lv, "Level")
turn_metrics   = grouped(turn_type, "Turn-type")

print("\n", classification_report(y_true, y_pred, digits=3))

# ---------- Kaydet ----------
timestamp = datetime.now().strftime("%Y%m%d_%H%M")
out_dir = Path("results") / f"evaluation-ensemble-dualthr-{timestamp}"
out_dir.mkdir(exist_ok=True, parents=True)

def convert_numpy_types(obj):
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert_numpy_types(i) for i in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    return obj

metrics_dict = {
    "model": model_metrics,
    "attack_types": attack_metrics,
    "attack_levels": level_metrics,
    "turn_types": turn_metrics,
    "threshold_single": THR_SINGLE,
    "threshold_multi": THR_MULTI,
    "csv": CSV_PATH.name,
    "evaluation_time": datetime.now().isoformat(),
}
with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
    json.dump(convert_numpy_types(metrics_dict), f, indent=2)

df_out = pd.DataFrame({
    "id"           : df.get("id", pd.Series(range(len(df)))),
    "prompt"       : df["text"],
    "true"         : y_true,
    "prob"         : y_prob,
    "pred"         : y_pred,
    "turn_type"    : turn_type,
    "thr_used"     : thr_used,
    "margin"       : y_prob - thr_used,
    "attack_group" : atk_fn,
    "attack_level" : atk_lv,
})
df_out.to_csv(out_dir / "preds.csv", index=False, encoding="utf-8")
print(f"\n✓  {out_dir/'metrics.json'} saved")
print(f"✓  {out_dir/'preds.csv'} saved")

# ---------- Kaynak & hız ----------
BATCH_SIZE = min(32, len(texts))
process = psutil.Process()
start_mem = process.memory_info().rss / 1e6  # MB
start_cpu = process.cpu_percent(interval=None)

t0_single = time.perf_counter(); _ = det.predict_proba([texts[0]]); t_single = time.perf_counter() - t0_single
t0_batch  = time.perf_counter(); _ = det.predict_proba(texts[:BATCH_SIZE]); t_batch  = time.perf_counter() - t0_batch

end_mem = process.memory_info().rss / 1e6
end_cpu = process.cpu_percent(interval=None)

print("\n=== Inference Resource & Speed ===")
print(f"Total inference time      : {t_infer:.3f} s   ({len(texts)} samples)")
print(f"Per-sample average        : {t_infer/len(texts):.4f} s")
print(f"Single prompt latency     : {t_single:.4f} s")
print(f"Batch {BATCH_SIZE} latency: {t_batch:.4f} s ({t_batch/BATCH_SIZE:.4f} s/sample)")
print(f"RAM usage (start → end)   : {start_mem:.1f} → {end_mem:.1f} MB (Δ {end_mem-start_mem:.1f} MB)")
print(f"CPU usage                 : {end_cpu:.1f} %")

resource_stats = {
    "total_inference_time_sec": float(t_infer),
    "per_sample_avg_sec": float(t_infer/len(texts)),
    "single_prompt_latency_sec": float(t_single),
    "batch_latency_sec": float(t_batch),
    "batch_size": int(BATCH_SIZE),
    "ram_start_mb": float(start_mem),
    "ram_end_mb": float(end_mem),
    "ram_delta_mb": float(end_mem-start_mem),
    "cpu_usage_percent": float(end_cpu),
}
with open(out_dir / "resource_stats.json", "w", encoding="utf-8") as f:
    json.dump(resource_stats, f, indent=2)
print(f"✓  {out_dir/'resource_stats.json'} saved")