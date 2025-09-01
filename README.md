# Ensemble‑Based Prompt‑Injection Detection for Small Language Models (SLMs)

> Reproducible code and artifacts for the thesis defense. Builds datasets, generates sanitized attacks, trains a stacked ensemble detector, and evaluates single‑ vs. multi‑turn robustness.

<p align="center">
  <img alt="pipeline" src="https://img.shields.io/badge/pipeline-ETL%E2%86%92attacks%E2%86%92train%E2%86%92eval-4c8eda" />
  <img alt="python" src="https://img.shields.io/badge/python-3.10%2B-blue" />
  <img alt="license" src="https://img.shields.io/badge/license-MIT-green" />
</p>

---

## Table of Contents

* [Overview](#overview)
* [Repository Structure](#repository-structure)
* [Setup](#setup)
* [Quick Start](#quick-start)
* [Method](#method)
* [Reproducing Headline Results](#reproducing-headline-results)
* [Experiments & Ablations](#experiments--ablations)
* [CLI Reference](#cli-reference)
* [Evaluation Notes](#evaluation-notes)
* [Ethical Use](#ethical-use)
* [Roadmap](#roadmap)
* [Citing](#citing)
* [License](#license)
* [Acknowledgments](#acknowledgments)

---

## Overview

This repository contains the end‑to‑end code and configurations for the thesis **“Ensemble‑Based Prompt Injection Detection for Small Language Models (SLMs)”**. The detector combines three complementary channels and a stacked meta‑learner with **turn‑type‑aware dual thresholds** (separate decision thresholds for **single‑turn** and **multi‑turn** inputs):

1. **BoW channel** — TF‑IDF bag‑of‑words + Logistic Regression.
2. **Embedding channel** — Distilled sentence embeddings (e.g., SBERT/DistilRoBERTa) + Gradient‑Boosted Trees.
3. **Feature‑engineering channel** — Hand‑crafted lexical/semantic/statistical indicators + Gradient Boosting.

**Cross‑validation (5‑fold) summary** of the final ensemble:

* ROC–AUC: **0.9990** (mean)
* PR–AUC: **0.9990** (mean)
* F1: **0.9903** (mean)
* Brier: **0.0090** (mean)

A held‑out **OOD** split quantifies domain shift and includes separate single‑ vs. multi‑turn evaluation.

> **Safety note:** All shipped “attack” generators are **sanitized**; they stress‑test detectors with safe templates and obfuscation patterns only.

---

## Repository Structure

> Adjust paths if your clone differs. Script names are stable.

```
.
├── scripts/00_build_dataset.py        # Dataset creation: benign/attack pools, balancing, de‑duplication, stats
├── scripts/01_attack_models.py        # Safe attack generators & transformations (no harmful payloads)
├── scripts/02_evaluate_detector.py    # Train channels + stacker, CV, OOD eval, threshold selection
├── detector/train.py                   # (If used) Stand‑alone trainer for channel(s)/stacker
├── detector/detector.py                # Ensemble head, calibration, thresholding, utilities
├── detector/features.py                # Hand‑crafted feature extractors & indicators
├── detector/helpers.py                 # Common utilities (IO, seeds, logging, metrics helpers)
├── prompts/                   # Seed benign prompts and sanitized adversarial patterns (optional)
├── data/                      # Auto‑created: datasets and intermediate outputs
├── results/                   # Auto‑created: models, metrics, figures
└── README.md
```

---

## Setup

### Option A - Conda (Apple Silicon / CPU)

Recommended for macOS on M-series.

```bash
conda env create -f environment-mac.yml
conda activate thesis-ml
```

### Option B — Conda (CUDA GPU)

For Linux/Windows with an NVIDIA GPU (CUDA 12.4 toolchain).

```bash
conda env create -f environment-cuda.yml
conda activate thesis-ml
```

Make sure your NVIDIA driver supports CUDA 12.x. If you have multiple
CUDA versions, prefer a clean, project-specific env.

### Reproducibility

```bash
export SEED=42
```

All scripts set fixed `random_state`/seeds internally.

---

## Quick Start

### 1) Build datasets

Constructs balanced train/validation splits and an out-of-distribution (OOD) evaluation split.
The script loads benign and adversarial prompt pools, applies optional transformations, removes near-duplicates (via embedding similarity), and logs dataset statistics.

```bash
python scripts/00_build_dataset.py 
```

**Outputs** (default):
* `data/train.csv` — balanced & de‑duplicated
* `data/ood.csv` — domain‑shifted/OOD benign + adversarial mix

### 2) (Optional) Generate additional sanitized attack conversations 

Runs attack wrappers against small language models to create new adversarial prompts (e.g., role-assumption, policy-override, obfuscation).
Useful for augmenting the dataset with more diverse jailbreak attempts.
Builds conversation datasets for both classes:
	•	Adversarial: jailbreak prompts (single-turn and multi-turn, with history).
	•	Benign: multi-turn benign chats used as controls.

```bash
python scripts/01_attack_models.py 
```

### 3) Train the ensemble 

Trains the three base detection channels and fits the stacked ensemble head.
Cross-validation is performed for channel weights and hyperparameters.

```bash
python detector/train.py
```

### 4) Evaluate the ensemble

This performs CV + OOD evaluation with **dual thresholds**.

```bash
python scripts/02_evaluate_detector.py
```

Artifacts:

* `results/evaluation-ensemble-dualthr/metrics.json` — fold metrics, thresholds, calibration stats
* `results/evaluation-ensemble-dualthr/preds.csv` — prediction of OOD evaluation 
* `results/evaluation-ensemble-dualthr/resource_stats.json` — resource stats 

> CPU‑only is supported; embedding extraction is the slowest step.

---

## Method

**Channels**

* **BoW:** word/char TF‑IDF (configurable n‑grams), Logistic Regression (ℓ2, class weighting).
* **Embedding:** distilled sentence embeddings; tree‑based head (XGBoost/LightGBM) with early‑stopping on PR–AUC.
* **Feature‑engineering:** lexical cues (policy‑override, role‑assumption), obfuscation signals (spacing/homoglyph), and simple statistics.

**Ensemble Head**

* Stacked meta‑learner over **calibrated** channel probabilities (Platt/Isotonic; auto‑selected via CV Brier).
* **Turn‑type dual thresholds:** optimize separately for single‑turn vs. multi‑turn. Default rule: maximize F1. Alternatives: target‑recall, $F_\beta$.

**Reported metrics**
We report Accuracy, Precision, Recall, F1, ROC–AUC, PR–AUC, Brier, and TP/FP/FN/TN per split.

---

## Reproducing Headline Results

**Cross‑validation (5‑fold) mean**

```
ROC–AUC  : 0.9990
PR–AUC   : 0.9990
F1       : 0.9903
Brier    : 0.0090
thr_single ≈ 0.40
thr_multi  ≈ 0.66
```

**Held‑out OOD (single vs. multi‑turn)**

```
turn     n     precision   recall     f1        tp    fp   fn    tn
single   3221  0.9525      0.5837     0.7238    401   20   286   2514
multi    2466  0.9904      0.9125     0.9499   1449   14   139    864
overall  5687  0.9820      0.8132     0.8896   1850   34   425   3378
```

> Thresholds selected with the default F1 rule per turn‑type. Use `--threshold-rule target-recall --recall 0.95` to bias toward fewer false negatives.

---

## Experiments & Ablations

* **Thresholding strategies:** F1 (default), target‑recall (e.g., ≥0.95), or $F_\beta$ with $\beta>1$ for high‑recall ops.
* **Channel ablations:** disable one channel at a time to quantify marginal contribution to ROC–AUC/PR–AUC and Brier.
* **Calibration:** compare Platt vs. Isotonic; report Brier and reliability diagrams.
* **Multi‑turn windowing:** vary history window (e.g., `--window 1/3/5`) and observe precision/recall changes on OOD.
* **Domain shift:** evaluate on topical OOD benign sets to inspect false positives on long/context‑heavy benign prompts.

---

## CLI Reference

```bash
python scripts/00_build_dataset.py 

python scripts/01_attack_models.py 

python scripts/02_evaluate_detector.py

python detector/train.py 
```

---

## Evaluation Notes

* **Common FN mode:** subtle single‑turn prompts that mask intent; multi‑turn coercion lacking explicit override cues.
* **Common FP mode:** obfuscated benign text (unicode variants, spacing tricks), long benign context with code blocks.
* **Mitigations:** adjust per‑turn thresholds, favor target‑recall in safety‑critical deployments, and add OOD benign data.

---

## Ethical Use

This repository is strictly for **defensive research**. No harmful payloads are included. Attack modules rely on sanitized templates to evaluate detection and robustness. Do **not** use this code to create or distribute harmful content.

---

## Citing


```bibtex
@thesis{taskin2025ensemble_jbd_slm,
  title  = {Ensemble-Based Prompt Injection Detection for Small Language Models},
  author = {Taskin, Can Huseyin},
  school = {IU International University of Applied Sciences},
  year   = {2025}
}
```

---

## License

Released under the **MIT License**. See `LICENSE` for details.

---

## Acknowledgments

Supervision by **Dr. Nghia Duong‑Trung**. Thanks to the open‑source communities behind **Sentence‑Transformers**, **XGBoost/LightGBM**, **scikit‑learn**, and related tooling.
