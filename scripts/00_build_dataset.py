#!/usr/bin/env python3
# scripts/00_build_dataset.py

"""
Balanced & deduped prompt dataset builder.
- Train: 30k benign + 30k attack (flat & multi-turn, deduped)
- OOD:   attack L5/6 + 1.5x benign, merged txt + jsonl OOD (multi-turn/single-turn), with OOD benign augmentation.
Usage:
    python scripts/00_build_dataset.py
"""

from __future__ import annotations
import csv, json, random, re, sys
from datetime import datetime
from pathlib import Path
from typing import List, Set
import tiktoken
import random
from collections import Counter

import numpy as np
import pandas as pd
from datasets import load_dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

# ───────── CONFIG ─────────
SEED = 42
TARGET_PER_CLASS = 30_000
ENC = tiktoken.get_encoding("cl100k_base")
MAX_TOK = 2048
NEAR_DUPE_THR = .89
OOD_BENIGN_RATIO = 1.5

SRC_OOD_TXT = Path("prompts/attacks/handcrafted_ood.txt")
SRC_OOD_JSONL = Path("prompts/attacks/slm_in_the_wild_ood_20250728-081826.jsonl")
MULTI_TURN_PATHS = [
    Path("prompts/attacks/harmbench_labeled_hb_prompts.json"),
    Path("prompts/attacks/harmbench_labeled_forbidden_prompts.json"),
]
SRC_ATTACK_CSVS = [
    "prompts/attacks/adv_training_behaviors.csv",
    "prompts/attacks/advbench_behaviors.csv",
    "prompts/attacks/harmbench_behaviors_text_all.csv",
    "prompts/attacks/forbidden_question_set_filtered.csv",
]
FN_CSV = Path("prompts/attacks/FN_all_train.csv")

HF_BENIGN = load_dataset("lmsys/chatbot_arena_conversations", split="train")

OUT_DIR = Path("data/raw"); OUT_DIR.mkdir(parents=True, exist_ok=True)
random.seed(SEED)

# ───────── HELPERS ─────────
from helpers import (
    add_char_noise, wrap_long_context, add_base64_like_noise,
    benign_rot13, benign_homoglyph, filter_near_dupes, is_clean_benign
)
from attacks_lib.attacks import (
    apply_all_single, apply_nested, apply_random_chain,
)

def n_tokens(txt: str) -> int:
    return len(ENC.encode(txt, disallowed_special=()))

def truncate_tokens(text: str, max_tokens=2048):
    toks = n_tokens(text)
    if  toks <= max_tokens:
        return text
    dialog_blocks = re.split(r'(?=(user:|assistant:))', text, flags=re.I)
    merged_blocks = []
    i = 0
    # For each block, merge user and assistant turns, single prompts
    while i < len(dialog_blocks):
        if dialog_blocks[i].lower() in ['user:', 'assistant:'] and i+1 < len(dialog_blocks):
            merged_blocks.append(dialog_blocks[i] + dialog_blocks[i+1])
            i += 2
        else:
            merged_blocks.append(dialog_blocks[i])
            i += 1
    # For each block, check if it fits within max_tokens, multi-turn prompts
    while len(merged_blocks) > 1:
        test_text = ''.join(merged_blocks)
        toks = n_tokens(test_text)
        if toks <= max_tokens:
            return test_text
        # Remove the earliest block (assumed to be the least relevant)
        merged_blocks = merged_blocks[1:]

    return ENC.decode(ENC.encode(''.join(merged_blocks), disallowed_special=())[:max_tokens])

def normalize(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip().lower()
    text = re.sub(r"[^\w\s]", "", text)
    return text

def _dedup(rows: List[dict]) -> List[dict]:
    seen = {}
    for r in rows:
        key = f"{r.get('family','')}_{normalize(r['text'])}"
        r["__k"] = key
        seen[key] = r
    return list(seen.values())

def _aug_benign(text: str) -> List[tuple[str, str]]:
    outs = [(text, "raw_benign")]
    if random.random() < .30: outs.append((add_char_noise(text),      "benign_noise"))
    if random.random() < .15: outs.append((wrap_long_context(text),   "benign_ctx"))
    if random.random() < .20: outs.append((add_base64_like_noise(text),"benign_b64"))
    if random.random() < .30: outs.append((benign_rot13(text),        "benign_rot13"))
    if random.random() < .20: outs.append((benign_homoglyph(text),    "benign_homoglyph"))
    return outs

# ───────── 1) BENIGN SEEDS ─────────
print("[INFO] Loading benign seed prompts from lmsys/chatbot_arena_conversations…")

flat_prompts: List[str] = []
multi_turn_prompts: List[str] = []

all_conv = [
    sample["conversation_a"]
    for sample in HF_BENIGN
    if sample["conversation_a"]
       and not sample.get("toxic_chat_tag", {}).get("roberta-large", {}).get("flagged", False)
       and not sample.get("toxic_chat_tag", {}).get("t5-large", {}).get("flagged", False)
]
random.shuffle(all_conv)

half = len(all_conv) // 2

for conv in all_conv[:half]:
    if conv[0]["role"].lower() == "user":
        user_prompt = conv[0]["content"].strip()
        if user_prompt:
            flat_prompts.append(user_prompt)

for conv in all_conv[half:]:
    merged = " ".join([
        f"{turn['role']}: {turn['content'].strip()}"
        for turn in conv if turn.get("content")
    ])
    if merged:
        multi_turn_prompts.append(merged)

# Combine all and shuffle:
all_benign = flat_prompts + multi_turn_prompts
random.shuffle(all_benign)

OOD_BENIGN_MAX = 2000
benign_seeds_ood = all_benign[-OOD_BENIGN_MAX:]
benign_seeds_train = all_benign[:-OOD_BENIGN_MAX]

print(f"[INFO] Collected {len(benign_seeds_train):,} train benign seeds.")
print(f"[INFO] Collected {len(benign_seeds_ood):,} OOD benign seeds.")
# ───────── MULTI TURN ATTACK PROMPT SET ─────────
multi_turn_attack_norms: Set[str] = set()
for mt_path in MULTI_TURN_PATHS:
    if not mt_path.exists():
        continue
    with mt_path.open("r", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except Exception:
            data = [json.loads(line) for line in f]
        for row in data:
            attack = row.get("attack", {})
            prompt = attack.get("prompt", "")
            chain = attack.get("chain", "")
            for var in [prompt, f"{chain}::{prompt}"]:
                if var and n_tokens(var) <= MAX_TOK:
                    multi_turn_attack_norms.add(normalize(var))

# ───────── 3) DATASET BUILD (MULTI-TURN + FLAT) ─────────
print("[INFO] Building training dataset with benign seeds and multi-turn prompts…")
rows: List[dict] = []
for txt in benign_seeds_train:
    fam = hash(txt) & 0xffffffff
    for aug, tag in _aug_benign(txt):
        if n_tokens(aug) > MAX_TOK:
            aug = truncate_tokens(aug, MAX_TOK)
        if n_tokens(aug) <= MAX_TOK:
            rows.append(dict(
                text=aug, label=0, attack_fn=tag,
                attack_level=0, source=txt, family=fam, policy="", phase="train"
            ))

FN_PROMPTS_ALL = []
if FN_CSV.exists():
    df_fn = pd.read_csv(FN_CSV)
    for _, row in df_fn.iterrows():
        txt = row.get("prompt") or row.get("text")
        if txt:
            if n_tokens(txt) > MAX_TOK:
                txt = truncate_tokens(txt, MAX_TOK)
            FN_PROMPTS_ALL.append(txt)
random.shuffle(FN_PROMPTS_ALL)
SPLIT = int(len(FN_PROMPTS_ALL) * 0.7)  # FN's 70% for training and evaluation

FN_PROMPTS_TRAIN = set(normalize(x) for x in FN_PROMPTS_ALL[:SPLIT])
for txt in FN_PROMPTS_TRAIN:
    if not txt or n_tokens(txt) > MAX_TOK:
        txt = truncate_tokens(txt, MAX_TOK)
    fam = hash(txt) & 0xffffffff
    rows.append(dict(
        text=txt, label=1, attack_fn="hard_negative",
        attack_level=6, source="fn_hard_negative",
        family=fam, policy="", phase="train"
    ))

for mt_path in MULTI_TURN_PATHS:
    if not mt_path.exists():
        continue
    with mt_path.open("r", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except Exception:
            data = [json.loads(line) for line in f]
        for row in data:
            convo = row.get("conversation", [])
            merged = " ".join([f"{turn['role']}: {turn['content']}" for turn in convo if turn.get('content')])
            if "response" in row and row["response"]:
                last_assistant = [t['content'] for t in convo[::-1] if t['role'] == 'assistant']
                if not last_assistant or row["response"].strip() != last_assistant[0].strip():
                    merged += f" assistant: {row['response']}"
            if n_tokens(merged) > MAX_TOK:
                merged = truncate_tokens(merged, MAX_TOK)
            if n_tokens(merged) <= MAX_TOK:
                label = row.get("label", 1)
                attack_fn = row.get("data_type", "multi")
                family = hash(merged) & 0xffffffff
                policy = row.get("attack", {}).get("policy", "") if label == 1 else ""
                rows.append(dict(
                    text=merged, label=label, attack_fn=attack_fn,
                    attack_level=6 if label == 1 else 0,
                    family=family, policy=policy, phase="train"
                ))

for csv_path in SRC_ATTACK_CSVS:
    df = pd.read_csv(csv_path).fillna("")
    prompt_col = "question" if "question" in df.columns else "Behavior"
    policy_col = "content_policy_name" if "content_policy_name" in df.columns else "SemanticCategory"
    for _, row in df.iterrows():
        prompt = row[prompt_col]
        policy = row[policy_col] if policy_col in row else ""
        fam = hash(prompt) & 0xffffffff
        variants = [ (prompt, "raw", 1) ]
        variants += list(apply_all_single(prompt))
        variants += list(apply_nested(prompt, depth=2, n=4))
        variants += list(apply_random_chain(prompt, lam=1.2, max_len=4, n=4))
        for var, wrap_name, lvl in variants:
            if n_tokens(var) > MAX_TOK:
                var = truncate_tokens(var, MAX_TOK)
            nvar = normalize(var)
            if nvar in multi_turn_attack_norms:
                continue
            if n_tokens(var) <= MAX_TOK:
                rows.append(dict(
                    text=var, label=1, attack_fn=wrap_name,
                    attack_level=lvl, family=fam, 
                    policy=policy, phase="train"
                ))

print(f"[INFO] {len(rows):,} training rows before dedup")
train = _dedup(rows)
train = filter_near_dupes(train, NEAR_DUPE_THR)
print(f"[INFO] {len(train):,} training rows after deduplication and near-dupe filtering")

keep = lambda lvl: random.random() < {1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 0.5, 6: 1.0}.get(lvl, 1.0)
benign = [r for r in train if r["label"] == 0]
attack = [r for r in train if r["label"] == 1 and keep(r["attack_level"])]
print(f"[INFO] {len(attack):,} attack rows kept after level-ratio sampling")

random.shuffle(benign); random.shuffle(attack)
k = min(len(benign), len(attack), TARGET_PER_CLASS)
train = random.sample(benign, k) + random.sample(attack, k)
random.shuffle(train)

print(
    f"[INFO] {len(train):,} rows after balancing "
    f"(benign pool={len(benign)}, attack pool={len(attack)})"
)

# ───────── 5) OOD BUILD: JSONL + TXT + BENIGN ─────────
print("[INFO] Building OOD dataset with JSONL, TXT, and benign augment…")
ood: List[dict] = []

# 1. JSONL'den OOD Prompts (attack + benign)
if SRC_OOD_JSONL.exists():
    with SRC_OOD_JSONL.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                row = json.loads(line)
            except Exception:
                continue
            convo = row.get("conversation", [])
            merged = " ".join([f"{turn['role']}: {turn['content']}" for turn in convo if turn.get("content")])
            if "response" in row and row["response"]:
                last_assistant = [t['content'] for t in convo[::-1] if t['role'] == 'assistant']
                if not last_assistant or row["response"].strip() != last_assistant[0].strip():
                    merged += f" assistant: {row['response']}"
            if n_tokens(merged) > MAX_TOK:
                merged = truncate_tokens(merged, MAX_TOK)
            if n_tokens(merged) <= MAX_TOK:
                label = row.get("label", 1)
                attack_fn = row.get("attack_type", "")
                phase = "ood_test"
                attack_level = 6 if label == 1 else 0
                policy = row.get("attack", {}).get("policy", "") if label == 1 else ""
                source = row.get("model", "ood")
                family = hash(merged) & 0xffffffff
                ood.append(dict(
                    text=merged, label=label, attack_fn=attack_fn,
                    attack_level=attack_level, source=source,
                    family=family, policy=policy, phase=phase
                ))

# 2. OOD Prompts (attack) from handcrafted txt
if SRC_OOD_TXT.exists():
    with SRC_OOD_TXT.open("r", encoding="utf-8") as f:
        for jb in f:
            jb = jb.strip()
            if n_tokens(jb) > MAX_TOK:
                jb = truncate_tokens(jb, MAX_TOK)
            if not jb or n_tokens(jb) > MAX_TOK:
                continue
            ood.append(dict(
                text=jb, label=1, attack_fn="human", attack_level=5,
                source="handcrafted", family=hash(jb) & 0xffffffff,
                policy="", phase="ood_test"
            ))

# 3. OOD Benign augment 
ood_benigns = [x for x in benign_seeds_ood if is_clean_benign(x) and n_tokens(x) <= MAX_TOK]
random.shuffle(ood_benigns)
for txt in ood_benigns:
    for aug, tag in _aug_benign(txt):
        if n_tokens(aug) > MAX_TOK:
            aug = truncate_tokens(aug, MAX_TOK)
        if n_tokens(aug) <= MAX_TOK:
            ood.append(dict(
                text=aug, label=0, attack_fn=f"{tag}_ood", attack_level=0,
                source=txt, family=hash(aug) & 0xffffffff,
                policy="", phase="ood_test"
            ))

# ============================= 6) OOD DEDUP & BALANCE ==========================
ood = _dedup(ood)
ood = [r for r in ood if normalize(r["text"]) not in FN_PROMPTS_TRAIN]
ood = filter_near_dupes(ood, .87)
print(f"[INFO] {len(ood):,} OOD rows after deduplication and near-dupe filtering")
atk = [r for r in ood if r["label"] == 1]
ben = [r for r in ood if r["label"] == 0]
random.shuffle(ben)
ood = atk + ben[:int(len(atk)*OOD_BENIGN_RATIO)]
random.shuffle(ood)
print(f"[INFO] {len(ood):,} OOD rows after balancing (benign={len(ben)}, attack={len(atk)})")

# ==================== 7) WRITE CSVs ====================================
ts = datetime.now().strftime("%Y%m%d-%H%M%S")
pd.DataFrame(train).drop(columns="__k").to_csv(
    f := OUT_DIR/f"train_{len(train)}_{ts}.csv", index=False, quoting=csv.QUOTE_ALL)
pd.DataFrame(ood).drop(columns="__k").to_csv(
    g := OUT_DIR/f"ood_test_{len(ood)}_{ts}.csv", index=False, quoting=csv.QUOTE_ALL)

print(json.dumps(dict(train_rows=len(train), ood_rows=len(ood),
                      benign_per_class=k, csv_train=f.name, csv_ood=g.name),
                 indent=2))


# ======================= 8) DATASET SUMMARY ====================================
print(f"=== DATASET SUMMARY ===")


print(f"Train: {len(train):,} rows (benign={len([r for r in train if r['label']==0])}, attack={len([r for r in train if r['label']==1])})")
print(f"OOD  : {len(ood):,} rows (benign={len([r for r in ood if r['label']==0])}, attack={len([r for r in ood if r['label']==1])})")

print(f"\n--- Deduplication ---")
print(f"Deduplication ratio (train): {len(rows):,} → {len(train):,} ({100*len(train)/max(1,len(rows)):.2f}%)")

print(f"\n--- Augmentation Breakdown (train benign) ---")
aug_counts = Counter([r['attack_fn'] for r in train if r['label']==0])
for k, v in aug_counts.most_common():
    print(f"   {k:>20}: {v}")

print(f"\n--- Prompt Diversity ---")
unique_train = len(set(normalize(r['text']) for r in train))
print(f"Unique prompt ratio (train): {unique_train}/{len(train)} ({100*unique_train/len(train):.2f}%)")

unique_ood = len(set(normalize(r['text']) for r in ood))
print(f"Unique prompt ratio (OOD): {unique_ood}/{len(ood)} ({100*unique_ood/len(ood):.2f}%)")

print(f"\n--- Toxic Filtering ---")
toxic_filtered = len(HF_BENIGN) - len(all_conv)
print(f"Toxic-filtered samples removed: {toxic_filtered}")

print(f"\n--- Prompt Length Statistics ---")
benign_lengths = [len(r["text"].split()) for r in train if r["label"] == 0]
attack_lengths = [len(r["text"].split()) for r in train if r["label"] == 1]
print(f"Benign prompt avg. length (words): {np.mean(benign_lengths):.2f} ± {np.std(benign_lengths):.2f}")
print(f"Attack prompt avg. length (words): {np.mean(attack_lengths):.2f} ± {np.std(attack_lengths):.2f}")

print(f"\n--- Label Distribution (Full) ---")
label_counts = Counter([r["label"] for r in train])
for k, v in label_counts.items():
    print(f"   Label={k}: {v}")

print(f"\n--- Top 10 Most Common Prompts (train, normalized) ---")
top_prompts = Counter([normalize(r["text"]) for r in train]).most_common(10)
for prompt, count in top_prompts:
    print(f"   ({count:>4}x): {prompt[:70]}{'...' if len(prompt)>70 else ''}")

print(f"\n--- Attack Type Breakdown (train attacks) ---")
attack_fn_counts = Counter([r['attack_fn'] for r in train if r['label'] == 1])
for k, v in attack_fn_counts.most_common(10):
    print(f"   {k:>20}: {v}")

print(f"\n--- Attack Level Distribution (train attacks) ---") 
attack_level_counts = Counter([r['attack_level'] for r in train])
for k, v in sorted(attack_level_counts.items()):
    print(f"   Level {k}: {v}")