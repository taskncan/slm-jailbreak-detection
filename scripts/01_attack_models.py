import os, sys, json, time, random
from pathlib import Path
from datetime import datetime
import pandas as pd
from collections import defaultdict
import tiktoken
from datasets import load_dataset
import re

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
from helpers import sha256, start_model, stop_model, query_chat
from attacks_lib.attacks import apply_all_single, apply_nested, apply_random_chain

# ------------------- Configurable Settings ------------------------
SLM_TAGS = [
    "gemma3:1b", 
    "moondream",
    "llama3.2:1b", 
    "phi4-mini", 
    "gemma3", 
    "mistral", 
    "codellama", 
    "llama2-uncensored",
    "granite3.3:2b", 
    "deepseek-r1:1.5b", 
]

MAX_TOKEN_LIMIT = 1024
NUM_PREDICT_JB = 256
NUM_PREDICT_BENIGN = 128
NUM_BENIGN_HISTORY = (2, 5)
NUM_BENIGN_ONLY   = 100
NUM_ATTACK_MULTI  = 500
NUM_ATTACK_SINGLE = 500

# ------------------- Tokenizer ------------------------
ENC = tiktoken.get_encoding("cl100k_base")
def convo_token_count(convo):
    merged = " ".join([t["content"] for t in convo if t.get("content")])
    return len(ENC.encode(merged, disallowed_special=()))

def token_length(text):
    return len(ENC.encode(str(text), disallowed_special=()))

# ========== GENERIC POOL GENERATION FUNCTION =============

def generate_attack_variants(
    csv_path, 
    prompt_col="question", 
    policy_col="", 
    sampling_ratio=None, 
    wrap_n=4, 
    max_token=256
):
    if sampling_ratio is None:
        sampling_ratio = {
            "base": 1,
            "single-wrapped": 2,
            "nested-wrapped": 2,
            "random-chain-wrapped": 2,
        }

    df = pd.read_csv(csv_path)
    uniq_attack_variants_by_type = defaultdict(list)
    seen = set()
    for _, row0 in df.iterrows():
        attack = row0[prompt_col]
        policy_name = row0[policy_col] if policy_col else "default"

        # BASE
        row = {"prompt": attack, "type": "base", "chain": "", "level": 0, "policy": policy_name}
        key = (row["prompt"], row["chain"])
        if key not in seen:
            uniq_attack_variants_by_type["base"].append(row)
            seen.add(key)

        # SINGLE-WRAPPED
        for _ in range(wrap_n):
            for v, name, lvl in apply_all_single(attack):
                row = {"prompt": v, "type": "single-wrapped", "chain": name, "level": lvl, "policy": policy_name}
                key = (row["prompt"], row["chain"])
                if key not in seen and len(str(row["prompt"])) > 5 and token_length(row["prompt"]) <= max_token:
                    uniq_attack_variants_by_type["single-wrapped"].append(row)
                    seen.add(key)

        # NESTED-WRAPPED
        for _ in range(wrap_n):
            for v, name, lvl in apply_nested(attack, depth=2, n=wrap_n):
                row = {"prompt": v, "type": "nested-wrapped", "chain": name, "level": lvl, "policy": policy_name}
                key = (row["prompt"], row["chain"])
                if key not in seen and len(str(row["prompt"])) > 5 and token_length(row["prompt"]) <= max_token:
                    uniq_attack_variants_by_type["nested-wrapped"].append(row)
                    seen.add(key)

        # RANDOM-CHAIN-WRAPPED
        for _ in range(wrap_n):
            for v, name, lvl in apply_random_chain(attack, lam=1.2, max_len=4, n=wrap_n):
                row = {"prompt": v, "type": "random-chain-wrapped", "chain": name, "level": lvl, "policy": policy_name}
                key = (row["prompt"], row["chain"])
                if key not in seen and len(str(row["prompt"])) > 5 and token_length(row["prompt"]) <= max_token:
                    uniq_attack_variants_by_type["random-chain-wrapped"].append(row)
                    seen.add(key)

    # Balanced sampling:
    min_base = len(uniq_attack_variants_by_type["base"])
    sampled_attacks = []
    for typ, lst in uniq_attack_variants_by_type.items():
        ratio = sampling_ratio.get(typ, 1)
        n_sample = min(len(lst), min_base * ratio)
        random.shuffle(lst)
        print(f"Taking {n_sample} from type {typ} (ratio {ratio}x base, pool {len(lst)})")
        sampled_attacks.extend(lst[:n_sample])
    print(f"Total balanced attack pool: {len(sampled_attacks)}")
    return sampled_attacks

# ========== BENIGN POOL ===========
HF_BENIGN = [(", ["train", "test"], "prompt")]
BLACKLIST_KEYWORDS_PATH = Path("prompts") / "blacklist_keywords.txt"
with BLACKLIST_KEYWORDS_PATH.open(encoding="utf-8") as f:
    BLACKLIST_KEYWORDS = [line.strip() for line in f if line.strip()]
def is_clean_benign(prompt):
    prompt = prompt.lower()
    for kw in BLACKLIST_KEYWORDS:
        pattern = r'\b' + re.escape(kw.lower()) + r'\b'
        if re.search(pattern, prompt): return False
        if ' ' in kw and re.search(re.escape(kw.lower()).replace(' ', r'[\s_\-]+'), prompt): return False
    return True
def get_benign_prompts(n=124):
    benign_pool = []
    for ds_name, splits, prompt_col in HF_BENIGN:
        for split in splits:
            ds = load_dataset(ds_name, split=split)
            for row in ds:
                prompt = row[prompt_col].strip()
                if not (10 < len(prompt) < 280): continue
                benign_pool.append(prompt)
                if len(benign_pool) >= n: break
            if len(benign_pool) >= n: break
        if len(benign_pool) >= n: break
    benign_pool = list(dict.fromkeys(benign_pool))  
    return [p for p in benign_pool if is_clean_benign(p)]

# ========== MAIN ===============
PROMPTS_CSV_FORB = Path("prompts") / "attacks" / 
out_dir = Path("results"); out_dir.mkdir(exist_ok=True)
out_path = out_dir / f"slm_multi_turn_{datetime.now():%Y%m%d-%H%M%S}.jsonl"

benign_pool = get_benign_prompts(n=20000)
print(f"Filtered benign prompts: {len(benign_pool)}")

multi_intro = [
    "By the way,", "Also,", "Additionally,", "One more thing,", "Just to add,",
    "Furthermore,", "In addition,", "Moreover,", "On a related note,", "Lastly,",
    "Ayrıca,", "Ek olarak,", "Bununla birlikte,", "Ayrıca şunu da belirtmek isterim,",
    "Bunun dışında,", "Ayrıca şunu da eklemek istiyorum,", "Además,", "Por cierto,",
    "Asimismo,", "Una cosa más,", "Cabe añadir,", "Por otra parte,", "En relación a esto,",
    "D'ailleurs,", "De plus,", "En outre,", "Par ailleurs,", "Une chose de plus,",
    "Pour ajouter,", "Sur ce sujet,", "Übrigens,", "Außerdem,", "Zusätzlich,", "Noch etwas,",
    "Darüber hinaus,", "Im Zusammenhang damit,", "Abschließend,", "Inoltre,", "A proposito,",
    "In aggiunta,", "Un'altra cosa,", "Per aggiungere,", "In relazione a questo,", "Infine,",
    "ちなみに、", "また、", "さらに、", "加えて、", "それに、", "最後に、",
    "顺便说一下，", "另外，", "此外，", "还有一点，", "最后，",
    "Кстати,", "Также,", "Дополнительно,", "Кроме того,", "В заключение,",
]

sampled_attacks = generate_attack_variants(
    PROMPTS_CSV_FORB, 
    prompt_col="question", 
    policy_col="",
    sampling_ratio={"base": 1, "single-wrapped": 2, "nested-wrapped": 2, "random-chain-wrapped": 2},
    wrap_n=4,
    max_token=256
)
with open(out_dir / "sampled_attacks.jsonl", "w", encoding="utf-8") as f:
    for row in sampled_attacks:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

def run_attack_turns(fp, tag, pool, num, multi_intro, label, data_type, attack_type, is_multi=True):
    for attack in pool[:num]:
        if not attack:
            break
        if is_multi:
            n_history = random.randint(*NUM_BENIGN_HISTORY)
            if len(local_benign_pool) < n_history:
                refill_and_shuffle_benign()
                if len(local_benign_pool) < n_history:
                    continue
            used_benigns = random.sample(local_benign_pool, n_history)
            [local_benign_pool.remove(p) for p in used_benigns if p in local_benign_pool]
            convo = []
            for prompt in used_benigns:
                convo.append({"role": "user", "content": prompt})
                reply, lat, code = query_chat(convo, tag, NUM_PREDICT_JB)
                convo.append({"role": "assistant", "content": reply or "[NO_RESPONSE]"})
            intro = random.choice(multi_intro)
            user_attack = f"{intro} {attack['prompt']}"
            convo.append({"role": "user", "content": user_attack})
        else:
            convo = [{"role": "user", "content": attack["prompt"]}]
        if convo_token_count(convo) > MAX_TOKEN_LIMIT:
            continue
        reply, lat, code = query_chat(convo, tag, NUM_PREDICT_JB)
        convo.append({"role": "assistant", "content": reply or "[NO_RESPONSE]"})
        out = {
            "model": tag,
            "label": label,
            "data_type": data_type,
            "attack_type": attack_type,
            "attack": attack,
            "conversation": convo,
            "response": reply,
            "hash": sha256(reply),  
            "latency": lat,
            "http": code,
        }
        fp.write(json.dumps(out, ensure_ascii=False) + "\n")
        fp.flush()

def run_benign_turns(fp, tag, pool, num):
    for _ in range(min(num, len(pool))):
        if not pool:
            break
        main_prompt = pool.pop()
        n_history = random.randint(*NUM_BENIGN_HISTORY)
        if len(pool) < n_history - 1:
            refill_and_shuffle_benign()
            if len(pool) < n_history - 1:
                continue
        history_prompts = [main_prompt] + random.sample(pool, max(0, n_history - 1))
        [pool.remove(p) for p in history_prompts[1:] if p in pool]
        convo = []
        for prompt in history_prompts:
            convo.append({"role": "user", "content": prompt})
            reply, lat, code = query_chat(convo, tag, NUM_PREDICT_BENIGN)
            convo.append({"role": "assistant", "content": reply or "[NO_RESPONSE]"})
        if convo_token_count(convo) > MAX_TOKEN_LIMIT:
            continue
        out = {
            "model": tag,
            "label": 0,
            "data_type": "multi_turn_benign",
            "conversation": convo,
            "response": convo[-1]["content"],
            "hash": sha256(convo[-1]["content"]),
            "latency": lat,
            "http": code,
        }
        fp.write(json.dumps(out, ensure_ascii=False) + "\n")
        fp.flush()

def refill_and_shuffle_benign():
    global local_benign_pool
    local_benign_pool = benign_pool.copy()
    random.shuffle(local_benign_pool)
    
# ---- MAIN LOOP ----
with out_path.open("w", encoding="utf-8") as fp:
    local_attack_pool = sampled_attacks.copy()
    random.shuffle(local_attack_pool)
    for tag in SLM_TAGS:
        print(f"🧪 Starting model: {tag}")
        proc = start_model(tag)
        try:
            local_benign_pool = benign_pool.copy()
            random.shuffle(local_benign_pool)
            run_attack_turns(fp, tag, local_attack_pool, NUM_ATTACK_MULTI, multi_intro, 1, "multi", "multi_turn_attack", is_multi=True)
            run_attack_turns(fp, tag, local_attack_pool[NUM_ATTACK_MULTI:], NUM_ATTACK_SINGLE, multi_intro, 1, "single", "single_turn_attack", is_multi=False)
            run_benign_turns(fp, tag, local_benign_pool, NUM_BENIGN_ONLY)
        except Exception as e:
            print(f"[ERROR] {tag} crashed: {e}")
            with open(out_dir / f"errors_{tag}.txt", "a") as ef:
                ef.write(f"{datetime.now()} - {e}\n")
        finally:
            stop_model(tag, proc)

print("🎉 All done. Results at", out_path.resolve())