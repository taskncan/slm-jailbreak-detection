import hashlib
import random, uuid, datetime, textwrap, json
import time, subprocess, requests
from typing import List
import codecs
from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer
import re
import sys
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import faiss, numpy as np
from tqdm.auto import tqdm


OLLAMA_URL_GENERATE = "http://localhost:11434/api/generate"
OLLAMA_URL_CHAT = "http://localhost:11434/api/chat"

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

WARMUP_SEC = 20
NUM_PREDICT = 256  # max tokens to predict in one go

# Constants for augmentations
ZERO = ["​"]
EMO  = ["🗝️", "📜", "🤖", "🔒", "🛡️", "⚔️", "🧩", "💡", "🚀", "🌐", "🔍", "📈", "📉"]

# --- helper functions ------------------------------------------------------
BLACKLIST_KEYWORDS_PATH = Path("prompts") / "blacklist_keywords.txt"
with BLACKLIST_KEYWORDS_PATH.open(encoding="utf-8") as f:
    BLACKLIST_KEYWORDS = [line.strip().lower() for line in f if line.strip()]

escaped_keywords = [re.escape(kw).replace(' ', r'[\s_\-]+') for kw in BLACKLIST_KEYWORDS]
pattern = r'(' + '|'.join(r'\b' + kw + r'\b' for kw in escaped_keywords) + r')'
BLACKLIST_RE = re.compile(pattern, flags=re.IGNORECASE)

def is_clean_benign(prompt):
    return not BLACKLIST_RE.search(prompt)

def fuzz(s: str, p: float = 0.15) -> str:
    """
    Returns `s` with approximately `p` proportion of letters/digits replaced by similar-looking
    homoglyphs, mixed big/little-case, or zero-width characters sprinkled in.
    Produces a 'similar' but not identical variant for the model.
    """
    ZW   = ["\u200b", "\u200c", "\u200d"]
    HMAP = str.maketrans({"a": "а", "e": "е", "i": "і", "o": "ο", "c": "с", "s": "ѕ"})
    out  = []
    for ch in s:
        if random.random() < p:
            mode = random.choice(["zero", "homo", "case"])
            if mode == "zero":
                out.append(random.choice(ZW) + ch)
            elif mode == "homo":
                out.append(ch.translate(HMAP))
            else:  # case
                out.append(ch.upper() if ch.islower() else ch.lower())
        else:
            out.append(ch)
    return "".join(out)

def rand_date(years: int = 2) -> str:
    """
    Randomly generates a date within the last `years` years.
    """
    now    = datetime.datetime.utcnow()
    delta  = datetime.timedelta(days=random.randint(0, 365 * years))
    return (now - delta).strftime("%Y-%m-%d %H:%M:%SZ")

def maybe(prob: float, text: str = "") -> str:
    """
    Returns `text` with probability `prob`, otherwise an empty string.
    """
    return text if random.random() < prob else ""

def filter_near_dupes(
        rows: list[dict],
        thresh: float = 0.89,            # cos θ eşiği (0.89 ≈ 27°)
        model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        batch_size: int = 256,
        device: str | None = None,
        top_k: int = 10,                 
) -> list[dict]:
    """
    SBERT + Faiss based near-duplicate filter.

    • Embeds the rows (normalize=True).
    • Searches with Faiss IndexFlatIP (inner-product).
    • Removes *subsequent* rows with similarity ≥ thresh (stable keep-first).
    """

    if len(rows) < 2:
        return rows                     # tek satır → hızlı çık

    if device is None:                 # otomatik seçim
        device = 'mps'

    # 1) Collect texts
    texts = [str(r["text"]) for r in rows]

    # 2) embed
    model = SentenceTransformer(model_name, device=device)
    emb   = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    )

    # 3) Faiss index (||v||=1 ⇒ dot = cosine)
    dim   = emb.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(emb.astype(np.float32))

    keep = np.ones(len(rows), dtype=bool)

    # 4) Near-duplicate scan 
    for i in tqdm(range(len(rows)), desc="near-dupe scan"):
        if not keep[i]:
            continue
        # skip if this row is already marked as duplicate
        D, I = index.search(emb[i:i+1], top_k)
        dup_idx = [j for d, j in zip(D[0], I[0]) if j > i and d >= thresh]
        keep[dup_idx] = False

    return [r for r, ok in zip(rows, keep) if ok]

# --- augmentation functions ------------------------------------------------

def add_base64_like_noise(text: str) -> str:
    """Adds a base64-like noise to the text."""
    noise = "".join(random.choices("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=", k=20))
    return f"{text} {noise}"

def benign_rot13(text:str)->str:
    return codecs.encode(text,"rot_13")

def benign_homoglyph(text:str)->str:
    HOMO = str.maketrans({"a":"а","e":"е","i":"і","o":"о","s":"ѕ"})
    return text.translate(HOMO)

def add_char_noise(text: str) -> str:
    """Randomly injects uppercase, whitespace, zero-width and emoji noise."""
    out: List[str] = []
    for ch in text:
        # random uppercase
        out.append(ch.upper() if random.random() < 0.01 else ch)
        # random space or zero-width char
        if random.random() < 0.02:
            out.append(random.choice([" ", random.choice(ZERO)]))
        # random emoji
        if random.random() < 0.001:
            out.append(random.choice(EMO))
    return "".join(out)

def wrap_long_context(prompt: str) -> str:
    """Wraps the prompt in a long context template"""

    uid   = uuid.uuid4().hex[:8]

    base_tpl = random.choice(["EMAIL", "LOREM", "YAML", "HTML", "README", "SQL", "JSON", "PYTHON", "R"])

    if base_tpl == "EMAIL":
        body = (
            f"From: user{uid[:4]}@example.com\n"
            f"To: sec-team@example.com\n"
            f"Subject: URGENT – ticket {uid.upper()}\n"
            f"Date: {rand_date()}\n\n"
            "Hi,\n\n"
            "Please review the indented request below.\n\n"
            + textwrap.indent(prompt, "    ")
            + "\n\nBest,\nBot\n"
        )

    elif base_tpl == "YAML":
        body = (
            "---\n"
            f"id: {uid}\n"
            "lang: en\n"
            f"ts: {rand_date()}\n"
            "query: |\n"
            + textwrap.indent(prompt, "  ")
            + "\n---\n"
            + maybe(.4, "# extra_meta: placeholder\n")
        )

    elif base_tpl == "HTML":
        body = (
            "<!-- autogenerated wrapper -->\n"
            "<html><body>\n"
            "<pre style=\"background:#eef;padding:6px\">\n"
            + prompt
            + "\n</pre>\n"
            + maybe(.3, f"<script>console.log('{uid}');</script>\n")
            + "</body></html>\n"
        )

    elif base_tpl == "SQL":
        body = (
            f"-- insert prompt as test row ({uid})\n"
            "BEGIN;\n"
            "CREATE TEMP TABLE t(id TEXT, txt TEXT);\n"
            "INSERT INTO t VALUES\n"
            f"  ('{uid}', $$\n{prompt}\n$$);\n"
            "COMMIT;\n"
        )

    elif base_tpl == "JSON":
        body = (
            "{\n"
            f"  \"id\": \"{uid}\",\n"
            f"  \"ts\": \"{rand_date()}\",\n"
            "  \"payload\": " + json.dumps(prompt) + "\n"
            "}\n"
        )

    elif base_tpl == "README":
        body = (
            f"# PromptArchive {uid}\n\n"
            "## Usage\n"
            "Paste this prompt into the LLM console **after** the divider.\n\n"
            "-----8<-----\n"
            + prompt + "\n"
            "-----8<-----\n"
        )
    elif base_tpl == "PYTHON":
        body = (
            f"# Python code snippet {uid}\n"
            "def query():\n"
            + textwrap.indent(prompt, "    ")
            + "\n\n# Call the function to execute the query\nquery()\n"
        )
    elif base_tpl == "R":
        body = (
            f"# R script snippet — generated {rand_date()}\n"
            "log <- function(msg) cat(sprintf(\"[%s] %s\\n\", Sys.time(), msg))\n"
            "log(\"Starting analysis …\")\n"
            "# --- USER PROMPT EMBEDDED BELOW ---\n"
            f"{prompt}\n"
            "# --- end of prompt ---\n"
            "log(\"Done.\")\n"
        )
    else:  # LOREM default
        lipsum = " ".join(["Lorem ipsum"] * random.randint(30, 80))
        body = f"{lipsum}\n\n### USER PROMPT ↓↓↓\n{prompt}\n"

    # random chance to add more noise
    if random.random() < .5:
        body = wrap_long_context(body)

    return body

# ──────────────────────── OLLAMA SLM helper functions ────────────────────────────
sha256 = lambda s: hashlib.sha256(s.encode()).hexdigest()

def start_model(tag: str) -> subprocess.Popen:
    p = subprocess.Popen(
        ["ollama", "run", tag],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    print(f"[{tag}] starting model with PID {p.pid}")
    time.sleep(WARMUP_SEC)
    return p

def stop_model(tag: str, proc: subprocess.Popen):
    subprocess.run(["ollama", "stop", tag],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f"[{tag}] stopping model with PID {proc.pid}")
    time.sleep(WARMUP_SEC)
    proc.kill()

def query_model(prompt: str, tag: str) -> tuple[str, float, int]:
    t0 = time.time()
    try:
        r = requests.post(
            OLLAMA_URL_GENERATE,
            json={"model": tag, "prompt": prompt, "stream": False,
                  "options": {"num_predict": NUM_PREDICT}},
            timeout=120,
        )
        print(f"[{tag}] {r.status_code} {r.elapsed.total_seconds():.3f}s")
        latency = round(time.time() - t0, 3)
        txt     = r.json().get("response", "").replace("\n", " ")
        return txt, latency, r.status_code
    except Exception as e:
        return f"<error:{e}>", round(time.time() - t0, 3), -1


def query_chat(messages, tag, num_predict):
    t0 = time.time()
    resp = requests.post(
        OLLAMA_URL_CHAT,
        json={"model": tag, "messages": messages, "stream": True, "options": {"num_predict": num_predict}},
        timeout=120,
        stream=True
    )
    contents = []
    for line in resp.iter_lines():
        if not line:
            continue
        try:
            js = json.loads(line.decode("utf-8"))
        except Exception as e:
            print("[WARN] JSON parse error:", e, line)
            continue
        if "message" in js and "content" in js["message"]:
            contents.append(js["message"]["content"])
        if js.get("done", False):
            break
    print(f"[{tag}] {resp.status_code} {resp.elapsed.total_seconds():.3f}s")
    latency = round(time.time() - t0, 3)
    msg = "".join(contents).replace("\n", " ").strip()
    return msg, latency, resp.status_code