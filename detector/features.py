# detector/features.py
import math, re, gzip
from typing import Dict, Optional, List, Tuple
import numpy as np
from scipy.spatial.distance import cosine

# --- NLTK stopwords (safe fallback) ---
try:
    from nltk.corpus import stopwords
    import nltk
    nltk.download('stopwords', quiet=True)
    STOPWORDS = set(stopwords.words("english"))
except Exception:
    STOPWORDS = set()

# --- Langdetect (optional, error-safe) ---
try:
    from langdetect import detect, detect_langs
except Exception:
    detect = detect_langs = None

from detector.helpers import classify_turn_type, TURN_PROMPT_ONLY, TURN_SINGLE_PAIR, TURN_MULTI_PAIR

CTRL_CHARS = "".join([chr(c) for c in [0x200B,0x200C,0x200D,0x2060,0xFEFF,0x202A,0x202B,0x202C,0x202D,0x202E]])
ZERO_WIDTH_SET = {"\u200b","\u200c","\u200d","\u2060","\ufeff"}
RTL_SET = {"\u202b","\u202e"}

# ---------------- FEATURES ----------------
def is_english(text: str) -> int:
    try:
        if detect is None: return 0
        return int(detect(text) == "en")
    except Exception:
        return 0

def multi_lang_flag(text: str) -> int:
    try:
        if detect_langs is None: return 0
        langs = detect_langs(text)
        return int(len(langs) > 1 and langs[1].prob > 0.1)
    except Exception:
        return 0

def long_flag(text: str) -> int:
    return int(len(text) > 200)

def unicode_stats(text: str) -> Dict[str, float]:
    if not text:
        return dict(len=0, non_ascii=0.0, zero_width=0, ctrl=0, rtl=0, entropy=0.0)
    length = len(text)
    non_ascii_ratio = sum(ord(ch) > 127 for ch in text) / length
    zero_width_count = sum(text.count(ch) for ch in ZERO_WIDTH_SET)
    rtl_count = sum(text.count(ch) for ch in RTL_SET)
    ctrl_count = sum(text.count(ch) for ch in CTRL_CHARS)
    probs = [text.count(c) / length for c in set(text)]
    entropy_val = -sum(p * math.log2(p) for p in probs)
    return dict(len=length, non_ascii=non_ascii_ratio, zero_width=zero_width_count,
                ctrl=ctrl_count, rtl=rtl_count, entropy=entropy_val)

def lexical_stats(text: str) -> Dict[str, float]:
    words = re.findall(r"\w+", text.lower())
    unique_words = set(words)
    stopword_count = sum(1 for w in words if w in STOPWORDS)
    stopword_ratio = stopword_count / max(1, len(words))
    type_token_ratio = len(unique_words) / max(1, len(words))
    return dict(
        word_count=len(words),
        unique_word_count=len(unique_words),
        stopword_ratio=stopword_ratio,
        type_token_ratio=type_token_ratio,
    )

# ---------------- Turn typing ----------------

def dialog_features(text: str) -> Dict[str, int]:
    tl = text.lower()
    u = tl.count("user:")
    a = tl.count("assistant:")
    tt = classify_turn_type(text)
    return dict(
        user_turn_count=u,
        assistant_turn_count=a,
        pair_count=min(u, a),
        prompt_only=int(tt == TURN_PROMPT_ONLY),
        single_pair=int(tt == TURN_SINGLE_PAIR),
        multi_turn=int(tt == TURN_MULTI_PAIR),
    )

# ---------------- Info-theoretic bits ----------------
def entropy(text: str) -> float:
    tokens = text.split()
    if not tokens: return 0.0
    probs = [tokens.count(t)/len(tokens) for t in set(tokens)]
    return -sum(p * math.log2(p) for p in probs)

def entropy_drift(last_prompt: str, history: List[str]) -> float:
    if not history: return 0.0
    hist_ents = [entropy(h) for h in history if h]
    if not hist_ents: return 0.0
    return entropy(last_prompt) - float(np.mean(hist_ents))

# ---------------- Distances & projections ----------------
def _safe_cosine_distance(u: np.ndarray, v: np.ndarray) -> float:
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    nu = np.linalg.norm(u); nv = np.linalg.norm(v)
    if nu < 1e-8 or nv < 1e-8:
        return 1.0  # max distance
    return cosine(u, v)

def prompt_history_coherence(prompt_emb: np.ndarray, history_emb: np.ndarray) -> float:
    d = _safe_cosine_distance(prompt_emb, history_emb)
    return float(np.clip(1.0 - d, 0.0, 1.0))

def axis_projection(emb: np.ndarray, benign_c: np.ndarray, attack_c: np.ndarray) -> Tuple[float, float]:
    v  = np.asarray(attack_c, dtype=float) - np.asarray(benign_c, dtype=float)
    vn = v / (np.linalg.norm(v) + 1e-12)
    x  = np.asarray(emb, dtype=float) - np.asarray(benign_c, dtype=float)
    proj = float(np.dot(x, vn))
    cos_axis = float(np.clip(1.0 - _safe_cosine_distance(x, vn), 0.0, 1.0))
    return proj, cos_axis

# ---------------- Optional LOF (geri uyum) ----------------
from sklearn.neighbors import LocalOutlierFactor
def embedding_outlier_scores(train_embs, test_embs, n_neighbors=20):
    lof = LocalOutlierFactor(n_neighbors=n_neighbors, novelty=True)
    lof.fit(train_embs)
    train_scores = -lof.decision_function(train_embs)
    test_scores  = -lof.decision_function(test_embs)
    return train_scores, test_scores

# ---------------- Compression ----------------
def compression_ratio(text: str) -> float:
    if not text: return 0.0
    raw = text.encode("utf-8", "ignore")
    try:
        comp = gzip.compress(raw, compresslevel=6)
        return len(comp) / max(1, len(raw))
    except Exception:
        return 0.0

# =============================================================================
def intent_soft_flag(text: str) -> int:
    patterns = [
        r"\bcould you\b", r"\bwould you\b", r"\bplease\b", r"\blet'?s\b", r"\bimagine\b", r"\bhypothetically\b",
        r"\bjust curious\b", r"\bfor fun\b", r"\bas if\b", r"\bassume\b",
        r"\bpretend\b", r"\bsuppose\b", r"\bwhat if\b", r"\bif you\b", r"\bexplain as if\b", r"\bexplain.*ignoring.*rule",
        r"\broleplay\b", r"\bact as\b", r"\bbecome\b", r"\bimagine you are\b", r"\bcan you\b", r"\bdo you think\b",
        r"\bfor (research|education(al)?|testing|demonstration|illustration|academic|analysis|evaluation|assessment)\b",
        r"\bfor (curiosity|exploration|entertainment|amusement|enjoyment|leisure|pleasure|personal (interest|curiosity))\b",
    ]
    return int(any(re.search(p, text, re.I) for p in patterns))

def build_feature_vector(
    text: str,
    emb: np.ndarray,
    benign_c: np.ndarray,
    attack_c: np.ndarray,
    history_emb: np.ndarray = None,
    last_prompt_text: str = "",
    history_texts: List[str] = None,
    outlier_score: Optional[float] = None
) -> np.ndarray:

    # unicode / lexical / dialog
    stats  = unicode_stats(text)
    lex    = lexical_stats(text)
    dialog = dialog_features(text)
    features: List[float] = list(stats.values()) + list(lex.values()) + list(dialog.values())

    # embeddings & centroids
    emb = np.asarray(emb, dtype=float).ravel()
    benign_c = None if benign_c is None else np.asarray(benign_c, dtype=float).ravel()
    attack_c = None if attack_c is None else np.asarray(attack_c, dtype=float).ravel()
    history_emb = np.zeros_like(emb) if history_emb is None else np.asarray(history_emb, dtype=float).ravel()

    if benign_c is not None and attack_c is not None:
        features += [
            _safe_cosine_distance(emb, benign_c),
            _safe_cosine_distance(emb, attack_c),
            _safe_cosine_distance(emb, history_emb),
        ]
        ax_proj, ax_cos = axis_projection(emb, benign_c, attack_c)
    else:
        features += [1.0, 1.0, 1.0]  # max distance as neutral
        ax_proj, ax_cos = 0.0, 0.0

    # coherence & entropy drift
    features += [prompt_history_coherence(emb, history_emb)]
    features += [entropy_drift(last_prompt_text, history_texts or [])]

    # outlier score (opsiyonel)
    features += [float(outlier_score) if outlier_score is not None else 0.0]

    # norms
    features += [float(np.linalg.norm(emb))]
    features += [float(np.linalg.norm(emb - history_emb))]

    # dot products to centroids (varsa)
    if benign_c is not None and attack_c is not None:
        features += [float(np.dot(emb, benign_c)), float(np.dot(emb, attack_c))]
    else:
        features += [0.0, 0.0]

    # simple textual ratios
    uppercase_ratio = sum(1 for ch in text if ch.isupper()) / max(1, len(text))
    special_ratio   = sum(1 for ch in text if not ch.isalnum() and not ch.isspace()) / max(1, len(text))
    tokens = text.split()
    unique_token_ratio = len(set(tokens)) / max(1, len(tokens))
    avg_token_len      = float(np.mean([len(tok) for tok in tokens])) if tokens else 0.0
    intent    = intent_soft_flag(text)
    is_eng    = is_english(text)
    l_flag    = long_flag(text)
    multi_lang = multi_lang_flag(text)

    features += [
        uppercase_ratio, special_ratio, unique_token_ratio, avg_token_len,
        intent, is_eng, multi_lang, l_flag
    ]

    # axis + compression
    comp_ratio = compression_ratio(text)
    features += [ax_proj, ax_cos, comp_ratio]

    return np.asarray(features, dtype=np.float32)