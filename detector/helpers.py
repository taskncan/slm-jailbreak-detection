# detector/turns.py
from __future__ import annotations
import re
from typing import List, Tuple
import numpy as np

TURN_PROMPT_ONLY = "prompt_only"
TURN_SINGLE_PAIR = "single_pair"
TURN_MULTI_PAIR  = "multi_pair"

USER_RE = re.compile(r"\buser:", re.I)
ASST_RE = re.compile(r"\bassistant:", re.I)

def _count_roles(text: str):
    tl = text.lower()
    u_pos = [m.start() for m in USER_RE.finditer(tl)]
    a_pos = [m.start() for m in ASST_RE.finditer(tl)]
    return len(u_pos), len(a_pos), u_pos, a_pos

def classify_turn_type(text: str) -> str:
    """
    This preserves your desired "single-prompt" behavior:
      - a==0 and u<=1  → prompt_only
      - u==1 and a==1 and (user first) → single_pair
      - others → multi_pair
    """
    u, a, u_pos, a_pos = _count_roles(text)
    if a == 0 and u <= 1:
        return TURN_PROMPT_ONLY
    if u == 1 and a == 1 and (not u_pos or not a_pos or u_pos[0] < a_pos[0]):
        return TURN_SINGLE_PAIR
    return TURN_MULTI_PAIR

def is_single_turn_text(text: str) -> bool:
    return classify_turn_type(text) in {TURN_PROMPT_ONLY, TURN_SINGLE_PAIR}

def is_single_turn_batch(texts: List[str]) -> np.ndarray:
    return np.array([is_single_turn_text(t) for t in texts], dtype=bool)

def extract_last_user_and_history(text: str, window: int = 3) -> Tuple[str, str]:
    pairs = re.split(r"(?=user:)", text, flags=re.I)
    user_prompts = [p for p in pairs if p.strip().lower().startswith("user:")]
    if not user_prompts:
        return text, ""
    if len(user_prompts) == 1:
        return user_prompts[-1], ""
    hist = user_prompts[max(0, len(user_prompts) - window - 1):-1]
    return user_prompts[-1], (" ".join(hist) if hist else "")