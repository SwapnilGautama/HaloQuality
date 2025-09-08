# core/reason_labeller.py
from __future__ import annotations

import os
import re
import json
import hashlib
from typing import Dict, List, Optional, Tuple

import pandas as pd

# ------- OpenAI optional import (safe if not installed / no key) -------
OPENAI_READY = False
_openai_client = None
_openai_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

try:
    # Streamlit secrets (optional). Don't fail if streamlit not present.
    try:
        import streamlit as st  # type: ignore
        _OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", None)
    except Exception:
        _OPENAI_API_KEY = None

    # Env fallback
    if not _OPENAI_API_KEY:
        _OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_APIKEY") or os.getenv("OPENAI_KEY")

    if _OPENAI_API_KEY:
        from openai import OpenAI  # type: ignore
        _openai_client = OpenAI(api_key=_OPENAI_API_KEY)
        OPENAI_READY = True
except Exception:
    OPENAI_READY = False
    _openai_client = None

# ------------------ Category taxonomy (+ keywords) ------------------
# Keep taxonomy broad + actionable. These are intentionally inclusive.
_CATEGORIES: Dict[str, List[str]] = {
    "Communication / update": [
        r"update", r"chase", r"follow[- ]?up", r"await", r"waiting", r"no response",
        r"respond(ed|ing)?", r"email", r"call(ed|ing)?", r"letter", r"advise", r"inform"
    ],
    "Data entry / setup": [
        r"data (entry|issue|error)", r"setup", r"set[- ]?up", r"record(s)?", r"index", r"scan",
        r"document(ation)?", r"upload", r"capture", r"input"
    ],
    "Bank / payment": [
        r"payment", r"bank", r"bacs", r"faster payment", r"cheque", r"refund", r"overpayment",
        r"underpayment", r"remit", r"remittance"
    ],
    "Trustee / AVC": [
        r"trustee", r"avc", r"additional voluntary", r"additional contribution", r"external provider"
    ],
    "Postal / dispatch": [
        r"post(al)?", r"dispatch", r"mail(ed|ing)?", r"courier", r"deliver(y|ed)"
    ],
    "Manual calculation": [
        r"manual (calc|calculation|workaround)", r"hand[- ]?calc", r"complex", r"bespoke"
    ],
    "System": [
        r"system (issue|error|down)", r"access", r"permission", r"it ticket", r"bug", r"crash"
    ],
    "Case not created": [
        r"case not created", r"no case", r"missing case"
    ],
    "Death benefits payout": [
        r"death benefit", r"bereave(d|ment)", r"executor", r"probate"
    ],
    "Waiting on member/TPA": [
        r"waiting on (member|tpa|third party)", r"await(ing)? member", r"await(ing)? tpa"
    ],
}

# join into compiled regex
_COMPILED = {k: re.compile("|".join(v), re.I) for k, v in _CATEGORIES.items()}

def _keyword_match(text: str) -> str:
    """Simple semantic-ish keyword pass. Returns first category hit."""
    if not isinstance(text, str) or not text.strip():
        return "Other"
    t = text.strip()
    for cat, rx in _COMPILED.items():
        if rx.search(t):
            return cat
    return "Other"

def _hashable_id(text: str) -> str:
    return hashlib.sha1((text or "").encode("utf-8")).hexdigest()[:12]

def _gpt_refine(comments: List[str], seed_labels: List[str]) -> List[str]:
    """
    Ask GPT to refine the 'Other' labels into the taxonomy above.
    Returns a list of final labels aligned with comments order.
    """
    if not OPENAI_READY or _openai_client is None or not comments:
        return seed_labels

    # Build concise instruction with categories.
    cats = list(_CATEGORIES.keys())
    sys_msg = (
        "You are an assistant that classifies short case comments into exactly one of the given categories. "
        "Return an array of JSON objects with fields: idx (int), reason (string chosen from categories). "
        "Be decisive; never return 'Other' if a category plausibly fits.\n\n"
        f"Categories: {cats}\n"
    )

    # Minimize tokens. Send compact examples.
    items = [{"idx": i, "comment": c} for i, c in enumerate(comments)]
    user_msg = "Classify these:\n" + json.dumps(items, ensure_ascii=False)

    try:
        resp = _openai_client.chat.completions.create(
            model=_openai_model,
            messages=[
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.2,
        )
        txt = resp.choices[0].message.content or "[]"
        parsed = json.loads(txt)
        out = seed_labels[:]
        for row in parsed:
            i = int(row.get("idx", -1))
            r = str(row.get("reason", "")).strip()
            if 0 <= i < len(out) and r:
                # Keep only valid known categories
                out[i] = r if r in _CATEGORIES else out[i]
        return out
    except Exception:
        # Any issue: fall back silently to seed labels
        return seed_labels

def classify_comments(series: pd.Series, use_openai: Optional[bool] = None,
                      sample_cap: int = 220) -> pd.Series:
    """
    Main entry: classify a pandas Series of comment strings -> Series of final categories.
    1) Keyword/semantic pass
    2) (optional) GPT pass to refine only 'Other' rows (up to sample_cap to keep usage bounded)
    """
    if series is None or series.empty:
        return pd.Series([], dtype="object")

    # 1) keyword pass
    seed = series.astype(str).map(_keyword_match)

    # Decide GPT usage
    do_gpt = OPENAI_READY if use_openai is None else bool(use_openai)
    if not do_gpt:
        return seed

    # 2) refine only Others (sample to bound cost)
    idx_other = seed[seed.eq("Other")].index.tolist()
    if not idx_other:
        return seed

    if len(idx_other) > sample_cap:
        idx_other = idx_other[:sample_cap]

    refined_labels = _gpt_refine(series.loc[idx_other].tolist(), ["Other"] * len(idx_other))
    if refined_labels:
        seed.loc[idx_other] = refined_labels

    return seed
