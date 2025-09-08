# core/reason_labeller.py
# Deterministic reason classification for FAIL rows using a frozen rulebook
# Built from RCA2 usage in FirstPassAccuracy_Aug'25.xlsx
from __future__ import annotations

import re
from typing import Dict, Iterable, List, Optional

import pandas as pd

# -----------------------------
# 1) Normalise common RCA2 variants (optional direct mapping)
# -----------------------------
_NORMALIZE_RCA2: Dict[str, str] = {
    "Incorrect Data Input": "Incorrect Data Input",
    "Incorrect Calculator": "Incorrect Calculator",
    "Incorrect Factors": "Incorrect Factors",
    "Incorrect Formula": "Incorrect Formula",
    "Incorrect revaluation rates": "Incorrect revaluation rates",
    "Incorrect handoff": "Incorrect handoff",
    "Knowledge Gap-Offshore": "Knowledge Gap-Offshore",
    # looser normalisations
    "Knowledge Gap–Offshore": "Knowledge Gap-Offshore",
    "Knowledge Gap-Off-Shore": "Knowledge Gap-Offshore",
    "Incorrect revaluation rate": "Incorrect revaluation rates",
    "Incorrect calc": "Incorrect Calculator",
    "Calculator incorrect": "Incorrect Calculator",
}

def _normalize_rca2(v: Optional[str]) -> Optional[str]:
    if not isinstance(v, str):
        return None
    s = v.strip()
    if not s:
        return None
    return _NORMALIZE_RCA2.get(s, s)  # accept unseen explicit RCA2s as-is


# -----------------------------
# 2) Frozen rulebook — learned from your data
#    Each label maps to a list of terms/phrases.
#    We compile them into a single regex per label.
# -----------------------------
_RULE_TERMS: Dict[str, List[str]] = {
    "Incorrect Calculator": [
        "calc incorrect","used","calc","calculation","rejected incorrect","gmp","member",
        "comments","rejected comments","incorrect calculation","factors","revaluation"
    ],
    "Incorrect Data Input": [
        "updated","incorrect input","run","calc run","calc","incorrect","incorrect data","dob",
        "date of birth","ni number","ni","postcode","address","entered","amended","keyed"
    ],
    "Incorrect Factors": [
        "factors","factors incorrect","applied","lrf","erf","arrears","increase","annuity factor",
        "annuity","factor","gmp","incorrect","incorrect factor"
    ],
    "Incorrect Formula": [
        "incorrect calculation","calculation","mathematical","mathematical error","calculation incorrect",
        "calculated incorrect","arrears","error","calculated","gmp","comments","rejected comments"
    ],
    "Incorrect handoff": [
        "send","member","retirement","pension","peer review","handover","handoff","sent to",
        "transfer to","moved to","escalate","escalated","reassign","re-assigned"
    ],
    "Incorrect revaluation rates": [
        "revaluation rate","reval rate","revaluation","reval","increase applied","incorrect revaluation",
        "revaluation incorrect","revaluation factors"
    ],
    "Knowledge Gap-Offshore": [
        "rejected comments","comments","rejected comment","comment","split","need","incorrect",
        "calc","calculation","rejected incorrect","review rejected","peer review"
    ],
}

def _compile_regex(words: Iterable[str]) -> re.Pattern:
    # word-boundary + allow whitespace for multi-words
    parts = []
    for w in words:
        w = w.strip()
        if not w:
            continue
        esc = re.escape(w).replace(r"\ ", r"\s+")
        parts.append(rf"(?:\b{esc}\b)")
    if not parts:
        # match nothing
        parts = [r"a^"]
    return re.compile("|".join(parts), flags=re.I)

_COMPILED = {lbl: _compile_regex(ws) for lbl, ws in _RULE_TERMS.items()}

# -----------------------------
# 3) Public API
# -----------------------------
def classify_text(text: Optional[str], rca2: Optional[str] = None) -> str:
    """
    Purely rule-based with RCA2 override. Never returns NaN/None/"".
    1) If RCA2 present → normalise and return.
    2) Else → apply regex rules; highest hit-count wins.
    3) Else → "Other".
    """
    # 1) Direct RCA2 (most authoritative)
    r2 = _normalize_rca2(rca2)
    if isinstance(r2, str) and r2.strip():
        return r2

    # 2) Rules on comment text
    t = (text or "").strip()
    if t:
        scores = {}
        for label, pat in _COMPILED.items():
            hits = pat.findall(t)
            if hits:
                scores[label] = len(hits)
        if scores:
            # pick label with max hits; tie-breaker: longer rule list
            best = sorted(scores.items(), key=lambda kv: (-kv[1], -len(_RULE_TERMS[kv[0]])))[0][0]
            return best

    # 3) Fallback
    return "Other"


def label_dataframe(df: pd.DataFrame,
                    text_col: str = "Case Comment",
                    rca2_col: str = "RCA2") -> pd.Series:
    """
    Vectorised wrapper. Ensures no null/empty labels leak out.
    """
    txt = df[text_col] if text_col in df.columns else pd.Series(index=df.index, dtype=object)
    r2 = df[rca2_col] if rca2_col in df.columns else pd.Series(index=df.index, dtype=object)
    txt = txt.fillna("").astype(str)
    r2 = r2.fillna("").astype(str)

    out = []
    for t, rr in zip(txt, r2):
        lab = classify_text(t, rr)
        if not lab or str(lab).strip().lower() in ("", "nan", "none"):
            lab = "Other"
        out.append(lab)
    return pd.Series(out, index=df.index, name="reason")
