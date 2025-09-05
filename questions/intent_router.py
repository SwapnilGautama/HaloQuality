# questions/intent_router.py
"""
Lightweight intent router -> returns (module_path, kwargs_dict)
We match common phrases and synonyms to question modules.
"""

from __future__ import annotations
from typing import Tuple, Dict, Optional
import re

try:
    # optional fuzzy
    from rapidfuzz import process, fuzz  # type: ignore
except Exception:
    process = None
    fuzz = None

# Map canonical question module to phrases/synonyms that should trigger it.
INTENT_SYNONYMS = {
    # FPA — drivers / reasons / rate
    "questions.fpa_fail_drivers": [
        "fpa fail drivers",
        "fpa failure drivers",
        "drivers of fpa fails",
        "reasons for fpa fails",
        "fpa fail reasons",
        "biggest drivers of case fails",
        "why are cases failing",
        "fpa fail rate",
        "fpa failure rate",
        "fpa fail by reason",
        "fpa fail analysis",
    ],

    # Keep your other mappings here (examples):
    "questions.complaints_per_thousand": [
        "complaints per 1000",
        "complaints per thousand",
        "complaints per 1k",
        "cpt",
    ],
    "questions.complaint_volume_rate": [
        "complaint volume rate",
        "complaints volume",
        "complaints trend",
        "complaints by month",
    ],
    "questions.corr_nps": [
        "nps correlation",
        "complaints nps correlation",
        "corr nps",
        "complaints vs nps",
    ],
    "questions.rca1_portfolio_process": [
        "rca1 by portfolio",
        "rca portfolio process",
        "rca analysis by portfolio",
    ],
    "questions.unique_cases_mom": [
        "unique cases",
        "unique cases mom",
        "cases month over month",
    ],
    "questions.mom_overview": [
        "mom overview",
        "month over month overview",
        "overall trends",
    ],
}

# Flatten for fuzzy
ALL_PHRASES = [(phrase, mod) for mod, phrases in INTENT_SYNONYMS.items() for phrase in phrases]

def _simple_contains(q: str) -> Optional[str]:
    ql = q.lower()
    for mod, phrases in INTENT_SYNONYMS.items():
        for p in phrases:
            if p in ql:
                return mod
    return None

def match_intent(user_text: str) -> Tuple[str, Dict]:
    """
    Return (module_path, kwargs) for the best matching intent.
    Falls back to substring contains if rapidfuzz isn't available.
    """
    if not user_text or not user_text.strip():
        # default to a general overview, tweak if you like
        return "questions.mom_overview", {}

    # 1) exact/contains
    mod = _simple_contains(user_text)
    if mod:
        return mod, {}

    # 2) fuzzy (optional)
    if process is not None and fuzz is not None:
        choices = [p for p, _ in ALL_PHRASES]
        best = process.extractOne(user_text, choices, scorer=fuzz.WRatio)
        if best and best[1] >= 80:
            phrase = best[0]
            for p, m in ALL_PHRASES:
                if p == phrase:
                    return m, {}

    # Default fallback
    return "questions.mom_overview", {}
