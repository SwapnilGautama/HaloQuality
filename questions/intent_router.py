# questions/intent_router.py
"""
Semantic intent -> question slug router.

Returns a module slug that corresponds to a file under questions/<slug>.py
(e.g., 'fpa_fail_rate' -> questions/fpa_fail_rate.py).

We resolve in three passes:
  1) exact phrase match (fast path)
  2) keyword pattern match
  3) fuzzy best-match (rapidfuzz if available, else difflib)
"""

from __future__ import annotations
import re
from typing import Dict, List, Tuple

# Optional dependency: rapidfuzz
try:
    from rapidfuzz import fuzz
except Exception:  # pragma: no cover
    fuzz = None

import difflib


# ------------ Canonical slugs you currently have ------------
KNOWN_SLUGS = {
    "complaint_volume_rate",
    "complaints_per_thousand",
    "corr_nps",
    "fpa_fail_rate",
    "mom_overview",
    "rca1_portfolio_process",
    "unique_cases_mom",
}

# ------------ Helpful synonyms/aliases ------------
ALIASES: Dict[str, List[str]] = {
    # complaints volume & MoM
    "complaint_volume_rate": [
        "complaint volume rate",
        "complaints volume rate",
        "complaint volume",
        "complaints volume",
        "complaints count",
        "complaints by month",
        "complaints trend by month",
        "complaints month on month",
        "complaint trend",
        "complaints trend",
        "complaints overview",
    ],

    # complaints per 1000 cases
    "complaints_per_thousand": [
        "complaints per thousand",
        "complaints per 1000",
        "complaints per 1,000",
        "complaints per k",
        "complaint rate per thousand",
        "complaints per 1000 cases",
        "complaints per k cases",
    ],

    # NPS correlation
    "corr_nps": [
        "nps correlation",
        "correlation with nps",
        "complaints nps correlation",
        "corr nps",
        "correlation nps complaints",
    ],

    # FPA
    "fpa_fail_rate": [
        "fpa fail rate",
        "first pass accuracy fail rate",
        "fpa failure rate",
        "fpa defects rate",
        "fpa defects",
        "fpa failures",
        "fpa fail drivers",     # <-- map driver(s) to same analysis
        "fpa fail driver",
        "fpa drivers",
    ],

    # MoM overview of cases/metrics
    "mom_overview": [
        "month on month overview",
        "mom overview",
        "overview by month",
        "trend overview",
        "monthly overview",
    ],

    # RCA1 by portfolio for a process
    "rca1_portfolio_process": [
        "rca1 by portfolio for process",
        "rca1 by portfolio",
        "root cause by portfolio",
        "reasons by portfolio",
        "rca by portfolio",
        "rca1 portfolio process",
    ],

    # unique cases month on month
    "unique_cases_mom": [
        "unique cases month on month",
        "unique case count by month",
        "case count by month",
        "cases trend by month",
        "unique cases trend",
    ],
}

# ------------ Keyword patterns (cheap contains checks) ------------
KEYWORD_RULES: List[Tuple[str, List[str]]] = [
    ("complaints_per_thousand", ["per 1000", "per 1,000", "per thousand", "per k"]),
    ("corr_nps", ["nps", "correlation", "corr"]),
    ("fpa_fail_rate", ["fpa", "fail", "driver", "drivers", "defect", "failure"]),
    ("rca1_portfolio_process", ["rca", "rca1", "root cause", "reasons", "portfolio", "process"]),
    ("unique_cases_mom", ["unique cases", "case count", "cases trend", "month on month"]),
    ("complaint_volume_rate", ["complaint", "complaints", "volume", "month on month", "trend"]),
    ("mom_overview", ["overview", "month on month", "mom"]),
]


# ------------ Utilities ------------
_NORMALIZE_RX = re.compile(r"[^a-z0-9 ]+")

def _norm(s: str) -> str:
    return _NORMALIZE_RX.sub(" ", s.lower()).strip()


def _fuzzy_best(user: str, choices: List[Tuple[str, str]]) -> str | None:
    """Return slug with best fuzzy score, or None if score too low."""
    if not choices:
        return None

    best_slug, best_score = None, -1.0
    for slug, phrase in choices:
        if fuzz:
            score = fuzz.token_set_ratio(user, phrase)
        else:
            score = difflib.SequenceMatcher(None, user, phrase).ratio() * 100
        if score > best_score:
            best_slug, best_score = slug, score

    # threshold tuned to reduce false positives
    return best_slug if best_score >= 75 else None


# ------------ Public API ------------
def route_intent(user_text: str) -> str | None:
    """
    Return a question module slug (e.g., 'fpa_fail_rate') or None if no match.
    """
    if not user_text:
        return None

    text = _norm(user_text)

    # 1) exact-phrase pass
    exact_pairs: List[Tuple[str, str]] = []
    for slug, phrases in ALIASES.items():
        for p in phrases:
            phrase = _norm(p)
            exact_pairs.append((slug, phrase))
            if phrase and phrase in text:
                return slug

    # 2) keyword pass (cheap contains checks)
    for slug, keywords in KEYWORD_RULES:
        # must contain at least one keyword
        if any(k in text for k in keywords):
            return slug

    # 3) fuzzy pass
    return _fuzzy_best(text, exact_pairs)


# Convenience name for older code that imported `run_intent`
run_intent = route_intent
