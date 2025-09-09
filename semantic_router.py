# semantic_router.py
from __future__ import annotations
import re
from typing import Dict, Iterable

# loose month matcher; router only passes through a hint month if present
_MONTH_RX = r"(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s*(\d{2,4})?"

def _norm_q(q: str) -> str:
    return (q or "").strip().lower()

def _any_in(text: str, phrases: Iterable[str]) -> bool:
    t = text
    return any(p in t for p in phrases)

def _any_regex(text: str, patterns: Iterable[str]) -> bool:
    return any(re.search(p, text) for p in patterns)

def match(q: str) -> Dict:
    """
    Very small, safe router that only returns a slug + lightweight params.
    Q1: complaints_june_by_portfolio
    Q2: first_pass_accuracy
    Q4: nps_by_portfolio
    """
    text = _norm_q(q)

    # Extract month hint if present (used by questions if they want it)
    m = re.search(_MONTH_RX, text)
    params = {"hint_month": m.group(0)} if m else {}

    # ---------------------------
    # NPS triggers (Q4)
    # ---------------------------
    nps_trigs = [
        "nps", "nps ", " nps", "net promoter", "promoter score", "net promoter score",
        "promoters", "detractors", "passives",
        "nps by portfolio", "nps trend",
        "nps sentiment", "nps sentiment analysis", "suggestions sentiment"
    ]
    if _any_in(text, nps_trigs):
        return {"slug": "nps_by_portfolio", "params": params}

    # ---------------------------
    # FPA triggers (Q2)
    # ---------------------------
    fpa_trigs = [
        "fpa", "first pass accuracy", "first-pass accuracy", "first pass  accuracy", "accuracy",
        "team comparison", "manager comparison", "portfolio comparison", "location comparison",
        " comparison", "compare ", " vs ",
        "fail reasons", "fail reason", "fail reason analysis",
        "complaint vs accuracy", "complaint and accuracy", "compaint accuracy", "compaint & accuracy"
    ]
    # Also: if both "complain*" and "accuracy" appear, treat as FPA
    if _any_in(text, fpa_trigs) or (_any_regex(text, [r"\bcomplain\w*\b", r"\bcomplint\b"]) and "accuracy" in text):
        return {"slug": "first_pass_accuracy", "params": params}

    # ---------------------------
    # Complaints triggers (Q1)
    # ---------------------------
    # Robust words/typos + per-1000 patterns + RCA + explicit locations
    complaint_word_regexes = [
        r"\bcomp\b(?!any)",              # 'comp' as a standalone token, not 'company'
        r"\bcomplaints?\b",
        r"\bcomplain(?:s|ed|ing)?\b",
        r"\bcomplint\b",
        r"\brca\b",
        r"complaints\s*/\s*1000",
        r"complaints\s+per\s+1000",
        r"complint\s*/\s*1000"
    ]
    # Location-driven complaints queries
    locations = [
        "chichester", "exeter", "leatherhead", "london",
        "scotland", "northwest", "north west", "north-west"
    ]
    mentions_location = any(loc in text for loc in locations)
    if _any_regex(text, complaint_word_regexes) or (
        mentions_location and _any_regex(text, [r"\bcomplain", r"\bcomplaints?\b", r"\bcomplint\b", r"\bcomp\b(?!any)"])
    ) or _any_in(text, ["complaint analysis", "complaint reason", "complaint rca", "rca by portfolio"]):
        return {"slug": "complaints_june_by_portfolio", "params": params}

    # Default to Complaints (Q1) if unknown
    return {"slug": "complaints_june_by_portfolio", "params": params}
