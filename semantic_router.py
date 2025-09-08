# semantic_router.py
from __future__ import annotations
import re
from typing import Dict, Optional

# Month hint extraction (e.g., "Aug 25", "aug-25", "august 2025")
_MONTH_RX = re.compile(
    r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s*[-/]?\s*(\d{2,4})?\b",
    re.IGNORECASE,
)

def _norm(text: str) -> str:
    return (text or "").strip().lower()

# Trigger patterns per question (use word boundaries; keep them tight)
_TRIGGERS: Dict[str, list[str]] = {
    # Q3 — NEW: fail reasons analysis (FRA)
    "fail_reasons_analysis": [
        r"\breasons?\s+for\s+fail(?:ure)?\b",
        r"\bfail(?:ure)?\s+(?:reasons?|drivers?)\b",
        r"\bfpa\s+(?:reasons?|drivers?)\b",
        r"\bfail\s+pareto\b",
        r"\bpareto\b",  # generic, but useful in FRA context
    ],
    # Q2 — First-Pass Accuracy
    "first_pass_accuracy": [
        r"\bfirst[-\s]?pass\s+accuracy\b",
        r"\bfirst[-\s]?pass\b",
        r"\bfpa\b",
        r"\bfpa\s+accuracy\b",
    ],
    # Q1 — Complaints by portfolio (default)
    "complaints_june_by_portfolio": [
        r"\bcomplaints?\b",
        r"\bby\s+portfolio\b",
        r"\brca2\b",
        r"\bnorthwest\b",
    ],
}

# Order matters when scoring ties — the more specific tasks first
_ORDER = ["fail_reasons_analysis", "first_pass_accuracy", "complaints_june_by_portfolio"]

def _score(text: str, pats: list[str]) -> int:
    return sum(1 for p in pats if re.search(p, text, re.IGNORECASE))

def _month_hint(text: str) -> Optional[str]:
    m = _MONTH_RX.search(text)
    return m.group(0) if m else None

def match(q: str) -> Dict:
    """
    Returns a dict with:
      - slug: one of 'complaints_june_by_portfolio', 'first_pass_accuracy', 'fail_reasons_analysis'
      - params: lightweight hints (e.g., 'hint_month') that the question can ignore safely
    """
    text = _norm(q)
    if not text:
        return {"slug": "complaints_june_by_portfolio", "params": {}}

    scores = {slug: _score(text, _TRIGGERS[slug]) for slug in _ORDER}
    best = max(_ORDER, key=lambda s: scores[s])

    # If nothing matched, choose a safe default based on a couple of explicit cues
    if scores[best] == 0:
        if "first" in text and "pass" in text or "fpa" in text:
            best = "first_pass_accuracy"
        elif ("fail" in text and ("reason" in text or "driver" in text)) or "pareto" in text:
            best = "fail_reasons_analysis"
        else:
            best = "complaints_june_by_portfolio"

    params: Dict[str, str] = {}
    mh = _month_hint(text)
    if mh:
        params["hint_month"] = mh

    return {"slug": best, "params": params}
