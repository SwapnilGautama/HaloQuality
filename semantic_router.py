# semantic_router.py
from __future__ import annotations
import re
from typing import Dict, List

# loose month matcher; router only passes through a hint month if present
_MONTH_RX = r"(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s*(\d{2,4})?"

def _norm_q(q: str) -> str:
    return (q or "").strip().lower()

def _any_in(text: str, phrases: List[str]) -> bool:
    return any(p in text for p in phrases)

def match(q: str) -> Dict:
    """
    Small, safe router returning a slug + lightweight params.
    Q1: complaints_june_by_portfolio  (aka "comp" question)
    Q2: first_pass_accuracy           (aka "fpa" question)

    Rules requested by Swap:
      - FPA / Accuracy intents (incl. comparisons & fail-reason analysis) -> Q2
      - Complaints/1000 & RCA intents (incl. locations & typos) -> Q1
    """
    text = _norm_q(q)

    # Detect an optional month hint in the raw text
    m = re.search(_MONTH_RX, text)
    params = {"hint_month": m.group(0)} if m else {}

    # ---------------- FPA / Accuracy triggers -> Q2 ----------------
    fpa_triggers = [
        # canonical
        "fpa", "first pass accuracy", "first-pass accuracy", "first pass",
        "accuracy",

        # comparisons
        "team comparison", "manager comparison", "portfolio comparison",
        "location comparison", "comparison",

        # complaints vs accuracy views under the FPA umbrella
        "complaint vs accuracy", "complaint and accuracy", "compaint accuracy", "complaint accuracy",

        # fail reasons should now route to FPA
        "fail reasons", "fail reason", "fail reason analysis",
    ]

    if _any_in(text, fpa_triggers):
        return {"slug": "first_pass_accuracy", "params": params}

    # ---------------- Complaints / 1000 & RCA triggers -> Q1 ----------------
    comp_triggers = [
        # canonical + common short forms
        "comp", "complaint", "complaints", "complaint analysis",
        "complaint reason", "complaint rca", "rca", "rca by portfolio",

        # metric variants
        "complaints per 1000", "complaints/1000", "complaint/1000",
        "complint/1000", "complaints per thousand", "complaints per k",

        # location-intent phrases (route to comp)
        "chichester", "exeter", "leatherhead", "london", "scotland", "northwest",
        "complains for", "complains by", "complaints for", "complaints by",
    ]

    if _any_in(text, comp_triggers):
        return {"slug": "complaints_june_by_portfolio", "params": params}

    # ---------------- Default fallback ----------------
    # Keep previous behavior: default to the complaints question.
    return {"slug": "complaints_june_by_portfolio", "params": params}
