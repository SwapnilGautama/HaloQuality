# semantic_router.py
from __future__ import annotations
import re
from typing import Dict

_MONTH_RX = r"(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s*(\d{2,4})?"

def _norm_q(q: str) -> str:
    return (q or "").strip().lower()

def match(q: str) -> Dict:
    """
    Very small, safe router that only returns a slug + lightweight params.
    Q1: complaints_june_by_portfolio
    Q2: first_pass_accuracy
    """
    text = _norm_q(q)

    # Q2 trigger phrases
    if any(p in text for p in ["first pass", "fpa", "first-pass accuracy"]):
        # optional: pick a month if the user mentions one; Q2 still handles full-range internally
        m = re.search(_MONTH_RX, text)
        params = {"hint_month": m.group(0)} if m else {}
        return {"slug": "first_pass_accuracy", "params": params}

    # Default to Q1
    return {"slug": "complaints_june_by_portfolio", "params": {}}
