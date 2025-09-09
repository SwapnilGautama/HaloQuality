# semantic_router.py
from __future__ import annotations
import re
from typing import Dict

# loose month matcher; router only passes through a hint month if present
_MONTH_RX = r"(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s*(\d{2,4})?"

def _norm_q(q: str) -> str:
    return (q or "").strip().lower()

def match(q: str) -> Dict:
    """
    Very small, safe router that only returns a slug + lightweight params.
    Q1: complaints_june_by_portfolio
    Q2: first_pass_accuracy
    Q3: fail_reasons_analysis
    Q4: nps_by_portfolio
    """
    text = _norm_q(q)

    # NPS triggers (Q4)
    nps_trigs = [
        "nps", "net promoter", "promoter score", "promoters", "detractors", "passives",
        "nps by portfolio", "nps trend", "net promoter score"
    ]
    if any(p in text for p in nps_trigs):
        m = re.search(_MONTH_RX, text)
        params = {"hint_month": m.group(0)} if m else {}
        return {"slug": "nps_by_portfolio", "params": params}

    # Q2 trigger phrases
    if any(p in text for p in ["first pass", "fpa", "first-pass accuracy", "first pass accuracy"]):
        m = re.search(_MONTH_RX, text)
        params = {"hint_month": m.group(0)} if m else {}
        return {"slug": "first_pass_accuracy", "params": params}

    # Q3 trigger phrases
    if any(p in text for p in ["fail reasons", "fra", "fail reasons analysis"]):
        m = re.search(_MONTH_RX, text)
        params = {"hint_month": m.group(0)} if m else {}
        return {"slug": "fail_reasons_analysis", "params": params}

    # Default to Q1
    return {"slug": "complaints_june_by_portfolio", "params": {}}
