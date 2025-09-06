# semantic_router.py
from __future__ import annotations
import re
from typing import Dict


def _parse_month_range(q: str) -> Dict:
    m = re.search(r"([a-z]{3}\s*\d{4})\s*to\s*([a-z]{3}\s*\d{4})", q, re.I)
    if not m:
        return {}
    return {"start": m.group(1).title(), "end": m.group(2).title()}


def _parse_portfolio(q: str) -> Dict:
    # "portfolio London", or "for London ..."
    p = re.search(r"\bportfolio\s+([a-z\s]+)", q, re.I)
    if p:
        return {"portfolio": p.group(1).strip().title()}
    p2 = re.search(r"\bfor\s+([a-z\s]+?)\s+(jun|june|jul|aug|sep|oct|nov|dec|\d{4}|to|last|month)", q, re.I)
    if p2:
        return {"portfolio": p2.group(1).strip().title()}
    return {}


def match(q: str) -> Dict:
    ql = q.lower()
    params: Dict = {}
    params.update(_parse_month_range(ql))
    params.update(_parse_portfolio(ql))

    # New dedicated June analysis (by portfolio, no process)
    if "complaint analysis" in ql or "june analysis" in ql:
        return {"slug": "complaints_june_by_portfolio", "params": params}

    # Keep your other routes intact
    if "complaints per 1000" in ql or "complaints per thousand" in ql:
        return {"slug": "complaints_per_thousand", "params": params}
    if "rca1" in ql:
        return {"slug": "rca1_portfolio_process", "params": params}
    if "unique cases" in ql and "mom" in ql or "month on month" in ql or "apr 2025 to jun 2025" in ql:
        return {"slug": "unique_cases_mom", "params": params}

    # Default to June analysis if unsure (safe and fast)
    return {"slug": "complaints_june_by_portfolio", "params": params}
