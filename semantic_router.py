# semantic_router.py
from __future__ import annotations
import re
from typing import Dict

def _parse_month_range(q: str):
    m = re.search(r"([a-z]{3}\s*\d{4})\s*to\s*([a-z]{3}\s*\d{4})", q, re.I)
    if not m:
        return {}
    return {"start": m.group(1).title(), "end": m.group(2).title()}

def _parse_portfolio(q: str):
    # portfolio <name> OR "for London" etc.
    p = re.search(r"\bportfolio\s+([a-z\s]+)", q, re.I)
    if p:
        return {"portfolio": p.group(1).strip().title()}
    p2 = re.search(r"\bfor\s+([a-z\s]+?)\s+(jun|jul|aug|sep|oct|nov|dec|\d{4}|to|last|month)", q, re.I)
    if p2:
        return {"portfolio": p2.group(1).strip().title()}
    return {}

def match(q: str) -> Dict:
    ql = q.lower()
    params = {}
    params.update(_parse_month_range(ql))
    params.update(_parse_portfolio(ql))

    if "complaints per 1000" in ql or "complaints per thousand" in ql:
        return {"slug": "complaints_per_thousand", "params": params}
    if "complaint analysis" in ql or "complaints dashboard" in ql or "june analysis" in ql:
        return {"slug": "complaints_june_by_portfolio", "params": params}
    return {"slug": "complaints_per_thousand", "params": params}
