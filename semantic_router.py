# semantic_router.py
from __future__ import annotations
import re
from typing import Dict

def _to_month_key(text: str) -> str | None:
    """
    Accept 'Jun 2025', 'June 2025', or 'Jun'/'June' (assume 2025)
    Return 'YYYY-MM' when possible.
    """
    m = re.search(r"(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s*(\d{4})?", text, re.I)
    if not m:
        return None
    mon = m.group(1)
    yr = int(m.group(2)) if m.group(2) else 2025
    import pandas as pd
    return pd.to_datetime(f"1 {mon} {yr}", errors="coerce").to_period("M").astype(str)

def match(q: str) -> Dict:
    ql = q.lower()
    params: Dict[str, str] = {}

    mk = _to_month_key(ql)
    if mk:
        params["month"] = mk

    # complaint analysis intent
    if "complaint analysis" in ql or "complaints dashboard" in ql:
        return {"slug": "complaints_june_by_portfolio", "params": params}

    # default to complaints_june_by_portfolio for now
    return {"slug": "complaints_june_by_portfolio", "params": params}
