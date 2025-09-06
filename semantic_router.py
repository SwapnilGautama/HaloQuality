# semantic_router.py
from __future__ import annotations
import re
from typing import Dict, Any, Optional

import pandas as pd


def _to_month_key(text: str) -> Optional[str]:
    """
    Accept 'Jun', 'June', 'Jun 2025', 'June 2025' (case-insensitive).
    If year missing, assume 2025. Return 'YYYY-MM' or None if parsing fails.
    """
    m = re.search(
        r"(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s*(\d{4})?",
        text,
        re.I,
    )
    if not m:
        return None
    mon = m.group(1)
    yr = int(m.group(2)) if m.group(2) else 2025

    # Build a scalar Timestamp safely and format ourselves (no .to_period on scalars)
    dt = pd.to_datetime(f"1 {mon} {yr}", errors="coerce", dayfirst=True)
    if pd.isna(dt):
        return None
    return f"{dt.year:04d}-{dt.month:02d}"


def _parse_portfolio(q: str) -> Dict[str, str]:
    """
    Try to extract a portfolio name if the query says 'portfolio <name>' or 'for <name> ...'.
    """
    p = re.search(r"\bportfolio\s+([a-z\s]+)", q, re.I)
    if p:
        return {"portfolio": p.group(1).strip().title()}

    p2 = re.search(
        r"\bfor\s+([a-z\s]+?)\s+(jun|jul|aug|sep|oct|nov|dec|\d{4}|to|last|month)",
        q,
        re.I,
    )
    if p2:
        return {"portfolio": p2.group(1).strip().title()}
    return {}


def match(q: str) -> Dict[str, Any]:
    """
    Return a route dict with a slug and params. Default to the
    complaints-by-portfolio month view.
    """
    ql = q.lower()
    params: Dict[str, Any] = {}

    mk = _to_month_key(ql)
    if mk:
        params["month"] = mk

    params.update(_parse_portfolio(ql))

    if "complaint analysis" in ql or "complaints dashboard" in ql:
        return {"slug": "complaints_june_by_portfolio", "params": params}

    # keep working default
    return {"slug": "complaints_june_by_portfolio", "params": params}
