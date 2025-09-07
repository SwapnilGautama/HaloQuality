from __future__ import annotations
import re
from typing import Dict, Any, Optional
import pandas as pd

def _to_month_key(text: str) -> Optional[str]:
    m = re.search(
        r"(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s*(\d{4})?",
        text,
        re.I,
    )
    if not m:
        return None
    mon = m.group(1)
    yr = int(m.group(2)) if m.group(2) else 2025
    dt = pd.to_datetime(f"1 {mon} {yr}", errors="coerce", dayfirst=True)
    if pd.isna(dt):
        return None
    return f"{dt.year:04d}-{dt.month:02d}"

def _parse_portfolio(q: str) -> Dict[str, str]:
    p = re.search(r"\bportfolio\s+([a-z\s]+)", q, re.I)
    if p:
        return {"portfolio": p.group(1).strip().title()}
    p2 = re.search(
        r"\bfor\s+([a-z\s]+?)\s+(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|\d{4}|to|last|month)",
        q,
        re.I,
    )
    if p2:
        return {"portfolio": p2.group(1).strip().title()}
    return {}

def match(q: str) -> Dict[str, Any]:
    ql = q.lower()
    params: Dict[str, Any] = {}
    mk = _to_month_key(ql)
    if mk:
        params["month"] = mk
    params.update(_parse_portfolio(ql))

    if any(k in ql for k in [
        "first pass accuracy", "first-pass accuracy", "first pass", "fpa", "accuracy analysis"
    ]):
        return {"slug": "first_pass_accuracy", "params": params}

    if any(k in ql for k in [
        "complaint analysis", "complaints dashboard", "complaints analysis"
    ]):
        return {"slug": "complaints_june_by_portfolio", "params": params}

    return {"slug": "complaints_june_by_portfolio", "params": params}
