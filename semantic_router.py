# semantic_router.py
# Tiny, robust semantic matcher for HaloQuality
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

# --- helpers ---------------------------------------------------------------

_MONTH = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}

def _to_ym(month_name: str, year: str) -> str:
    m = _MONTH[month_name.lower()[:3]]
    return f"{int(year):04d}-{m:02d}"

def _find_month_range(text: str) -> Optional[Tuple[str, str]]:
    """
    Find 'Apr 2025 to Jun 2025' → ('2025-04','2025-06')
    Case-insensitive. Returns None if not found.
    """
    rx = re.compile(
        r"\b("
        r"jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec"
        r")\s+(\d{4})\s+to\s+("
        r"jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec"
        r")\s+(\d{4})",
        flags=re.I,
    )
    m = rx.search(text)
    if not m:
        return None
    m1, y1, m2, y2 = m.group(1, 2, 3, 4)
    return _to_ym(m1, y1), _to_ym(m2, y2)

def _find_last_n_months(text: str) -> Optional[int]:
    m = re.search(r"\blast\s+(\d+)\s+months?\b", text, flags=re.I)
    return int(m.group(1)) if m else None

def _find_process(text: str) -> Optional[str]:
    # crude, but works for “process <name>” or “… for process <name> …”
    m = re.search(r"\bprocess\s+([A-Za-z][A-Za-z\s&/,-]{1,60})", text, flags=re.I)
    if not m:
        return None
    # trim trailing 'last N months' or months range words if they leaked in
    val = re.split(r"\blast\s+\d+\s+months?|\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\b", m.group(1), flags=re.I)[0]
    return val.strip(" ,-/")

def _find_portfolio(text: str) -> Optional[str]:
    # “… for portfolio London …” OR “… by portfolio London …”
    m = re.search(r"\bfor\s+portfolio\s+([A-Za-z][A-Za-z\s&/,-]{1,60})", text, flags=re.I)
    if not m:
        m = re.search(r"\bby\s+portfolio\s+([A-Za-z][A-Za-z\s&/,-]{1,60})", text, flags=re.I)
    if not m:
        return None
    val = re.split(r"\blast\s+\d+\s+months?|\bto\b", m.group(1), flags=re.I)[0]
    return val.strip(" ,-/")

# --- public API ------------------------------------------------------------

@dataclass
class IntentMatch:
    slug: str
    args: Dict[str, object]
    title: str

def match_query(query: str) -> Optional[IntentMatch]:
    """
    Return an IntentMatch or None. Handles the three core questions:
      - complaints_per_thousand
      - rca1_portfolio_process
      - unique_cases_mom
    """
    q = query.strip()
    q_lc = q.lower()

    # 1) complaints per 1000 ...
    if "complaints per 1000" in q_lc or "complaints per thousand" in q_lc:
        months = _find_month_range(q)
        portfolio = _find_portfolio(q)
        args: Dict[str, object] = {}
        if months:
            args["start_month"], args["end_month"] = months
        if portfolio:
            args["portfolio"] = portfolio
        title = "Complaints per 1,000 cases"
        return IntentMatch("complaints_per_thousand", args, title)

    # 2) rca1 by portfolio for process … last N months
    if "rca1" in q_lc and "process" in q_lc:
        n = _find_last_n_months(q) or 3
        proc = _find_process(q)
        port = _find_portfolio(q)  # optional
        args = {"relative_months": n}
        if proc:
            args["process"] = proc
        if port:
            args["portfolio"] = port
        title = f"RCA1 by Portfolio × Process — last {n} months"
        return IntentMatch("rca1_portfolio_process", args, title)

    # 3) unique cases by process and portfolio <Apr 2025 to Jun 2025>
    if "unique cases" in q_lc:
        months = _find_month_range(q)
        port = _find_portfolio(q)  # optional
        args: Dict[str, object] = {}
        if months:
            args["start_month"], args["end_month"] = months
        if port:
            args["portfolio"] = port
        title = "Unique cases (MoM)"
        return IntentMatch("unique_cases_mom", args, title)

    return None
