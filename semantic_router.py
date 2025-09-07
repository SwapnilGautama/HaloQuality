# semantic_router.py
from __future__ import annotations

import re
from typing import Dict, Tuple, Optional

# --- simple month parser (optional; safe if no month present) ---
_MONTHS = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}

def _parse_month_key(q: str) -> Optional[str]:
    """
    Extract something like 'June 2025' / 'Jun 2025' / 'jun-25' from the query and
    return 'YYYY-MM' (e.g., '2025-06'). If nothing found, return None.
    """
    ql = q.lower()
    m = re.search(r"(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*[^0-9]{0,3}(\d{2,4})", ql)
    if not m:
        return None
    mon = _MONTHS[m.group(1)[:3]]
    yr = int(m.group(2))
    if yr < 100:  # 25 -> 2025 (crude but fine for our scope)
        yr = 2000 + yr
    return f"{yr:04d}-{mon:02d}"

def route(q: str) -> Tuple[str, Dict]:
    """
    Decide which question to run and return (slug, params).
    Slugs:
      - 'first_pass_accuracy'                 -> questions/first_pass_accuracy.py
      - 'complaints_june_by_portfolio'       -> questions/complaints_june_by_portfolio.py
    """
    ql = (q or "").lower().strip()

    # FPA intents
    if any(k in ql for k in ["first pass", "first-pass", "fpa", "accuracy analysis", "firstpass"]):
        params: Dict = {}
        mk = _parse_month_key(ql)
        if mk:
            params["month_key"] = mk
        return "first_pass_accuracy", params

    # Complaints intents (default)
    if any(k in ql for k in ["complaint", "complaints", "rca", "portfolio"]):
        params = {}
        mk = _parse_month_key(ql)
        if mk:
            params["month_key"] = mk
        return "complaints_june_by_portfolio", params

    # Fallback to complaints
    return "complaints_june_by_portfolio", {}
