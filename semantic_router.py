# semantic_router.py
from __future__ import annotations
import re
from typing import Dict, Optional, Tuple
import pandas as pd

# --- helpers ---------------------------------------------------------------

_MONTH_RE = re.compile(
    r"(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*[\s\-\/]*(\d{2,4})?",
    re.I,
)

def _month_key_from_text(text: str) -> Tuple[str, str]:
    """
    Returns (month_key, month_label)
    month_key -> 'YYYY-MM' (e.g., '2025-06')
    month_label -> 'June 2025'
    Defaults to June 2025 if not present.
    """
    m = _MONTH_RE.search(text or "")
    if not m:
        dt = pd.Timestamp(2025, 6, 1)
        return dt.strftime("%Y-%m"), dt.strftime("%B %Y")

    mon = m.group(1).lower()
    yr  = m.group(2)
    mon_num = {
        "jan":1,"feb":2,"mar":3,"apr":4,"may":5,"jun":6,
        "jul":7,"aug":8,"sep":9,"oct":10,"nov":11,"dec":12
    }[mon[:3]]

    if yr is None:
        year = 2025
    else:
        yi = int(yr)
        year = 2000 + yi if yi < 100 else yi

    dt = pd.Timestamp(year, mon_num, 1)
    return dt.strftime("%Y-%m"), dt.strftime("%B %Y")


# --- public API ------------------------------------------------------------

def route(q: str) -> Tuple[Optional[str], Dict]:
    """
    Returns (slug, params)
    slug ∈ {'complaints_june_by_portfolio', 'first_pass_accuracy', None}
    params adds what Q1 needs when slug is Q1.
    """
    if not q:
        return None, {}

    t = q.lower()

    # Q1: complaints analysis
    if ("complaint" in t and "portfolio" in t) or "complaints_june_by_portfolio" in t:
        mk, ml = _month_key_from_text(q)
        params = {
            "month_key": mk,             # '2025-06'
            "month_label": ml,           # 'June 2025'
            "portfolio": None,           # overall
        }
        return "complaints_june_by_portfolio", params

    # Q2: FPA analysis
    if ("first pass" in t and "accuracy" in t) or "first_pass_accuracy" in t:
        # FPA does its own month spans; no special params needed here.
        return "first_pass_accuracy", {}

    return None, {}
