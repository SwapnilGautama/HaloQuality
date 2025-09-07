# -*- coding: utf-8 -*-
# semantic_router.py — very small router for two questions

from __future__ import annotations
import re
from typing import Dict, Optional
import pandas as pd  # (kept: sometimes used by callers for month coercion)

# ----------------------------- helpers ----------------------------------------
def _to_month_key(text: str) -> Optional[str]:
    """
    Extract 'mmm yyyy' like 'Jun 2025' from free text and normalise to YYYY-MM.
    """
    m = re.search(r"(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s*(\d{4})", text, re.I)
    if not m:
        return None
    mon = m.group(1).lower()
    yr = int(m.group(2))
    # map to month number
    mm = {
        "jan":"01","feb":"02","mar":"03","apr":"04","may":"05","jun":"06",
        "jul":"07","aug":"08","sep":"09","oct":"10","nov":"11","dec":"12"
    }[mon]
    return f"{yr}-{mm}"

# ----------------------------- routers ----------------------------------------
def match(q: str) -> Optional[Dict]:
    qs = q.strip().lower()

    # First Pass Accuracy
    if any(k in qs for k in ["first pass accuracy", "first-pass accuracy", "fpa"]):
        # Optional month if specified (not required)
        mon = _to_month_key(q)  # returns 'YYYY-MM' or None
        return {
            "slug": "first_pass_accuracy",   # questions/first_pass_accuracy.py
            "params": {"month": mon}         # None -> show full Jan-2025..latest
        }

    # Default: complaints
    if any(k in qs for k in ["complaint analysis", "complaints analysis", "complaints"]):
        mon = _to_month_key(q) or "2025-06"  # default Jun-25 to keep your slide view
        return {
            "slug": "complaints_june_by_portfolio",   # (your existing module)
            "params": {"month": mon}
        }

    return None
