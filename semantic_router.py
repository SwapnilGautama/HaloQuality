# semantic_router.py
from __future__ import annotations

import re
from datetime import date, datetime, timedelta
from typing import Dict, Tuple, Optional

# ----------------------------
# Helpers
# ----------------------------

_MONTHS = {
    "jan": 1, "january": 1,
    "feb": 2, "february": 2,
    "mar": 3, "march": 3,
    "apr": 4, "april": 4,
    "may": 5,
    "jun": 6, "june": 6,
    "jul": 7, "july": 7,
    "aug": 8, "august": 8,
    "sep": 9, "sept": 9, "september": 9,
    "oct": 10, "october": 10,
    "nov": 11, "november": 11,
    "dec": 12, "december": 12,
}

def _last_complete_month(today: Optional[date] = None) -> date:
    if today is None:
        today = date.today()
    first_this_month = date(today.year, today.month, 1)
    return first_this_month - timedelta(days=1)  # last day of previous month

def _to_month_key(text: str) -> Optional[str]:
    """
    Parse a month reference from free text. Returns YYYY-MM string if found.
    Handles 'Jun 2025', 'June-25', and 'last month'. Defaults to None if no hit.
    """
    t = text.lower().strip()

    # 'last month'
    if re.search(r"\blast\s+month\b", t):
        lm = _last_complete_month()
        return f"{lm.year:04d}-{lm.month:02d}"

    # explicit month + (optional) year
    m = re.search(
        r"\b(jan|january|feb|february|mar|march|apr|april|may|jun|june|jul|july|aug|august|"
        r"sep|sept|september|oct|october|nov|november|dec|december)"
        r"[-\s]*([']?\d{2,4})?\b",
        t
    )
    if not m:
        return None

    mon_txt = m.group(1)
    yr_txt = m.group(2)
    mon = _MONTHS.get(mon_txt, None)
    if mon is None:
        return None

    # Year logic
    if yr_txt:
        yr_txt = yr_txt.replace("'", "")
        year = int(yr_txt)
        if year < 100:
            # interpret '25' as 2025 (simple heuristic)
            year += 2000 if year < 70 else 1900
    else:
        # if no year provided, assume current year (or last complete month year if safer)
        lm = _last_complete_month()
        year = lm.year

    return f"{year:04d}-{mon:02d}"

def _extract_grouping(text: str) -> Optional[str]:
    """
    Detect grouping hint like 'by portfolio' or 'by scheme'.
    Returns 'portfolio' | 'scheme' | None
    """
    t = text.lower()
    if "by scheme" in t or "by plan" in t:
        return "scheme"
    if "by portfolio" in t:
        return "portfolio"
    return None

# ----------------------------
# Public API
# ----------------------------

def route(q: str) -> Tuple[str, Dict]:
    """
    Decide which question module (slug) to run and with what params.
    Must return: (slug, params_dict)

    Slugs used here:
      - 'complaints_june_by_portfolio'   (Q1)
      - 'first_pass_accuracy'            (Q2)
      - 'fail_reasons_analysis'          (Q3, new)
    """
    q_raw = q or ""
    ql = q_raw.strip().lower()

    # --------------- Q3: Fail Reasons Analysis (FRA) -----------------
    # Keep this narrow so it doesn't hijack Q1 / Q2 traffic.
    fra_hits = [
        "fra", "fail reasons", "reason for fail", "reasons for fail",
        "root cause", "fail drivers", "fail pareto", "rca", "reason breakdown"
    ]
    if any(h in ql for h in fra_hits):
        # month is optional for FRA; the module lets user pick, defaulting to latest
        mk = _to_month_key(q_raw)  # None allowed
        params = {}
        if mk:
            params["month"] = mk   # the module can ignore if not used
        return "fail_reasons_analysis", params

    # --------------- Q2: First Pass Accuracy -------------------------
    if ("first pass accuracy" in ql) or re.search(r"\bfpa\b", ql):
        mk = _to_month_key(q_raw)  # optional, Q2 already computes Jan..latest
        params = {}
        if mk:
            params["month"] = mk
        group = _extract_grouping(q_raw)
        if group:
            params["group_by"] = group
        return "first_pass_accuracy", params

    # --------------- Q1: Complaint analysis --------------------------
    # Use existing triggers; keep behaviour stable.
    if ("complaint analysis" in ql) or ("complaints analysis" in ql) or ("complaints" in ql and "analysis" in ql):
        mk = _to_month_key(q_raw) or _to_month_key("last month")
        params = {"month": mk} if mk else {}
        group = _extract_grouping(q_raw) or "portfolio"
        params["group_by"] = group
        return "complaints_june_by_portfolio", params

    # --------------- Fallback: keep Q1 as a gentle default -----------
    # This mirrors previous behaviour in your app so nothing breaks.
    mk = _to_month_key("last month")
    return "complaints_june_by_portfolio", {"month": mk, "group_by": "portfolio"}


# (Optional) chips helpers – used by the header chips in app.py.
# Safe if app.py doesn't call them.
def default_month_label() -> str:
    lm = _last_complete_month()
    return f"{lm.strftime('%b %Y')}"

def chips() -> Tuple[Tuple[str, str], ...]:
    """
    Returns UI chip label and the query string to seed the input.
    """
    lm_label = _last_complete_month().strftime("%b %Y")
    return (
        (f"complaint analysis — {lm_label} (by portfolio)", f"complaint analysis — {lm_label} (by portfolio)"),
        ("first pass accuracy analysis", "first pass accuracy analysis"),
        ("FRA — fail reasons (Pareto)", "FRA fail reasons"),
    )
