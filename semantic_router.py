# semantic_router.py
from __future__ import annotations
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Any, Optional

MONTHS = {
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

def _month_to_num(tok: str) -> Optional[int]:
    return MONTHS.get(tok.lower())

def _parse_month_year(chunk: str) -> Optional[datetime]:
    # Accept "Jun 2025" / "June 2025" / "Jun'25" / "Jun-2025" / "Jun"
    m = re.search(r"\b([a-z]{3,9})\s*[-']?\s*(\d{2,4})?\b", chunk, re.I)
    if not m:
        return None
    mon = _month_to_num(m.group(1))
    if not mon:
        return None
    yr_txt = m.group(2)
    if yr_txt:
        yr = int(yr_txt)
        if yr < 100:
            yr = 2000 + yr
    else:
        # if year missing we’ll fill in later (questions can override or use store default)
        yr = None
    if yr is None:
        return None
    return datetime(yr, mon, 1)

def _parse_range(text: str) -> (Optional[datetime], Optional[datetime]):
    # explicit "Jun 2025 to Aug 2025" / "Jun to Aug 2025" / "Jun 2025 - Aug 2025"
    rgx = re.compile(
        r"(?P<a>[A-Za-z]{3,9}(?:\s*['-]?\s*\d{2,4})?)\s*(?:to|-|–|—)\s*(?P<b>[A-Za-z]{3,9}(?:\s*['-]?\s*\d{2,4})?)",
        re.I,
    )
    m = rgx.search(text)
    if m:
        a = _parse_month_year(m.group("a"))
        b = _parse_month_year(m.group("b"))
        return a, b
    return None, None

def _parse_last_n_months(text: str) -> Optional[int]:
    m = re.search(r"last\s+(\d{1,2})\s+months?", text, re.I)
    return int(m.group(1)) if m else None

def _parse_portfolio(text: str) -> Optional[str]:
    # accept "portfolio London" or "for portfolio London"
    m = re.search(r"\bportfolio\s+([A-Za-z0-9 _-]+)", text, re.I)
    if m:
        return m.group(1).strip().lower()
    # also accept "for london" if preceded by "portfolio" chip context often omitted
    m = re.search(r"\bfor\s+([A-Za-z0-9 _-]+)\b", text, re.I)
    return m.group(1).strip().lower() if m else None

@dataclass
class Match:
    slug: str
    params: Dict[str, Any]

def match_query(user_text: str) -> Match:
    t = user_text.strip()

    # SLUG detection
    t_norm = t.lower()
    if "per 1000" in t_norm or "per thousand" in t_norm:
        slug = "complaints_per_thousand"
    elif "rca1" in t_norm:
        slug = "rca1_portfolio_process"
    elif "unique cases" in t_norm and "mom" in t_norm or "month" in t_norm:
        slug = "unique_cases_mom"
    else:
        # best guess
        slug = "complaints_per_thousand"

    # portfolio
    portfolio = _parse_portfolio(t)

    # date window
    start_dt, end_dt = _parse_range(t)
    last_n = _parse_last_n_months(t)

    params: Dict[str, Any] = {}
    if portfolio:
        params["portfolio"] = portfolio

    if start_dt and end_dt:
        params["start_month"] = start_dt.strftime("%Y-%m-01")
        params["end_month"] = end_dt.strftime("%Y-%m-01")
    elif last_n:
        params["last_n_months"] = last_n
    # else leave empty; question modules will fallback to last 3 months, or min/max in data

    return Match(slug=slug, params=params)
