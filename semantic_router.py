# semantic_router.py
"""
semantic_router.py — tiny semantic matcher for Halo questions

Returns an IntentMatch(slug, title, params) where slug corresponds to a module under questions/,
and params is a dict we pass to run(store, params, user_text="").

We use RapidFuzz similarity against a curated lexicon + a little date parsing.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from typing import Dict, List, Optional, Tuple

from rapidfuzz import process, fuzz


# ---------- Public dataclass ----------

@dataclass
class IntentMatch:
    slug: str
    title: str
    params: Dict[str, str]


# ---------- Lexicon ----------

# Canonical slugs -> surface forms we match against
LEXICON: Dict[str, List[str]] = {
    # 1) Complaints per 1000 cases
    "complaints_per_thousand": [
        "complaints per 1000",
        "complaints per thousand",
        "complaints per 1,000",
        "per 1000 cases",
        "complaint rate per 1000",
        "complaint rate",
        "complaints/1000",
        "rate per thousand",
    ],

    # 2) RCA1 by portfolio x process
    "rca1_portfolio_process": [
        "rca1 by portfolio",
        "rca1 by portfolio by process",
        "rca by portfolio",
        "root cause analysis 1",
        "rca1 x process",
        "rca1 × process",
        "rca1 portfolio process",
        "show rca1",
    ],

    # 3) Unique cases month-over-month
    "unique_cases_mom": [
        "unique cases",
        "distinct cases",
        "unique case count",
        "unique cases by process and portfolio",
        "month over month cases",
        "cases mom",
    ],
}

# For nicer chart titles
TITLES: Dict[str, str] = {
    "complaints_per_thousand": "Complaints per 1,000 cases",
    "rca1_portfolio_process": "RCA1 by Portfolio × Process",
    "unique_cases_mom": "Unique cases (MoM)",
}

# Some common process aliases
PROCESS_ALIASES = {
    "member enquiry": "Member Enquiry",
    "member inquiry": "Member Enquiry",
    "member enquiries": "Member Enquiry",
    "claims": "Claims",
    "claim": "Claims",
}

# Some city/portfolio aliases (extend as needed)
PORTFOLIO_ALIASES = {
    "london": "London",
}


# ---------- Date parsing utilities ----------

_MONTH_NAME = r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)"
_YEAR = r"(20\d{2})"

EXPLICIT_RANGE_RE = re.compile(
    rf"{_MONTH_NAME}\s+{_YEAR}\s+(?:to|-)\s+{_MONTH_NAME}\s+{_YEAR}", re.IGNORECASE
)
SINGLE_MONTH_RE = re.compile(rf"{_MONTH_NAME}\s+{_YEAR}", re.IGNORECASE)
LAST_N_MONTHS_RE = re.compile(r"last\s+(\d+)\s+month", re.IGNORECASE)

_MONTH_INDEX = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}

def _mkey(m: str) -> int:
    return _MONTH_INDEX[m[:3].lower()]

def _ym(yy: str, mm: str) -> str:
    """YYYY-MM canonical (month 2-digits)."""
    return f"{int(yy):04d}-{int(mm):02d}"

def _last_complete_month() -> Tuple[int, int]:
    today = date.today()
    y, m = today.year, today.month
    # Treat current month as in-progress; use previous month
    m -= 1
    if m == 0:
        y -= 1
        m = 12
    return y, m


def _parse_months(text: str) -> Dict[str, str]:
    """
    Extract {start_month, end_month, relative_months} if present.
    Months formatted as 'YYYY-MM'. If no dates in text, returns {}.
    """
    params: Dict[str, str] = {}

    # last N months
    m_rel = LAST_N_MONTHS_RE.search(text)
    if m_rel:
        params["relative_months"] = str(int(m_rel.group(1)))
        # Also produce concrete start/end relative to "now" so modules can use it directly if they want
        end_y, end_m = _last_complete_month()
        n = int(params["relative_months"])
        # compute start month
        start_y, start_m = end_y, end_m
        for _ in range(n - 1):
            start_m -= 1
            if start_m == 0:
                start_m = 12
                start_y -= 1
        params["end_month"] = _ym(end_y, end_m)
        params["start_month"] = _ym(start_y, start_m)
        return params

    # explicit range, e.g. "Apr 2025 to Jun 2025"
    m_rng = EXPLICIT_RANGE_RE.search(text)
    if m_rng:
        m1, y1, m2, y2 = m_rng.group(1), m_rng.group(2), m_rng.group(3), m_rng.group(4)
        params["start_month"] = _ym(int(y1), _mkey(m1))
        params["end_month"] = _ym(int(y2), _mkey(m2))
        return params

    # single month (we'll populate both start & end with same)
    m_single = SINGLE_MONTH_RE.search(text)
    if m_single:
        m, y = m_single.group(1), m_single.group(2)
        params["start_month"] = _ym(int(y), _mkey(m))
        params["end_month"] = params["start_month"]
        return params

    return params


def _extract_portfolio(text: str) -> Optional[str]:
    for k, v in PORTFOLIO_ALIASES.items():
        if re.search(rf"\b{k}\b", text, re.IGNORECASE):
            return v
    # simple fallback: look for 'portfolio <word>'
    m = re.search(r"portfolio\s+([A-Za-z]+)", text, re.IGNORECASE)
    if m:
        word = m.group(1).strip(",. ")
        return PORTFOLIO_ALIASES.get(word.lower(), word.title())
    return None


def _extract_process(text: str) -> Optional[str]:
    # known aliases
    for k, v in PROCESS_ALIASES.items():
        if re.search(rf"\b{k}\b", text, re.IGNORECASE):
            return v
    # generic 'process <name>'
    m = re.search(r"process\s+([A-Za-z ]+?)(?:\s+last|\s+for|\s+by|$)", text, re.IGNORECASE)
    if m:
        return m.group(1).strip(",. ").title()
    return None


# ---------- Main matcher ----------

def match_query(text: str) -> Optional[IntentMatch]:
    """
    Return IntentMatch(slug, title, params) or None if no intent is above threshold.
    """
    q = text.strip()
    if not q:
        return None

    # 1) choose best slug by fuzzy match
    choices = []
    for slug, phrases in LEXICON.items():
        choices.extend([(slug, p) for p in phrases])

    # rapidfuzz against all phrases; take highest score and its slug
    best_slug, best_score = None, 0
    for slug, phrase in choices:
        score = fuzz.token_set_ratio(q, phrase)
        if score > best_score:
            best_score = score
            best_slug = slug

    # threshold: 70 is permissive but avoids many false positives
    if not best_slug or best_score < 70:
        return None

    # 2) extract params (dates, portfolio, process etc.)
    params: Dict[str, str] = {}
    params.update(_parse_months(q))

    portfolio = _extract_portfolio(q)
    if portfolio:
        params["portfolio"] = portfolio

    process_name = _extract_process(q)
    if process_name:
        params["process"] = process_name

    # Normalize param keys questions care about
    # (All question modules receive the same dict and can ignore what they don't need.)
    title = TITLES.get(best_slug, best_slug.replace("_", " ").title())
    return IntentMatch(slug=best_slug, title=title, params=params)
