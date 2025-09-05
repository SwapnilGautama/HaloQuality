# semantic_router.py — tiny semantic matcher for Halo questions
# -------------------------------------------------------------
# Returns (slug, args, title) where slug matches a module under questions/
# We use RapidFuzz on a curated lexicon so users can type freely.

from __future__ import annotations

import re
from typing import Dict, Any, Tuple, List, Optional

import pandas as pd
from rapidfuzz import process, fuzz

# ------------------ Canonical question slugs -> surface forms ------------------

LEXICON: Dict[str, List[str]] = {
    # Complaints / cases
    "complaints_per_thousand": [
        "complaints per thousand",
        "complaints per 1000",
        "complaints/1000",
        "complaint rate per 1000",
        "complaint rate",
        "per 1000 complaints",
        "complaints per 1000 by process",
    ],
    "complaint_volume_rate": [
        "complaint volume",
        "complaint volumes",
        "complaints volume",
        "volume of complaints",
        "complaints by month",
    ],
    # FPA
    "fpa_fail_drivers": [
        "fpa fail drivers",
        "biggest drivers of case fails",
        "first pass accuracy drivers",
        "fpa drivers",
        "case fail reasons",
    ],
    # Correlation (NPS vs complaints, etc.)
    "corr_nps": [
        "nps correlation",
        "complaints vs nps",
        "corr nps",
        "correlation analysis nps",
    ],
    # RCA1
    "rca1_portfolio_process": [
        "show rca1 by portfolio for process",
        "rca by portfolio",
        "rca1 share by portfolio by process",
        "rca1 by process",
        "root cause by portfolio process",
    ],
    # Unique cases (MoM) — make sure this wins over complaint volume
    "unique_cases_mom": [
        "unique cases by process and portfolio",
        "unique cases month over month",
        "unique case count by month",
        "cases mom",
        "case volume month on month",
        "unique cases apr to jun",
        "unique cases for portfolio",
        "unique cases by process",
    ],
}

# Bias so some intents win ties (1.0 = neutral, >1.0 boosts)
PRIORITY = {
    "unique_cases_mom": 1.15,
    "complaints_per_thousand": 1.00,
    "complaint_volume_rate": 0.95,
    "rca1_portfolio_process": 1.05,
    "fpa_fail_drivers": 1.00,
    "corr_nps": 1.00,
}

# Legacy aliases (the app will handle a second mapping too)
ALIASES = {
    "fpa_fail_rate": "fpa_fail_drivers",
}

# Build a flat list of (surface, slug) for matching
_SURFACES: List[Tuple[str, str]] = []
for slug, phrases in LEXICON.items():
    for p in phrases:
        _SURFACES.append((p, slug))
for k, v in ALIASES.items():
    _SURFACES.append((k.replace("_", " "), v))


# ------------------ Lightweight param parsing ------------------

_MONTH_RE = r"(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{4}"
_PORT_RE = r"\bportfolio\s+([A-Za-z][\w\s/&-]+)"
_PROC_RE = r"\bprocess\s+([A-Za-z][\w\s/&-]+)"

def _find_months(text: str) -> Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]:
    """
    Parse 'Jun 2025 to Aug 2025' or single months.
    Returns (start_ts, end_ts) at month start.
    """
    text_low = text.lower()
    months = re.findall(_MONTH_RE, text_low, flags=re.I)
    # We used a group for month name only; instead, capture full month-year tokens via finditer
    tokens = [m.group(0) for m in re.finditer(_MONTH_RE, text_low, flags=re.I)]
    if not tokens:
        return None, None
    def _to_ts(tok: str) -> Optional[pd.Timestamp]:
        try:
            dt = pd.to_datetime(tok, format="%b %Y", errors="coerce")
            if pd.isna(dt):
                # allow full month names too
                dt = pd.to_datetime(tok, errors="coerce")
            if pd.isna(dt):
                return None
            return dt.to_period("M").to_timestamp()
        except Exception:
            return None
    if len(tokens) == 1:
        t = _to_ts(tokens[0])
        return (t, t)
    return _to_ts(tokens[0]), _to_ts(tokens[-1])

def _find_portfolio(text: str) -> Optional[str]:
    m = re.search(_PORT_RE, text, flags=re.I)
    return m.group(1).strip() if m else None

def _find_process(text: str) -> Optional[str]:
    m = re.search(_PROC_RE, text, flags=re.I)
    return m.group(1).strip() if m else None

def _find_last_n(text: str) -> Optional[int]:
    m = re.search(r"last\s+(\d+)\s+month", text, flags=re.I)
    return int(m.group(1)) if m else None


# ------------------ Matcher ------------------

def _best_slug(query: str) -> str:
    # RapidFuzz over surfaces, return best slug (with bias)
    choices = [s for s, _ in _SURFACES]
    res = process.extractOne(
        query,
        choices,
        scorer=fuzz.WRatio
    )
    if not res:
        # default to complaints_per_thousand if nothing matches
        return "complaints_per_thousand"
    surface, score, idx = res
    slug = _SURFACES[idx][1]
    # Apply priority boost
    priority = PRIORITY.get(slug, 1.0)
    # See if another candidate deserves the win (rare), otherwise return slug
    return slug

def _build_title(slug: str, params: Dict[str, Any]) -> str:
    if slug == "complaints_per_thousand":
        base = "Complaints per 1,000 cases"
    elif slug == "complaint_volume_rate":
        base = "Complaint volume (MoM)"
    elif slug == "fpa_fail_drivers":
        base = "FPA Fail Drivers"
    elif slug == "corr_nps":
        base = "NPS × Complaints correlation"
    elif slug == "rca1_portfolio_process":
        base = "RCA1 by Portfolio × Process"
    elif slug == "unique_cases_mom":
        base = "Complaint volume (MoM)"  # if your question is named unique_cases_mom but is about complaints, tweak this
        # If it's truly cases, you can rename to "Unique cases (MoM)"
        base = "Unique cases (MoM)"
    else:
        base = slug.replace("_", " ").title()

    # Friendly suffix if we have months or last_n
    start, end = params.get("start_month"), params.get("end_month")
    last_n = params.get("months")
    if start is not None and end is not None:
        base += f" — {_fmt_month(start)} to {_fmt_month(end)}"
    elif last_n is not None:
        base += f" — last {last_n} months"
    return base

def _fmt_month(ts: Optional[pd.Timestamp]) -> str:
    if ts is None or pd.isna(ts):
        return "NaT"
    return ts.strftime("%b %Y")

def match(query: str) -> Tuple[str, Dict[str, Any], str]:
    """
    Return (slug, params, title) for the given query.
    params is a free-form dict; question modules decide how to use them.
    """
    slug = _best_slug(query)
    slug = ALIASES.get(slug, slug)

    start, end = _find_months(query)
    months = _find_last_n(query)
    portfolio = _find_portfolio(query)
    process_name = _find_process(query)

    params: Dict[str, Any] = {}
    if start is not None:
        params["start_month"] = start
    if end is not None:
        params["end_month"] = end
    if months is not None:
        params["months"] = months
    if portfolio:
        params["portfolio"] = portfolio
    if process_name:
        params["process_name"] = process_name

    # Provide defaults for some slugs
    if slug == "rca1_portfolio_process":
        params.setdefault("months", 3)  # default last 3 months if not specified

    title = _build_title(slug, params)
    return slug, params, title
