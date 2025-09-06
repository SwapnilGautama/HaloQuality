# semantic_router.py — tiny semantic matcher that returns (slug, params, title)

from __future__ import annotations

import re
from datetime import datetime
from typing import Dict, Tuple, List

from rapidfuzz import process, fuzz

# Canonical slugs (file names under questions/)
# - complaints_per_thousand.py
# - rca1_portfolio_process.py
# - unique_cases_mom.py
_CHOICES = [
    ("complaints_per_thousand", "Complaints per 1,000 cases"),
    ("rca1_portfolio_process", "RCA1 by Portfolio × Process — last 3 months"),
    ("unique_cases_mom", "Unique cases (MoM) — Apr 2025 to Jun 2025"),
]

# Lexicon: map slug -> surface forms to fuzzy match
LEXICON: Dict[str, List[str]] = {
    "complaints_per_thousand": [
        "complaints per 1000",
        "complaints/1000",
        "complaint rate per 1000",
        "complaint rate",
        "per 1000 complaints",
        "complaints per thousand",
    ],
    "rca1_portfolio_process": [
        "rca1 by portfolio",
        "rca by portfolio",
        "root cause by portfolio",
        "rca1 last months",
        "rca1 last 3 months",
        "root cause analysis",
    ],
    "unique_cases_mom": [
        "unique cases",
        "unique cases by month",
        "unique cases mom",
        "distinct cases",
    ],
}

# ---------------- date parsing helpers ----------------

_MONTHS = {
    m.lower(): i
    for i, m in enumerate(
        ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"], start=1
    )
}

def _parse_month_token(tok: str) -> datetime:
    """
    Accept 'Apr 2025' (any case/extra spaces).
    """
    tok = re.sub(r"\s+", " ", tok.strip())
    m = re.match(r"([A-Za-z]{3,})\s+(\d{4})", tok)
    if not m:
        raise ValueError(f"Bad month token: {tok}")
    mon = _MONTHS[m.group(1).lower()[:3]]
    yr = int(m.group(2))
    return datetime(yr, mon, 1)

def _parse_range(text: str):
    """
    Find 'Apr 2025 to Jun 2025' style ranges.
    """
    m = re.search(
        r"([A-Za-z]{3,}\s+\d{4})\s*(?:-|to)\s*([A-Za-z]{3,}\s+\d{4})",
        text,
        flags=re.IGNORECASE,
    )
    if not m:
        return None, None
    return _parse_month_token(m.group(1)), _parse_month_token(m.group(2))

def _parse_last_n_months(text: str) -> int | None:
    m = re.search(r"last\s+(\d+)\s+months?", text, flags=re.IGNORECASE)
    return int(m.group(1)) if m else None

def _extract(text: str, key: str) -> str | None:
    """
    Extract simple named entities like portfolio <X> or process <Y>.
    """
    m = re.search(rf"{key}\s+([A-Za-z0-9&/ \-]+?)(?:\s+(?:to|from|last|by|for)\b|$)", text, flags=re.IGNORECASE)
    if not m:
        return None
    return m.group(1).strip()

# ---------------- main router ----------------

def match_intent(user_text: str) -> Tuple[str, Dict, str]:
    """
    Returns (slug, params, title)
    params is a dict carried into question modules uniformly as `params`.
    """
    text = user_text.strip()

    # 1) fuzzy choose a slug from lexicon
    candidates = []
    for slug, phrases in LEXICON.items():
        score = max(process.extractOne(text, phrases, scorer=fuzz.WRatio))[1]
        candidates.append((slug, score))
    slug = max(candidates, key=lambda x: x[1])[0]

    # 2) Build params used by question modules
    params: Dict = {}

    # Common entities
    start_dt, end_dt = _parse_range(text)
    if start_dt and end_dt:
        params["start_month"] = start_dt.strftime("%b %Y")
        params["end_month"] = end_dt.strftime("%b %Y")

    n_last = _parse_last_n_months(text)
    if n_last:
        params["months"] = n_last  # modules can interpret as trailing window

    portfolio = _extract(text, "portfolio")
    if portfolio:
        params["portfolio"] = portfolio

    process_name = _extract(text, "process")
    if process_name:
        params["process"] = process_name

    location = _extract(text, "london|location|site")  # crude; keeps prior behaviour for examples
    if location and "portfolio" not in params:
        params["portfolio"] = location

    # 3) Title (keep helpful defaults)
    title_map = {s: t for s, t in _CHOICES}
    title = title_map.get(slug, "Halo Quality — Result")

    # If user specified an explicit range, reflect it in title
    if start_dt and end_dt:
        rng = f"{start_dt.strftime('%b %Y')} to {end_dt.strftime('%b %Y')}"
        title = re.sub(r"—.*$", "", title).strip()  # remove trailing default range
        title = f"{title} — {rng}"
    elif n_last:
        title = re.sub(r"—.*$", "", title).strip()
        title = f"{title} — last {n_last} months"

    return slug, params, title
