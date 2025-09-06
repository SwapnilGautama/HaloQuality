# semantic_router.py — tiny semantic matcher for Halo questions
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from rapidfuzz import process, fuzz


@dataclass
class IntentMatch:
    slug: str                 # questions.<slug>
    title: str                # display title
    params: Dict[str, object] # parameters passed to the question


# ------------------------
# Canonical slugs -> surface forms we match against
# ------------------------
LEXICON: Dict[str, List[str]] = {
    # Q1: complaints / cases
    "complaints_per_thousand": [
        "complaints per 1000",
        "complaints per thousand",
        "per 1000 complaints",
        "complaint rate per 1000",
        "complaints per 1k",
        "complaints per 1000 by process",
        "complaints per 1000 by portfolio",
    ],

    # Q2: rca1 by portfolio × process
    "rca1_portfolio_process": [
        "rca1 by portfolio",
        "show rca1",
        "root cause rca1",
        "show rca1 by process",
        "rca1 by portfolio by process",
    ],

    # Q3: unique cases MoM
    "unique_cases_mom": [
        "unique cases",
        "unique cases month on month",
        "unique cases by process",
        "unique cases by portfolio",
        "unique cases mom",
    ],
}


# ------------------------
# Light parsers
# ------------------------
_MONTH_RE = r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+(\d{4})"


def _parse_month(s: str) -> Optional[datetime]:
    m = re.search(_MONTH_RE, s, flags=re.IGNORECASE)
    if not m:
        return None
    try:
        return datetime.strptime(f"{m.group(1)} {m.group(2)}", "%b %Y")
    except Exception:
        return None


def _parse_month_range(q: str) -> Tuple[Optional[datetime], Optional[datetime]]:
    # "... Jun 2025 to Aug 2025"
    m = re.search(f"{_MONTH_RE}\\s+to\\s+{_MONTH_RE}", q, flags=re.IGNORECASE)
    if not m:
        return None, None
    try:
        s = datetime.strptime(f"{m.group(1)} {m.group(2)}", "%b %Y")
        e = datetime.strptime(f"{m.group(3)} {m.group(4)}", "%b %Y")
        return s, e
    except Exception:
        return None, None


def _parse_relative_months(q: str) -> Optional[int]:
    # "last 3 months" / "last three months"
    m = re.search(r"last\s+(\d+)\s+months?", q, flags=re.IGNORECASE)
    if m:
        try:
            n = int(m.group(1))
            return max(1, min(24, n))
        except Exception:
            pass
    words = {
        "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
        "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    }
    m2 = re.search(r"last\s+(one|two|three|four|five|six|seven|eight|nine|ten)\s+months?", q, flags=re.IGNORECASE)
    if m2:
        return words.get(m2.group(1).lower(), None)
    return None


def _parse_portfolio(q: str) -> Optional[str]:
    # naïve grab after keyword 'portfolio'
    m = re.search(r"portfolio\s+([A-Za-z0-9 &/-]+)", q, flags=re.IGNORECASE)
    if m:
        return m.group(1).strip().rstrip(".")
    return None


def _parse_process(q: str) -> Optional[str]:
    # For now, we only really need "Member Enquiry". Add more if needed.
    known = [
        "Member Enquiry",
        "Member Enquiries",
        "Member Enquiry ",
    ]
    for k in known:
        if re.search(re.escape(k), q, flags=re.IGNORECASE):
            return "Member Enquiry"
    # Fallback generic capture
    m = re.search(r"process\s+([A-Za-z0-9 &/-]+)", q, flags=re.IGNORECASE)
    if m:
        return m.group(1).strip().rstrip(".")
    return None


def _best_slug(q: str) -> Optional[str]:
    # Flatten LEXICON into (surface, slug)
    surfaces = []
    owners = []
    for slug, forms in LEXICON.items():
        for f in forms:
            surfaces.append(f)
            owners.append(slug)

    choice = process.extractOne(q, surfaces, scorer=fuzz.WRatio, score_cutoff=60)
    if not choice:
        return None
    idx = surfaces.index(choice[0])
    return owners[idx]


# ------------------------
# Public API
# ------------------------
def match_query(q: str) -> Optional[IntentMatch]:
    """
    Returns an IntentMatch(slug, title, params) or None if no intent found.
    - We standardize params to keep question.run(store, params, user_text) simple.
    """
    q = (q or "").strip()
    if not q:
        return None

    slug = _best_slug(q)
    if not slug:
        return None

    params: Dict[str, object] = {}

    # Common captures
    start, end = _parse_month_range(q)
    if start:
        params["start_month"] = start.strftime("%Y-%m")
    if end:
        params["end_month"] = end.strftime("%Y-%m")

    portfolio = _parse_portfolio(q)
    if portfolio:
        params["portfolio"] = portfolio

    # Intent-specific tweaks
    if slug == "complaints_per_thousand":
        title = "Complaints per 1,000 cases"
        # Accept a location like 'London' after 'portfolio' when users type
        # 'for portfolio London ...'
        loc_match = re.search(r"\bfor\s+portfolio\s+([A-Za-z0-9 /&-]+)", q, flags=re.IGNORECASE)
        if loc_match:
            params["portfolio"] = loc_match.group(1).strip()
        # If nothing specified, leave params as-is; the question will show a picker.

    elif slug == "rca1_portfolio_process":
        title = "RCA1 by Portfolio × Process — last 3 months"
        process_name = _parse_process(q)
        if process_name:
            params["process"] = process_name
        rel = _parse_relative_months(q)
        if rel:
            params["relative_months"] = rel
        # If explicit start/end months exist, the question can use them instead of relative.

    elif slug == "unique_cases_mom":
        title = "Unique cases (MoM)"
        # If user asked a range, we already captured start/end above.
        # Otherwise, leave empty and let the question surface controls.
    else:
        return None

    return IntentMatch(slug=slug, title=title, params=params)
