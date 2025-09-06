# semantic_router.py
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any, Optional

import pandas as pd
from rapidfuzz import process, fuzz

# ----- intents & surfaces -----------------------------------------------------

LEXICON: Dict[str, List[str]] = {
    # complaints per 1,000 cases
    "complaints_per_thousand": [
        "complaints per 1000",
        "complaints per thousand",
        "complaints/1000",
        "complaint rate per 1000",
        "per 1000 complaints",
        "complaint rate",
        "complaint index",
    ],
    # rca1 by portfolio x process
    "rca1_portfolio_process": [
        "show rca1 by portfolio for process",
        "rca1 by portfolio by process",
        "rca by portfolio process",
        "drivers by portfolio and process",
        "top drivers by portfolio and process",
    ],
    # unique cases by month
    "unique_cases_mom": [
        "unique cases by process and portfolio",
        "unique cases month on month",
        "unique cases mom",
        "cases unique per month",
        "distinct cases per month",
    ],
}

TITLES = {
    "complaints_per_thousand": "Complaints per 1,000 cases",
    "rca1_portfolio_process": "RCA1 by Portfolio × Process — last 3 months",
    "unique_cases_mom": "Unique cases (MoM)",
}

IntentMatch(
    slug="complaints_dashboard",
    description="Portfolio-level complaints per 1000 with a monthly trend and a reasons deep-dive for the selected month.",
    patterns=[
        r"complaints analysis",
        r"complaints dashboard",
        r"show monthly complaints and reasons",
        r"complaints 1000 by portfolio .* reasons",
    ],
    module="questions.complaints_dashboard",
)


# ----- small data structure ---------------------------------------------------

@dataclass
class IntentMatch:
    slug: str
    title: str
    args: Dict[str, Any]

# ----- argument parsing helpers ----------------------------------------------

_MONTH_RX = r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+20\d{2}"
RANGE_RX = re.compile(
    rf"(?P<s>{_MONTH_RX})\s*(to|-|→)\s*(?P<e>{_MONTH_RX})",
    re.IGNORECASE,
)

def _find_months(text: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Return (start_month, end_month) as 'YYYY-MM' strings if found, else (None, None).
    Supports 'Apr 2025 to Jun 2025' or a single month 'Aug 2025'.
    """
    m = RANGE_RX.search(text)
    if m:
        s = pd.to_datetime(m.group("s")).strftime("%Y-%m")
        e = pd.to_datetime(m.group("e")).strftime("%Y-%m")
        return s, e

    single = re.search(_MONTH_RX, text, re.IGNORECASE)
    if single:
        s = pd.to_datetime(single.group(0)).strftime("%Y-%m")
        return s, s

    return None, None

def _find_relative_months(text: str) -> Optional[int]:
    m = re.search(r"last\s+(\d{1,2})\s+months?", text, re.IGNORECASE)
    if m:
        return int(m.group(1))
    return None

def _find_portfolio(text: str) -> Optional[str]:
    # examples: "for portfolio London", "portfolio: Member Services"
    m = re.search(r"(?:for\s+)?portfolio[:\s]+([A-Za-z0-9 &/_-]{2,})", text, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return None

def _find_process(text: str) -> Optional[str]:
    # examples: "for process Member Enquiry", "process: Billing"
    m = re.search(r"(?:for\s+)?process[:\s]+([A-Za-z0-9 &/_-]{2,})", text, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return None

def _parse_args(slug: str, text: str) -> Dict[str, Any]:
    """
    Extract a clean argument dict for the question `slug`.
    Only keys that questions commonly support are returned.
    """
    args: Dict[str, Any] = {}

    # time windows
    start, end = _find_months(text)
    rel = _find_relative_months(text)
    if start and end:
        args["start_month"] = start
        args["end_month"] = end
    elif rel is not None:
        args["relative_months"] = rel
    else:
        # good default for RCA and unique-cases is last 3 months
        if slug in {"rca1_portfolio_process"}:
            args["relative_months"] = 3

    # entities
    port = _find_portfolio(text)
    proc = _find_process(text)
    if port:
        args["portfolio"] = port
    if proc:
        args["process"] = proc

    return args

# ----- main matcher -----------------------------------------------------------

def match_query(text: str, *, cutoff: int = 70) -> Optional[IntentMatch]:
    """
    Return best (slug, args, title) for a query or None if nothing is good enough.
    """
    text_norm = " ".join(text.lower().split())

    # score every surface form of every intent
    candidates: List[Tuple[str, str, int]] = []
    for slug, surfaces in LEXICON.items():
        # RapidFuzz top score among surfaces for this intent
        choice = process.extractOne(text_norm, surfaces, scorer=fuzz.token_set_ratio)
        if choice:
            score = choice[1]
            candidates.append((slug, choice[0], score))

    if not candidates:
        return None

    # pick highest-scoring intent if it clears the cutoff
    slug, _, score = max(candidates, key=lambda x: x[2])
    if score < cutoff:
        return None

    args = _parse_args(slug, text_norm)
    title = TITLES.get(slug, slug.replace("_", " ").title())

    return IntentMatch(slug=slug, title=title, args=args)
