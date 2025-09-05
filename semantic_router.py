# semantic_router.py — tiny semantic matcher for Halo questions
# -------------------------------------------------------------
# Returns (slug, args, title) where slug matches a module under questions/
# We use RapidFuzz on a curated lexicon so users can type freely.

from __future__ import annotations

import re
from typing import Dict, Any, Tuple, List
from rapidfuzz import process, fuzz


# Canonical question slugs -> list of surface forms we match against
LEXICON: Dict[str, List[str]] = {
    # complaints / cases
    "complaints_per_thousand": [
        "complaints per 1000",
        "complaints per thousand",
        "complaints/1000",
        "complaint rate per 1000",
        "complaint rate",
        "complaints rate",
        "per 1000 complaints",
    ],
    "complaint_volume_rate": [
        "complaint volume",
        "complaints volume",
        "complaints trend",
        "complaints by month",
        "volume of complaints",
        "complaint count month on month",
    ],
    "corr_nps": [
        "complaints nps correlation",
        "nps correlation complaints",
        "correlation nps",
        "nps vs complaints",
    ],
    "unique_cases_mom": [
        "unique cases month on month",
        "cases trend",
        "cases by month",
        "cases mom",
    ],
    "mom_overview": [
        "overview month on month",
        "mom overview",
        "kpi overview",
        "dashboard overview month",
    ],
    "rca1_portfolio_process": [
        "rca1 by portfolio",
        "reasons heatmap",
        "complaint reasons by portfolio",
        "rca analysis portfolio process",
    ],
    # FPA
    "fpa_fail_drivers": [
        "fpa fail drivers",
        "drivers of case fails",
        "first pass accuracy fail drivers",
        "fpa drivers",
        "why are cases failing",
    ],
    # legacy/alias – you can still ask for 'fpa fail rate', we route to drivers module
    "fpa_fail_rate": [
        "fpa fail rate",
        "fpa failure rate",
        "case fail rate",
        "first pass accuracy rate",
    ],
}

# Small set of allowed slugs for quick validation
ALLOWED_SLUGS = set(LEXICON.keys())


def _best_slug(query: str) -> str:
    # Build a flat choices list “<slug>::<surface>”
    choices = []
    for slug, phrases in LEXICON.items():
        for p in phrases:
            choices.append(f"{slug}::{p}")

    best = process.extractOne(
        query,
        choices,
        scorer=fuzz.token_set_ratio,
        score_cutoff=55  # forgiving
    )
    if not best:
        # default to a sensible general metric
        return "complaints_per_thousand"

    slug = best[0].split("::", 1)[0]
    return slug


# light-month parsing (optional hooks your question modules may use)
MONTH_RX = re.compile(r"\b(20\d{2}[-/ ]?(?:0?[1-9]|1[0-2])|jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\b", re.I)


def _extract_args(query: str) -> Dict[str, Any]:
    """
    Keep this intentionally light. We just identify portfolio / process hints
    and pass them along. Each question module can decide whether to use them.
    """
    args: Dict[str, Any] = {"query": query}

    # portfolio hint
    m = re.search(r"\bportfolio\s+([A-Za-z][\w\- ]+)", query, flags=re.I)
    if m:
        args["portfolio"] = m.group(1).strip()

    # process hint
    m2 = re.search(r"\bprocess\s+([A-Za-z][\w\- ]+)", query, flags=re.I)
    if m2:
        args["process"] = m2.group(1).strip()

    # month hints (we just surface raw tokens to the module)
    months = MONTH_RX.findall(query)
    if months:
        args["months"] = months

    return args


def _title_for(slug: str, query: str) -> str:
    titles = {
        "complaints_per_thousand": "Complaints per 1,000 cases",
        "complaint_volume_rate": "Complaints volume (MoM)",
        "corr_nps": "Complaints ↔ NPS correlation",
        "unique_cases_mom": "Unique cases (MoM)",
        "mom_overview": "Month-on-Month Overview",
        "rca1_portfolio_process": "RCA-1 by Portfolio × Process",
        "fpa_fail_drivers": "FPA — Fail drivers",
        "fpa_fail_rate": "FPA — Fail rate",  # legacy (will be aliased)
    }
    return titles.get(slug, query)


def route_intent(query: str) -> Tuple[str, Dict[str, Any], str]:
    """
    Main entry. Returns (slug, args, title).
    We also normalize legacy slugs to their canonical implementations.
    """
    slug = _best_slug(query)

    # legacy normalization: treat fpa_fail_rate as drivers question
    if slug == "fpa_fail_rate":
        slug = "fpa_fail_drivers"

    args = _extract_args(query)
    title = _title_for(slug, query)
    return slug, args, title
