# semantic_router.py — tiny semantic matcher for Halo questions

from __future__ import annotations
from typing import Dict, List, Tuple, Any
import re
from rapidfuzz import process, fuzz

LEXICON: Dict[str, List[str]] = {
    "complaints_per_thousand": [
        "complaints per 1000", "complaints/1000", "complaint rate per 1000",
        "complaint rate", "complaints rate", "per 1000 complaints",
        "complaints per thousand"
    ],
    "complaint_volume_rate": [
        "complaint volume", "complaints volume", "overall complaints by month",
        "month on month complaints", "complaints mom"
    ],
    "unique_cases_mom": [
        "unique cases month on month", "cases mom", "unique cases trend", "unique case volume"
    ],
    "corr_nps": [
        "nps correlation", "complaints nps correlation", "correlation between complaints and nps",
        "nps vs complaints"
    ],
    "fpa_fail_rate": [
        "fpa fail rate", "first pass accuracy fail rate", "fpa failure rate"
    ],
    "fpa_fail_drivers": [
        "fpa fail drivers", "drivers of fpa fails", "root causes of fpa fails"
    ],
    "rca1_portfolio_process": [
        "rca1 by portfolio for process", "rca by portfolio process", "rca portfolio process"
    ],
    "mom_overview": [
        "overview", "summary", "show me an overview", "mom overview"
    ],
}

TITLES = {
    "complaints_per_thousand": "Complaints per 1,000 cases",
    "complaint_volume_rate": "Complaint volume (MoM)",
    "unique_cases_mom": "Unique cases (MoM)",
    "corr_nps": "Complaints vs NPS — correlation",
    "fpa_fail_rate": "FPA — Fail rate",
    "fpa_fail_drivers": "FPA — Fail drivers",
    "rca1_portfolio_process": "RCA1 by Portfolio × Process",
    "mom_overview": "MoM Overview",
}


def _extract_filters(text: str) -> Dict[str, Any]:
    args: Dict[str, Any] = {}
    m = re.search(r"\bportfolio\s+([A-Za-z]+(?:\s+[A-Za-z]+)?)", text, re.IGNORECASE)
    if m: args["portfolio"] = m.group(1).strip()

    m = re.search(r"\bprocess\s+([A-Za-z ]+?)(?:\s+for|\s+in|\s+last|\s+from|\s+to|$)", text, re.IGNORECASE)
    if m: args["process_name"] = m.group(1).strip()

    m = re.search(r"\b([A-Za-z]{3,9})\s+(\d{4})\s+to\s+([A-Za-z]{3,9})\s+(\d{4})", text, re.IGNORECASE)
    if m:
        args["start_month"] = f"{m.group(1)} {m.group(2)}"
        args["end_month"]   = f"{m.group(3)} {m.group(4)}"

    m = re.search(r"\blast\s+(\d+)\s+months?\b", text, re.IGNORECASE)
    if m: args["last_n_months"] = int(m.group(1))

    return args


def route_intent(user_text: str) -> Tuple[str, Dict[str, Any], str]:
    if not user_text or not user_text.strip():
        return "mom_overview", {}, TITLES.get("mom_overview", "Overview")

    candidates = []
    for slug, phrases in LEXICON.items():
        best = process.extractOne(user_text, phrases, scorer=fuzz.WRatio)
        if best:
            candidates.append((slug, best[1]))
    candidates.sort(key=lambda x: x[1], reverse=True)

    slug = candidates[0][0] if candidates else "mom_overview"
    title = TITLES.get(slug, slug.replace("_", " ").title())
    args = _extract_filters(user_text)
    return slug, args, title
