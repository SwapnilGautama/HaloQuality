# semantic_router.py
from dataclasses import dataclass
import re
import pandas as pd

@dataclass
class IntentMatch:
    slug: str
    params: dict

def _parse_month(s: str):
    # supports "Jun 2025", "Aug 25"
    try:
        return pd.to_datetime(s, errors="raise").to_period("M").to_timestamp()
    except Exception:
        # try adding "01" for dayfirst safety
        try:
            return pd.to_datetime(f"01 {s}", dayfirst=True).to_period("M").to_timestamp()
        except Exception:
            return None

def match_query(q: str) -> IntentMatch:
    ql = (q or "").strip().lower()

    # button-friendly fallbacks
    if ql.startswith("complaints per 1000"):
        # try to pull portfolio and month range
        # e.g., "complaints per 1000 by process for portfolio london jun 2025 to aug 2025"
        m = re.search(r"portfolio\s+([a-z\s\-]+)\s+([a-z]{3}\s?\d{2,4})\s+to\s+([a-z]{3}\s?\d{2,4})", ql)
        params = {}
        if m:
            params["portfolio"] = m.group(1).strip()
            params["start_month"] = _parse_month(m.group(2))
            params["end_month"]   = _parse_month(m.group(3))
            if params["start_month"] is not None:
                params["start_month"] = str(params["start_month"].date())
            if params["end_month"] is not None:
                params["end_month"] = str(params["end_month"].date())
        return IntentMatch(slug="complaints_per_thousand", params=params)

    if ql.startswith("show rca1"):
        # e.g., show rca1 by portfolio for process member enquiry last 3 months
        m = re.search(r"process\s+([a-z\s\-]+)", ql)
        params = {}
        if m:
            params["process"] = m.group(1).strip()
        return IntentMatch(slug="rca1_portfolio_process", params=params)

    if ql.startswith("unique cases"):
        # e.g., unique cases by process and portfolio apr 2025 to jun 2025
        m = re.search(r"([a-z]{3}\s?\d{2,4})\s+to\s+([a-z]{3}\s?\d{2,4})", ql)
        params = {}
        if m:
            params["start_month"] = _parse_month(m.group(1))
            params["end_month"]   = _parse_month(m.group(2))
            if params["start_month"] is not None:
                params["start_month"] = str(params["start_month"].date())
            if params["end_month"] is not None:
                params["end_month"] = str(params["end_month"].date())
        return IntentMatch(slug="unique_cases_mom", params=params)

    if "dashboard" in ql and "complaint" in ql:
        return IntentMatch(slug="complaints_per_thousand", params={})

    # default: try complaints per 1000
    return IntentMatch(slug="complaints_per_thousand", params={})
