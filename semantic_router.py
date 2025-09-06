# semantic_router.py
from __future__ import annotations
from dataclasses import dataclass
import re

@dataclass
class IntentMatch:
    slug: str
    params: dict

def _month_token(s: str) -> str | None:
    m = re.search(r"(20\d{2}[-/](0?[1-9]|1[0-2]))", s)
    return m.group(1) if m else None

def match_query(q: str) -> IntentMatch | None:
    ql = q.strip().lower()

    # Complaints per 1000
    if "complaints per 1000" in ql or "complaints per thousand" in ql:
        portfolio = None
        m = re.search(r"portfolio\s+([a-z &]+)", ql)
        if m: portfolio = m.group(1).strip().title()
        # month range e.g., 'jun 2025 to aug 2025' OR '2025-06 to 2025-08'
        sm = _month_token(ql)
        em = None
        if " to " in ql:
            tail = ql.split(" to ", 1)[1]
            em = _month_token(tail) or em
        return IntentMatch(
            slug="complaints_per_thousand",
            params={"portfolio": portfolio, "start_month": sm, "end_month": em},
        )

    # Unique cases MoM
    if "unique cases" in ql and ("mom" in ql or "month" in ql):
        portfolio = None
        m = re.search(r"portfolio\s+([a-z &]+)", ql)
        if m: portfolio = m.group(1).strip().title()
        sm = _month_token(ql)
        em = None
        if " to " in ql:
            tail = ql.split(" to ", 1)[1]
            em = _month_token(tail) or em
        return IntentMatch(
            slug="unique_cases_mom",
            params={"portfolio": portfolio, "start_month": sm, "end_month": em},
        )

    # RCA1 by portfolio for a process (last N months)
    if "show rca1" in ql or ("rca1" in ql and "portfolio" in ql):
        months = 3
        m = re.search(r"last\s+(\d+)\s*month", ql)
        if m: months = int(m.group(1))
        portfolio = None
        m = re.search(r"portfolio\s+([a-z &]+)", ql)
        if m: portfolio = m.group(1).strip().title()
        return IntentMatch(
            slug="rca1_portfolio_process",
            params={"portfolio": portfolio, "months": months},
        )

    return None
