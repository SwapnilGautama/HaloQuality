# question_engine/nl_router.py
import re
import pandas as pd
from .resolvers import complaints_per_1000_by_process, rca1_by_portfolio_for_process

# month tokens like "Jun 2025", "Aug 25", "September 2024"
_MON = r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+(\d{2,4})"

def _to_period(mstr: str) -> pd.Period | None:
    try:
        ts = pd.to_datetime(mstr, errors="coerce")
        if pd.isna(ts):
            return None
        return ts.to_period("M")
    except Exception:
        return None

def _parse_month_range(q: str):
    # try "Jun 2025 to Aug 2025"
    m = re.search(rf"({_MON})\s+to\s+({_MON})", q, flags=re.IGNORECASE)
    if m:
        start_txt = m.group(1)
        end_txt   = m.group(3)
        p1 = _to_period(start_txt)
        p2 = _to_period(end_txt)
        return p1, p2
    # single month: "for Jun 2025"
    m2 = re.search(rf"for\s+({_MON})", q, flags=re.IGNORECASE)
    if m2:
        p = _to_period(m2.group(1))
        return p, p
    return None, None

def run_nl(q: str, store: dict) -> dict:
    qn = q.strip().lower()

    # ---------- Pattern 1: complaints per 1000 by process for portfolio X [month range] ----------
    m = re.search(
        r"complaints\s+per\s+1000\s+by\s+process\s+for\s+portfolio\s+([a-z0-9\s\-_/]+)",
        qn, flags=re.IGNORECASE
    )
    if m:
        portfolio = m.group(1).strip()
        p1, p2 = _parse_month_range(q)
        return complaints_per_1000_by_process(store, portfolio=portfolio, start_month=p1, end_month=p2)

    # ---------- Pattern 2: show rca1 by portfolio for process Y ----------
    m2 = re.search(
        r"show\s+rca1\s+by\s+portfolio\s+for\s+process\s+(.+)$",
        qn, flags=re.IGNORECASE
    )
    if m2:
        process_name = m2.group(1).strip()
        return rca1_by_portfolio_for_process(store, process_name=process_name)

    # Fallback
    return {
        "kind": "text",
        "text": (
            "I didn’t recognize that yet. Try:\n"
            "• complaints per 1000 by process for portfolio London Jun 2025 to Aug 2025\n"
            "• show rca1 by portfolio for process Member Enquiry"
        ),
    }
