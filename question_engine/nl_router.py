# question_engine/nl_router.py
from __future__ import annotations
import re
from typing import Optional, Dict, Any, Tuple

from .blocks import complaints_per_1000_by_process, rca1_by_portfolio_for_process

# month helpers ---------------------------------------------------------

MONTHS = {
    "jan":"01","january":"01",
    "feb":"02","february":"02",
    "mar":"03","march":"03",
    "apr":"04","april":"04",
    "may":"05",
    "jun":"06","june":"06",
    "jul":"07","july":"07",
    "aug":"08","august":"08",
    "sep":"09","sept":"09","september":"09",
    "oct":"10","october":"10",
    "nov":"11","november":"11",
    "dec":"12","december":"12",
}

MONTH_TOKEN = r"(jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec|january|february|march|april|june|july|august|september|october|november|december)"
MONTH_RE = re.compile(rf"{MONTH_TOKEN}[ '\-_/]*([0-9]{{2,4}})", re.I)

def _norm_year(y: str) -> str:
    y = y.strip()
    if len(y) == 2:
        # assume 20xx
        return f"20{y}"
    return y

def _to_ym(match) -> str:
    mon_raw = match.group(1).lower()
    yr = _norm_year(match.group(2))
    mm = MONTHS[mon_raw]
    return f"{yr}-{mm}"

def parse_months_range(text: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Returns (m_from, m_to) in 'YYYY-MM'. Supports:
    - 'jun 2025 to aug 2025'
    - 'from jun 2025 to aug 2025'
    - single month: returns (month, month)
    """
    t = text.lower()
    # find all month tokens
    ms = list(MONTH_RE.finditer(t))
    if not ms:
        return None, None
    if len(ms) == 1:
        m = _to_ym(ms[0])
        return m, m
    # if 2+ months, use first and last
    return _to_ym(ms[0]), _to_ym(ms[-1])

# main routing ----------------------------------------------------------

def run_nl(prompt: str, store) -> Dict[str, Any]:
    """
    Parses the free-text prompt and routes to a handler.
    Returns a dict: {title, df, fig} or {error: "..."}.
    """
    txt = (prompt or "").strip()
    if not txt:
        return {"error": "Empty question."}

    low = txt.lower()

    # parse months (optional)
    m_from, m_to = parse_months_range(low)

    # 1) complaints per 1000 by process ... (optionally for Portfolio X)
    if "complaints per 1000" in low and "by process" in low:
        # extract optional 'for portfolio <name>'
        port = None
        m = re.search(r"for\s+portfolio\s+([a-z0-9 &/\-']+)", low)
        if m:
            port = m.group(1).strip()
            # trim trailing month phrase if captured
            port = re.sub(MONTH_TOKEN + r".*$", "", port, flags=re.I).strip(" ,.-")

        return complaints_per_1000_by_process(store, portfolio=port, m_from=m_from, m_to=m_to)

    # 2) show rca1 by portfolio for process <name>
    if ("rca1" in low or "rca 1" in low) and "by portfolio" in low and "process" in low:
        # find process name after 'process'
        pm = re.search(r"process\s+([a-z0-9 &'/_\-]+)", low)
        proc = pm.group(1).strip() if pm else ""
        proc = re.sub(MONTH_TOKEN + r".*$", "", proc, flags=re.I).strip(" ,.-")
        if not proc:
            return {"error": "Please specify the process name (e.g., 'process Member Enquiry')."}
        return rca1_by_portfolio_for_process(store, process=proc, m_from=m_from, m_to=m_to)

    # fallback
    return {
        "error": (
            "Sorry, I couldn't understand that. Try something like:\n"
            "• complaints per 1000 by process for portfolio London Jun 2025 to Aug 2025\n"
            "• show rca1 by portfolio for process Member Enquiry last 3 months"
        )
    }
