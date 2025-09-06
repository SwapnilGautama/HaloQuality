import re
from dataclasses import dataclass
from typing import Optional, Dict

# Months map for parsing "Apr 2025", "June 2025", etc.
_MMAP = {
    "jan": "01", "january": "01",
    "feb": "02", "february": "02",
    "mar": "03", "march": "03",
    "apr": "04", "april": "04",
    "may": "05",
    "jun": "06", "june": "06",
    "jul": "07", "july": "07",
    "aug": "08", "august": "08",
    "sep": "09", "sept": "09", "september": "09",
    "oct": "10", "october": "10",
    "nov": "11", "november": "11",
    "dec": "12", "december": "12",
}


@dataclass
class IntentMatch:
    slug: str
    params: Dict
    title: Optional[str] = None


# ---- helpers to parse ranges -------------------------------------------------

# e.g. "Apr 2025 to Jun 2025" or "Apr 25 to Jun 25"
RANGE_RE = re.compile(
    r"\b(?P<m1>[A-Za-z]{3,9})\.?\s*(?P<y1>\d{2,4})\s*(?:to|-|–|—)\s*(?P<m2>[A-Za-z]{3,9})\.?\s*(?P<y2>\d{2,4})\b",
    re.IGNORECASE,
)
# e.g. "last 3 months"
LAST_N_RE = re.compile(r"\blast\s*(?P<n>\d{1,2})\s*months?\b", re.IGNORECASE)
# e.g. "portfolio London", "for portfolio London"
PORT_RE = re.compile(r"(?:for\s+)?portfolio\s+(?P<p>[\w\s&'\-]+)", re.IGNORECASE)
# e.g. "process Member Enquiry", "for process Member Enquiry"
PROC_RE = re.compile(r"(?:for\s+)?process\s+(?P<p>[\w\s&'\-]+)", re.IGNORECASE)


def _ym(m: str, y: str) -> Optional[str]:
    m = (m or "").strip().lower()
    y = (y or "").strip()
    if not m or not y:
        return None
    if len(y) == 2:
        # assume 20xx for '25'
        y = "20" + y
    mm = _MMAP.get(m[:3])
    if not mm:
        return None
    return f"{y}-{mm}"


def _parse_time(text: str) -> Dict:
    """
    Returns either {'relative_months': N} or {'start_ym': 'YYYY-MM', 'end_ym': 'YYYY-MM'} or {}.
    """
    text = text or ""
    m = LAST_N_RE.search(text)
    if m:
        try:
            n = int(m.group("n"))
            return {"relative_months": n}
        except Exception:
            pass

    m = RANGE_RE.search(text)
    if m:
        s = _ym(m.group("m1"), m.group("y1"))
        e = _ym(m.group("m2"), m.group("y2"))
        if s and e:
            return {"start_ym": s, "end_ym": e}

    return {}


def _parse_dims(text: str) -> Dict:
    out = {}
    m = PORT_RE.search(text or "")
    if m:
        out["portfolio_text"] = m.group("p").strip()
    m = PROC_RE.search(text or "")
    if m:
        out["process_text"] = m.group("p").strip()
    return out


# ---- Main matcher ------------------------------------------------------------

def match_query(text: str) -> Optional[IntentMatch]:
    """
    Very lightweight router: recognizes 3 intents we have modules for.
    Returns IntentMatch(slug, params, title) or None if not understood.
    """
    if not text:
        return None

    t = text.lower().strip()

    # Complaints per thousand
    if re.search(r"\bcomplaints?\s+per\s+(1,?000|thousand)\b", t):
        params = {}
        params.update(_parse_time(t))
        params.update(_parse_dims(t))
        title = "Complaints per 1,000 cases"
        # If no explicit time found, default to a sensible 3-month window
        if not any(k in params for k in ("relative_months", "start_ym", "end_ym")):
            params["relative_months"] = 3
        return IntentMatch(slug="complaints_per_thousand", params=params, title=title)

    # RCA1 portfolio × process
    if re.search(r"\brca1\b|\broot\s*cause\b", t):
        params = {}
        params.update(_parse_time(t))
        params.update(_parse_dims(t))
        title = "RCA1 by Portfolio × Process"
        if not any(k in params for k in ("relative_months", "start_ym", "end_ym")):
            params["relative_months"] = 3
        return IntentMatch(slug="rca1_portfolio_process", params=params, title=title)

    # Unique cases (MoM)
    if re.search(r"\bunique\s+cases?\b", t):
        params = {}
        params.update(_parse_time(t))
        params.update(_parse_dims(t))
        title = "Unique cases (MoM)"
        # If no time, assume last 3 months for a compact MoM
        if not any(k in params for k in ("relative_months", "start_ym", "end_ym")):
            params["relative_months"] = 3
        return IntentMatch(slug="unique_cases_mom", params=params, title=title)

    # Friendly aliases → default complaints view
    if re.search(r"\bcomplaint(s)?\s+(dashboard|analysis|overview)\b", t):
        params = {}
        params.update(_parse_time(t))
        params.update(_parse_dims(t))
        if not any(k in params for k in ("relative_months", "start_ym", "end_ym")):
            params["relative_months"] = 3
        return IntentMatch(slug="complaints_per_thousand", params=params, title="Complaints dashboard")

    return None
