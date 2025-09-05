# question_engine/parser.py
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional, Tuple, List

# ---- fuzzy (fast) with graceful fallback ------------------------------------
try:
    from rapidfuzz import process as rf_process, fuzz as rf_fuzz  # type: ignore
    _HAVE_RF = True
except Exception:  # pragma: no cover
    # Fallback keeps the app booting even if rapidfuzz isn't ready yet.
    from difflib import SequenceMatcher

    _HAVE_RF = False

    class _Shim:  # tiny shim so call sites look the same
        @staticmethod
        def extractOne(q: str, choices: List[str]):
            best = None
            best_score = -1.0
            for c in choices:
                s = SequenceMatcher(None, q.lower(), c.lower()).ratio()
                if s > best_score:
                    best_score, best = s, c
            return (best, int(best_score * 100), None)

    rf_process = _Shim()  # type: ignore

# -----------------------------------------------------------------------------

INTENTS = {
    "complaints_per_1000": [
        "complaints per 1000",
        "complaints/1000",
        "complaints per thousand",
        "cpt",
    ],
    "rca1_by_portfolio": [
        "rca1 by portfolio",
        "reasons by portfolio",
        "complaint reasons by portfolio",
    ],
    "unique_cases": [
        "unique cases",
        "case volume",
        "cases by",
    ],
    "fpa_fail_drivers": [
        "drivers of case fails",
        "fpa failures",
        "first pass accuracy fails",
        "fpa fail reasons",
    ],
}

# map common field synonyms -> canonical columns in your data
FIELD_SYNONYMS = {
    "portfolio": ["portfolio", "portfolio_std", "location"],  # keep "portfolio_std" canonical
    "process": ["process", "process name", "parent case type", "case type", "member enquiry"],
}

MONTH_RX = re.compile(
    r"(?P<m1>(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*)\s*['’]?(?P<y1>\d{2,4})"
    r"(?:\s*(?:to|-|–|—)\s*(?P<m2>(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*)\s*['’]?(?P<y2>\d{2,4}))?",
    flags=re.I,
)

LAST_N_RX = re.compile(r"last\s*(?P<n>\d{1,2})\s*months?", re.I)

MONTH_ABBR = ["jan","feb","mar","apr","may","jun","jul","aug","sep","oct","nov","dec"]

def _norm_month_name(s: str) -> str:
    s = s.strip().lower()[:3]
    return s

def _yyyymm_from_match(m_abbr: str, y: str) -> str:
    m_idx = MONTH_ABBR.index(_norm_month_name(m_abbr)) + 1
    y = y.strip()
    y = f"20{y}" if len(y) == 2 else y
    return f"{int(y):04d}-{m_idx:02d}"

@dataclass
class ParseResult:
    intent: str
    portfolio: Optional[str] = None
    process: Optional[str] = None
    # months is either (start_yyyy_mm, end_yyyy_mm) inclusive, or None
    months: Optional[Tuple[str, str]] = None

def _pick_intent(q: str) -> str:
    # score the question against INTENTS vocabulary
    phrases = []
    labels = []
    for label, words in INTENTS.items():
        for w in words:
            phrases.append(w)
            labels.append(label)

    best = rf_process.extractOne(q, phrases)
    best_phrase = best[0] if best else None
    return labels[phrases.index(best_phrase)] if best_phrase else "complaints_per_1000"

def _extract_months(q: str) -> Optional[Tuple[str, str]]:
    m = MONTH_RX.search(q)
    if m:
        y1 = m.group("y1")
        y2 = m.group("y2") or y1
        mm1 = _yyyymm_from_match(m.group("m1"), y1)
        mm2 = _yyyymm_from_match(m.group("m2") or m.group("m1"), y2)
        return (mm1, mm2)

    m2 = LAST_N_RX.search(q)
    if m2:
        # "last N months" -> None here; router will interpret as rolling window
        return None
    return None

def _extract_named(q: str, choices: List[str]) -> Optional[str]:
    if not choices:
        return None
    hit = rf_process.extractOne(q, choices)
    if not hit:
        return None
    match, score, _ = hit
    # mild threshold so users can be sloppy
    return match if (score >= 70) else None

def parse(
    q: str,
    portfolios: List[str],
    processes: List[str],
) -> ParseResult:
    q = q.strip()
    intent = _pick_intent(q)

    months = _extract_months(q)

    portfolio = _extract_named(q, portfolios) if portfolios else None
    process = _extract_named(q, processes) if processes else None

    # Special-case: if "member enquiry" substring appears, prefer that as process
    if not process and "member" in q.lower() and "enquir" in q.lower():
        process = "Member Enquiry"

    return ParseResult(intent=intent, portfolio=portfolio, process=process, months=months)
