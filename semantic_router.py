# semantic_router.py
from __future__ import annotations
import re
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List
import streamlit as st

# --- fuzzy with safe fallback (so app boots even if rapidfuzz isn't ready) ---
try:
    from rapidfuzz import process as rf_process
    _HAVE_RF = True
except Exception:  # pragma: no cover
    from difflib import SequenceMatcher
    _HAVE_RF = False
    class _Shim:
        @staticmethod
        def extractOne(q: str, choices: List[str]):
            best = None; score = -1.0
            for c in choices:
                s = SequenceMatcher(None, q.lower(), c.lower()).ratio()
                if s > score:
                    score, best = s, c
            return (best, int(score*100), None)
    rf_process = _Shim()  # type: ignore

# ----------------------------- month parsing ----------------------------------
MONTH_ABBR = ["jan","feb","mar","apr","may","jun","jul","aug","sep","oct","nov","dec"]
MONTH_RX = re.compile(
    r"(?P<m1>(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*)\s*['’]?(?P<y1>\d{2,4})"
    r"(?:\s*(?:to|-|–|—)\s*(?P<m2>(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*)\s*['’]?(?P<y2>\d{2,4}))?",
    re.I,
)
LAST_N_RX = re.compile(r"last\s*(?P<n>\d{1,2})\s*months?", re.I)

def _mm_yyyy(abbr: str, y: str) -> str:
    abbr = abbr.lower()[:3]; mm = MONTH_ABBR.index(abbr)+1
    y = y.strip(); y = f"20{y}" if len(y)==2 else y
    return f"{int(y):04d}-{mm:02d}"

def parse_months(text: str) -> Tuple[Optional[Tuple[str,str]], Optional[int]]:
    """Return (explicit_range, last_n) where explicit_range == (YYYY-MM, YYYY-MM) or None."""
    m = MONTH_RX.search(text)
    if m:
        y1 = m.group("y1"); y2 = m.group("y2") or y1
        return ((_mm_yyyy(m.group("m1"), y1), _mm_yyyy(m.group("m2") or m.group("m1"), y2)), None)
    m2 = LAST_N_RX.search(text)
    if m2:
        return (None, int(m2.group("n")))
    return (None, None)

# --------------------------- router data model --------------------------------
@dataclass
class Parsed:
    qid: str
    portfolio: Optional[str]
    process: Optional[str]
    month_range: Optional[Tuple[str,str]]  # inclusive YYYY-MM range
    last_n: Optional[int]
    by_dim: Optional[str]  # team|manager|location|scheme|portfolio|process

# Registry entries
@dataclass
class QuestionSpec:
    qid: str
    title: str
    handler: str               # import path of function "run(store, params)"
    examples: List[str]

# --------------------------- question catalogue -------------------------------
CATALOG: List[QuestionSpec] = [
    QuestionSpec(
        qid="q1",
        title="Complaints per 1,000 cases (MoM) — optional process/portfolio",
        handler="questions.question_q1.run",
        examples=[
            "complaints per 1000 by process for portfolio London Jun 2025 to Aug 2025",
            "complaints/1000 last 3 months",
        ],
    ),
    QuestionSpec(
        qid="q2",
        title="RCA1 mix by portfolio for a process",
        handler="questions.question_q2.run",
        examples=[
            'show rca1 by portfolio for process "Member Enquiry" last 3 months',
            "reasons by portfolio for process member enquiry",
        ],
    ),
    QuestionSpec(
        qid="q3",
        title="Unique cases (MoM) — optional process/portfolio",
        handler="questions.question_q3.run",
        examples=[
            "unique cases by process and portfolio Apr 2025 to Jun 2025",
            "case volume last 6 months",
        ],
    ),
    QuestionSpec(
        qid="q4",
        title="Complaints volume & complaints/1000 — by portfolio or process",
        handler="questions.question_q4.run",
        examples=[
            "complaints by portfolio last 3 months",
            "complaints per 1000 by process last 3 months",
        ],
    ),
    QuestionSpec(
        qid="q5",
        title="FPA fail rate by dimension (team/manager/location/process/portfolio)",
        handler="questions.question_q5.run",
        examples=[
            "fpa fail rate by team last 3 months",
            "first pass accuracy fails by manager aug 2025",
        ],
    ),
    QuestionSpec(
        qid="q6",
        title="Top FPA fail drivers (labels or comment tokens)",
        handler="questions.question_q6.run",
        examples=[
            "biggest drivers of case fails",
            "fpa fail reasons last 3 months",
        ],
    ),
]

# ------------------------------- helpers --------------------------------------
def _choices_from_store(store) -> Tuple[List[str], List[str]]:
    portfolios = []
    processes = []
    try:
        if "cases" in store:
            df = store["cases"]
            if "Portfolio_std" in df.columns:
                portfolios += df["Portfolio_std"].dropna().unique().tolist()
            if "ProcessName" in df.columns:
                processes += df["ProcessName"].dropna().unique().tolist()
        if "complaints" in store:
            df = store["complaints"]
            if "Portfolio_std" in df.columns:
                portfolios += df["Portfolio_std"].dropna().unique().tolist()
            if "Parent_Case_Type" in df.columns:
                processes += df["Parent_Case_Type"].dropna().unique().tolist()
    except Exception:
        pass
    return sorted(set(portfolios)), sorted(set(processes))

def _fuzzy_pick(q: str, choices: List[str], threshold=70) -> Optional[str]:
    if not choices:
        return None
    match, score, _ = rf_process.extractOne(q, choices)
    return match if score >= threshold else None

def _pick_qid(prompt: str) -> str:
    phrases = []
    labels  = []
    for spec in CATALOG:
        for ex in spec.examples:
            phrases.append(ex)
            labels.append(spec.qid)
    best = rf_process.extractOne(prompt, phrases)
    if best:
        return labels[phrases.index(best[0])]
    # fallback: heuristics
    p = prompt.lower()
    if "complaints per" in p or "complaints/1000" in p:
        return "q1"
    if "rca1" in p or "reason" in p:
        return "q2"
    if "unique cases" in p or "case volume" in p:
        return "q3"
    if "fpa" in p and "driver" in p:
        return "q6"
    if "fpa" in p and ("rate" in p or "fail" in p):
        return "q5"
    return "q1"

def parse_prompt(prompt: str, store) -> Parsed:
    p = prompt.strip()
    qid = _pick_qid(p)
    month_range, last_n = parse_months(p)
    portfolios, processes = _choices_from_store(store)
    portfolio = _fuzzy_pick(p, portfolios)
    process   = _fuzzy_pick(p, processes)

    # by-dimension (for q5 etc.)
    by_dim = None
    low = p.lower()
    if " by team " in f" {low} ":
        by_dim = "team"
    elif " by manager " in f" {low} ":
        by_dim = "manager"
    elif " by location " in f" {low} ":
        by_dim = "location"
    elif " by process " in f" {low} ":
        by_dim = "process"
    elif " by portfolio " in f" {low} ":
        by_dim = "portfolio"
    elif " by scheme " in f" {low} ":
        by_dim = "scheme"

    return Parsed(
        qid=qid,
        portfolio=portfolio,
        process=process,
        month_range=month_range,
        last_n=last_n,
        by_dim=by_dim
    )

# ------------------------------- routing --------------------------------------
def _import_handler(path: str):
    mod_path, func_name = path.rsplit(".", 1)
    mod = __import__(mod_path, fromlist=[func_name])
    return getattr(mod, func_name)

def route(prompt: str, store: Dict):
    parsed = parse_prompt(prompt, store)
    spec = next(s for s in CATALOG if s.qid == parsed.qid)
    handler = _import_handler(spec.handler)
    # The handler renders directly with Streamlit:
    handler(store, parsed)
