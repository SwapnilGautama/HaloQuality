# semantic_router.py
from __future__ import annotations
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict
import streamlit as st

# --- fuzzy (with safe fallback) ---
try:
    from rapidfuzz import process as rf_process
    _RF = True
except Exception:  # fallback to difflib
    from difflib import SequenceMatcher
    _RF = False
    class _Shim:
        @staticmethod
        def extractOne(q: str, choices: List[str]):
            best, score = None, -1.0
            for c in choices:
                s = SequenceMatcher(None, q.lower(), c.lower()).ratio()
                if s > score: best, score = c, s
            return (best, int(score*100), None)
    rf_process = _Shim()  # type: ignore

# ---------------- date parsing ----------------
MONTH_ABBR = ["jan","feb","mar","apr","may","jun","jul","aug","sep","oct","nov","dec"]
MONTH_RX = re.compile(
    r"(?P<m1>(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*)\s*['’]?(?P<y1>\d{2,4})"
    r"(?:\s*(?:to|-|–|—)\s*(?P<m2>(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*)\s*['’]?(?P<y2>\d{2,4}))?",
    re.I,
)
LAST_N_RX = re.compile(r"last\s*(?P<n>\d{1,2})\s*months?", re.I)

def _mm_yyyy(abbr: str, year: str) -> str:
    abbr = abbr.lower()[:3]
    mm   = MONTH_ABBR.index(abbr)+1
    y    = f"20{year}" if len(year)==2 else year
    return f"{int(y):04d}-{mm:02d}"

def parse_months(text: str) -> Tuple[Optional[Tuple[str,str]], Optional[int]]:
    m = MONTH_RX.search(text)
    if m:
        y1, y2 = m.group("y1"), (m.group("y2") or m.group("y1"))
        return ((_mm_yyyy(m.group("m1"), y1), _mm_yyyy(m.group("m2") or m.group("m1"), y2)), None)
    m2 = LAST_N_RX.search(text)
    if m2: return (None, int(m2.group("n")))
    return (None, None)

# --------------- model ----------------
@dataclass
class Parsed:
    qid: str
    portfolio: Optional[str]
    process: Optional[str]
    month_range: Optional[Tuple[str,str]]
    last_n: Optional[int]
    by_dim: Optional[str]

@dataclass
class QuestionSpec:
    qid: str
    title: str
    handler: str               # module path to function run(store, params)
    examples: List[str]

# ------------- registry: YOUR filenames -------------
CATALOG: List[QuestionSpec] = [
    QuestionSpec(
        qid="complaints_per_thousand",
        title="Complaints per 1,000 cases (MoM)",
        handler="questions.complaints_per_thousand.run",
        examples=[
            "complaints per 1000 by process for portfolio London Jun 2025 to Aug 2025",
            "complaints/1000 last 3 months",
            "complaints per thousand by portfolio",
        ],
    ),
    QuestionSpec(
        qid="complaint_volume_rate",
        title="Complaints volume & rate by dimension",
        handler="questions.complaint_volume_rate.run",
        examples=[
            "complaints by portfolio last 3 months",
            "complaints rate by process",
        ],
    ),
    QuestionSpec(
        qid="unique_cases_mom",
        title="Unique cases month on month",
        handler="questions.unique_cases_mom.run",
        examples=[
            "unique cases by process and portfolio Apr 2025 to Jun 2025",
            "case volume last 6 months",
        ],
    ),
    QuestionSpec(
        qid="mom_overview",
        title="MoM overview (combined KPIs)",
        handler="questions.mom_overview.run",
        examples=[
            "month on month overview",
            "kpi trend last 6 months",
        ],
    ),
    QuestionSpec(
        qid="fpa_fail_rate",
        title="FPA fail rate by dimension",
        handler="questions.fpa_fail_rate.run",
        examples=[
            "fpa fail rate by team last 3 months",
            "first pass accuracy failures by manager aug 2025",
        ],
    ),
    QuestionSpec(
        qid="fpa_fail_drivers",
        title="Top FPA fail drivers",
        handler="questions.fpa_fail_drivers.run",
        examples=[
            "biggest drivers of case fails",
            "fpa fail reasons last 3 months",
        ],
    ),
    QuestionSpec(
        qid="rca1_portfolio_process",
        title="RCA1 by portfolio for a process",
        handler="questions.rca1_portfolio_process.run",
        examples=[
            'show rca1 by portfolio for process "Member Enquiry" last 3 months',
            "reasons by portfolio for process member enquiry",
        ],
    ),
    QuestionSpec(
        qid="corr_nps",
        title="Complaints vs NPS correlation",
        handler="questions.corr_nps.run",
        examples=[
            "correlation between complaints per 1000 and nps",
            "nps vs complaints trend",
        ],
    ),
]

# ----------------- helpers -----------------
def _choices_from_store(store) -> Tuple[List[str], List[str]]:
    ports, procs = [], []
    try:
        if "cases" in store:
            c = store["cases"]
            if "Portfolio_std" in c.columns: ports += c["Portfolio_std"].dropna().unique().tolist()
            if "ProcessName" in c.columns:   procs += c["ProcessName"].dropna().unique().tolist()
        if "complaints" in store:
            d = store["complaints"]
            if "Portfolio_std" in d.columns: ports += d["Portfolio_std"].dropna().unique().tolist()
            if "Parent_Case_Type" in d.columns: procs += d["Parent_Case_Type"].dropna().unique().tolist()
    except Exception:
        pass
    return sorted(set(ports)), sorted(set(procs))

def _fuzzy(q: str, choices: List[str], threshold=70) -> Optional[str]:
    if not choices: return None
    match, score, _ = rf_process.extractOne(q, choices)
    return match if score >= threshold else None

def _pick_qid(prompt: str) -> str:
    phrases, labels = [], []
    for s in CATALOG:
        for ex in s.examples:
            phrases.append(ex); labels.append(s.qid)
    match, score, _ = rf_process.extractOne(prompt, phrases)
    if match: return labels[phrases.index(match)]
    p = prompt.lower()
    if "rca" in p or "reason" in p: return "rca1_portfolio_process"
    if "first pass" in p or "fpa" in p:
        if "driver" in p or "reason" in p: return "fpa_fail_drivers"
        return "fpa_fail_rate"
    if "unique case" in p or "case volume" in p: return "unique_cases_mom"
    if "correlation" in p or "corr nps" in p: return "corr_nps"
    if "overview" in p: return "mom_overview"
    if "per 1000" in p or "per thousand" in p: return "complaints_per_thousand"
    return "complaint_volume_rate"

@dataclass
class Parsed:
    qid: str
    portfolio: Optional[str]
    process: Optional[str]
    month_range: Optional[Tuple[str,str]]
    last_n: Optional[int]
    by_dim: Optional[str]

def parse_prompt(prompt: str, store) -> Parsed:
    qid = _pick_qid(prompt)
    mr, lastn = parse_months(prompt)
    ports, procs = _choices_from_store(store)
    portfolio = _fuzzy(prompt, ports)
    process   = _fuzzy(prompt, procs)

    low = f" {prompt.lower()} "
    by_dim = None
    for token, key in [
        (" by team ", "team"), (" by manager ", "manager"), (" by location ", "location"),
        (" by process ", "process"), (" by portfolio ", "portfolio"), (" by scheme ", "scheme"),
    ]:
        if token in low: by_dim = key; break

    return Parsed(qid=qid, portfolio=portfolio, process=process,
                  month_range=mr, last_n=lastn, by_dim=by_dim)

def _import_handler(path: str):
    mod, func = path.rsplit(".", 1)
    m = __import__(mod, fromlist=[func])
    return getattr(m, func)

def route(prompt: str, store: Dict):
    parsed = parse_prompt(prompt, store)
    spec = next(s for s in CATALOG if s.qid == parsed.qid)
    handler = _import_handler(spec.handler)
    handler(store, parsed)  # handler is expected to render with Streamlit
