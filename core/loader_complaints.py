# core/loader_complaints.py
from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Optional
import re
import pandas as pd

from core.rca_labeller import label_complaints_rca

DATE_CANDIDATES = [
    "Report Month", "Report_Month", "Report_Date", "ReportDate",
    "Date", "Created Date", "Create Date"  # last two are rare in complaints, but supported
]

PORTFOLIO_CANDIDATES = ["Portfolio_std", "Portfolio", "Portfolio Name", "Location Portfolio"]
PROCESS_CANDIDATES   = ["ProcessName", "Process Name", "Process", "Process_Group"]
PARENT_CASE_CANDIDATES = ["Parent Case Type", "Parent_Case_Type", "Parent case type"]
TEAM_CANDIDATES      = ["Parent Team", "Team", "Team Name", "TeamName"]
SCHEME_CANDIDATES    = ["Scheme", "Scheme Name"]
RECEIPT_CANDIDATES   = ["Receipt Method", "Receipt_Method", "Receipt Channel"]
APTIA_ERR_CANDS      = ["Aptia Error", "Aptia_Error", "Is Aptia Error?"]
CONTROL_CANDIDATES   = ["Control", "Control Type", "Control_Flag"]
WHY_CANDS            = ["Why", "Root Cause", "Root_Cause", "Why?"]
ADMIN_RCA_CANDS      = ["Brief Description - RCA done by admin",
                        "Brief Description – RCA done by admin",
                        "Brief_Description_-_RCA_done_by_admin",
                        "Admin_RCA", "RCA_Admin"]

def _read_one(path: Path) -> pd.DataFrame:
    try:
        return pd.read_excel(path, engine="openpyxl")
    except Exception:
        # last resort, csv
        try:
            return pd.read_csv(path)
        except Exception as e:
            raise RuntimeError(f"Failed to read {path}: {e}")

def _first_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None

def _to_month_str(s: pd.Series) -> pd.Series:
    dt = pd.to_datetime(s, errors="coerce", dayfirst=False)
    return dt.dt.to_period("M").astype(str)  # YYYY-MM

def _std_portfolio(v: str) -> str:
    if not isinstance(v, str):
        return ""
    x = v.strip().lower()
    # normalize common variants; extend freely
    repl = {
        "london": "London",
        "lon": "London",
        "chichester": "Chichester",
        "chi": "Chichester",
        "pune": "Pune - Mer",
        "gurgaon": "Gurgaon",
        "mumbai": "Mumbai",
    }
    return repl.get(x, v.strip())

def _std_process(v: str) -> str:
    if not isinstance(v, str):
        return ""
    return re.sub(r"\s+", " ", v.strip())

def _ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Create normalized columns used by downstream joins."""
    out = df.copy()

    # Month
    dc = _first_col(out, DATE_CANDIDATES)
    if dc is None:
        out["Month"] = ""
    else:
        out["Month"] = _to_month_str(out[dc])

    # Portfolio_std
    pc = _first_col(out, PORTFOLIO_CANDIDATES)
    out["Portfolio_std"] = out[pc].map(_std_portfolio) if pc else ""

    # ProcessName — prefer explicit process; if missing, fall back to Parent Case Type
    prc = _first_col(out, PROCESS_CANDIDATES)
    pct = _first_col(out, PARENT_CASE_CANDIDATES)
    if prc:
        out["ProcessName"] = out[prc].map(_std_process)
    elif pct:
        out["ProcessName"] = out[pct].map(_std_process)
    else:
        out["ProcessName"] = ""

    # Parent_Case_Type (keep a clean copy for reports)
    if pct:
        out["Parent_Case_Type"] = out[pct].astype(str).str.strip()
    else:
        out["Parent_Case_Type"] = ""

    # TeamName (from Parent Team etc.)
    tc = _first_col(out, TEAM_CANDIDATES)
    out["TeamName"] = out[tc].astype(str).str.strip() if tc else ""

    # Scheme
    sc = _first_col(out, SCHEME_CANDIDATES)
    out["Scheme"] = out[sc].astype(str).str.strip() if sc else ""

    # Receipt Method
    rm = _first_col(out, RECEIPT_CANDIDATES)
    out["Receipt_Method"] = out[rm].astype(str).str.strip() if rm else ""

    # Aptia Error / Control flags (keep original names if present)
    ae = _first_col(out, APTIA_ERR_CANDS)
    out["Aptia_Error"] = out[ae] if ae else ""
    ctrl = _first_col(out, CONTROL_CANDIDATES)
    out["Control"] = out[ctrl] if ctrl else ""

    # Keep raw RCA columns (optional; rca_labeller builds RCA_text if missing)
    why_c = _first_col(out, WHY_CANDS)
    adm_c = _first_col(out, ADMIN_RCA_CANDS)
    if why_c and why_c not in out.columns:
        out["Why"] = out[why_c]
    if adm_c and adm_c not in out.columns:
        out["Brief Description - RCA done by admin"] = out[adm_c]

    return out

def load_complaints(path: str | Path) -> pd.DataFrame:
    """
    Loads all complaints files from a folder, normalizes fields, derives Month,
    standardizes Portfolio_std / ProcessName / Parent_Case_Type / TeamName / Scheme /
    Receipt_Method, and attaches RCA1/RCA2 using core.rca_labeller.

    Returns a single, clean DataFrame ready for joining and KPIs.
    """
    folder = Path(path)
    if not folder.exists():
        return pd.DataFrame()

    files = sorted([p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in {".xlsx", ".xls", ".csv"}])
    if not files:
        return pd.DataFrame()

    frames: List[pd.DataFrame] = []
    for f in files:
        df = _read_one(f)
        if df is None or df.empty:
            continue
        # normalize
        df = _ensure_columns(df)
        frames.append(df)

    if not frames:
        return pd.DataFrame()

    allc = pd.concat(frames, ignore_index=True)
    # Label RCA from free text ("Admin RCA" + "Why")
    allc = label_complaints_rca(allc, patterns_file="data/rca_patterns.yml")
    # final hygiene
    allc["Month"] = allc["Month"].fillna("").astype(str)
    allc["Portfolio_std"] = allc["Portfolio_std"].fillna("").astype(str)
    allc["ProcessName"] = allc["ProcessName"].fillna("").astype(str)
    allc["Parent_Case_Type"] = allc["Parent_Case_Type"].fillna("").astype(str)

    return allc
