# data_store.py
from __future__ import annotations

from pathlib import Path
import pandas as pd
import streamlit as st

DATA_DIR = Path("data")

def _first_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    cols = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols:
            return cols[cand.lower()]
    return None

def _norm_str(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower()

def _as_month_start(dt: pd.Series) -> pd.Series:
    return pd.to_datetime(dt, errors="coerce").dt.to_period("M").dt.to_timestamp()

def _read_any_excel_or_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    return pd.read_excel(path)

def _concat_dir(dir_path: Path) -> pd.DataFrame:
    if not dir_path.exists():
        return pd.DataFrame()
    parts = []
    for p in dir_path.glob("*"):
        if p.suffix.lower() in (".xlsx", ".xls", ".parquet", ".pq", ".csv"):
            try:
                parts.append(_read_any_excel_or_parquet(p))
            except Exception:
                pass
    if parts:
        return pd.concat(parts, ignore_index=True)
    return pd.DataFrame()

@st.cache_data(show_spinner=False)
def load_store(assume_year_for_complaints: int = 2025, **_cache_busters) -> dict:
    """
    Loads and standardizes data from data/cases, data/complaints, data/fpa.
    - Cases date: Create Date -> 'case_date' + '_month_dt'
    - Complaints date: 'Date Complaint Received - DD/MM/YY' -> 'complaint_date' + '_month_dt'
      Fallback: if a 'Month' column is present, assume provided year (default=2025).
    """
    cases_raw = _concat_dir(DATA_DIR / "cases")
    complaints_raw = _concat_dir(DATA_DIR / "complaints")
    fpa_raw = _concat_dir(DATA_DIR / "fpa")

    # ---------- CASES ----------
    if not cases_raw.empty:
        id_col   = _first_col(cases_raw, ["Case ID", "CaseID", "Id", "ID"])
        port_col = _first_col(cases_raw, ["Portfolio"])
        proc_col = _first_col(cases_raw, ["Process", "Process Name"])
        date_col = _first_col(cases_raw, ["Create Date", "Created Date", "Case Created", "Start Date"])

        cases = cases_raw.rename(columns={
            id_col: "id",
            port_col: "portfolio",
            proc_col: "process",
            date_col: "case_date",
        })

        if "portfolio" in cases:
            cases["portfolio"] = _norm_str(cases["portfolio"])
        if "process" in cases:
            cases["process"] = cases["process"].astype(str).str.strip()
        if "case_date" in cases:
            cases["case_date"] = pd.to_datetime(cases["case_date"], errors="coerce", dayfirst=True)
            cases["_month_dt"] = _as_month_start(cases["case_date"])
        else:
            cases["_month_dt"] = pd.NaT
    else:
        cases = pd.DataFrame(columns=["id", "portfolio", "process", "case_date", "_month_dt"])

    # ---------- COMPLAINTS ----------
    if not complaints_raw.empty:
        port_c  = _first_col(complaints_raw, ["Portfolio"])
        proc_c  = _first_col(complaints_raw, ["Parent Case Type", "Process", "Process Name"])
        recv_c  = _first_col(complaints_raw, ["Date Complaint Received - DD/MM/YY", "Complaint Received Date", "Date Received"])
        month_c = _first_col(complaints_raw, ["Month"])

        complaints = complaints_raw.rename(columns={
            port_c: "portfolio",
            proc_c: "process",
            recv_c: "complaint_date",
            month_c: "month_text",
        })

        if "portfolio" in complaints:
            complaints["portfolio"] = _norm_str(complaints["portfolio"])
        if "process" in complaints:
            complaints["process"] = complaints["process"].astype(str).str.strip()

        if "complaint_date" in complaints and complaints["complaint_date"].notna().any():
            complaints["complaint_date"] = pd.to_datetime(
                complaints["complaint_date"], errors="coerce", dayfirst=True
            )
            complaints["_month_dt"] = _as_month_start(complaints["complaint_date"])
        else:
            # Fallback to Month text (e.g., "June")
            if "month_text" in complaints:
                m = complaints["month_text"].astype(str).str.strip()
                complaints["_month_dt"] = pd.to_datetime(
                    f"{assume_year_for_complaints}-" + m + "-01", errors="coerce"
                ).dt.to_period("M").dt.to_timestamp()
            else:
                complaints["_month_dt"] = pd.NaT
    else:
        complaints = pd.DataFrame(columns=["portfolio", "process", "complaint_date", "_month_dt"])

    return {"cases": cases, "complaints": complaints, "fpa": fpa_raw}
