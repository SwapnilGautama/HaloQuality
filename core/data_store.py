# data_store.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import pandas as pd
import streamlit as st


# ---------------------------
# Column mapping helpers
# ---------------------------
def _first_col(df: pd.DataFrame, candidates: List[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
        # forgive subtle differences
        for col in df.columns:
            if col.strip().lower() == c.strip().lower():
                return col
    return None

def _to_month_start(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce", dayfirst=True).dt.to_period("M").dt.to_timestamp()


# ---------------------------
# Normalizers
# ---------------------------
def _normalize_cases(df: pd.DataFrame) -> pd.DataFrame:
    """
    Expect:
      - ID: 'Case ID' (or similar)
      - Portfolio: 'Portfolio'
      - Process (optional): 'Process' or 'Process Name'
      - Date (month key): Prefer 'Create Date', fallback 'Report_Date' / 'Report Date'
    """
    df = df.copy()

    id_col = _first_col(df, ["Case ID", "CaseID", "Unique Identifier", "Unique identifier", "id"])
    if id_col is None:
        raise ValueError("Cases: missing Case ID column.")
    portfolio_col = _first_col(df, ["Portfolio"])
    if portfolio_col is None:
        raise ValueError("Cases: missing Portfolio column.")

    process_col = _first_col(df, ["Process Name", "Process", "Process  ", "Process_name"])
    # date preference
    date_col = _first_col(df, ["Create Date", "Create_Date", "Report_Date", "Report Date"])
    if date_col is None:
        raise ValueError("Cases: missing date column (Create Date / Report_Date).")

    out = pd.DataFrame(
        {
            "id": df[id_col],
            "portfolio": df[portfolio_col],
            "process": df[process_col] if process_col else None,
            "date": pd.to_datetime(df[date_col], errors="coerce", dayfirst=True),
        }
    )
    # if Create Date present but empty and Report_Date exists, swap in
    if "Create Date" not in df.columns and "Report_Date" in df.columns:
        pass  # we already used Report_Date
    else:
        # if many NaT in date because Create Date empty, try to fallback to report date
        if out["date"].isna().mean() > 0.5:
            rd = _first_col(df, ["Report_Date", "Report Date"])
            if rd:
                out["date"] = out["date"].fillna(pd.to_datetime(df[rd], errors="coerce", dayfirst=True))

    out["month"] = _to_month_start(out["date"])
    return out.dropna(subset=["date"]).reset_index(drop=True)


def _normalize_complaints(df: pd.DataFrame) -> pd.DataFrame:
    """
    Expect:
      - Linked case (optional): 'Original Process Affected Case ID' == Cases 'Case ID'
      - Portfolio: 'Portfolio'
      - Process (for display if available): 'Parent Case Type'
      - Date: 'Date Complaint Received - DD/MM/YY'
    """
    df = df.copy()

    portfolio_col = _first_col(df, ["Portfolio"])
    if portfolio_col is None:
        raise ValueError("Complaints: missing Portfolio column.")

    process_col = _first_col(df, ["Parent Case Type", "Parent Case type", "ParentCaseType"])
    link_col = _first_col(df, ["Original Process Affected Case ID", "Original Case ID", "Original CaseId"])
    date_col = _first_col(df, ["Date Complaint Received - DD/MM/YY", "Date Complaint Received", "Complaint Date"])

    if date_col is None:
        raise ValueError("Complaints: missing 'Date Complaint Received - DD/MM/YY' column.")

    out = pd.DataFrame(
        {
            "portfolio": df[portfolio_col],
            "process": df[process_col] if process_col else None,
            "case_link": df[link_col] if link_col else None,
            "date": pd.to_datetime(df[date_col], errors="coerce", dayfirst=True),
        }
    )
    out["month"] = _to_month_start(out["date"])
    return out.dropna(subset=["date"]).reset_index(drop=True)


def _read_many(paths: List[Path]) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for p in paths:
        try:
            if p.suffix.lower() in [".xlsx", ".xls"]:
                frames.append(pd.read_excel(p))
            elif p.suffix.lower() in [".parquet", ".pq"]:
                frames.append(pd.read_parquet(p))
        except Exception:
            # tolerate odd files
            continue
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _discover_files(kind: str) -> List[Path]:
    """
    naive discovery: search typical folders & names
    """
    roots = [Path("."), Path("./data"), Path("./core"), Path("./datasets")]
    globs = {
        "cases": ["*case*.xlsx", "*cases*.xlsx", "*case*.parquet", "*cases*.parquet"],
        "complaints": ["*complaint*.xlsx", "*complaints*.xlsx", "*complaint*.parquet", "*complaints*.parquet"],
        "fpa": ["*fpa*.xlsx", "*first*pass*.xlsx", "*fpa*.parquet"],
    }[kind]

    paths: List[Path] = []
    for r in roots:
        for pat in globs:
            paths.extend(r.rglob(pat))
    # de-dupe
    seen = set()
    uniq = []
    for p in paths:
        if p.resolve() not in seen:
            uniq.append(p)
            seen.add(p.resolve())
    return uniq


@st.cache_data(show_spinner=False)
def load_store() -> Dict[str, pd.DataFrame]:
    # Discover & read
    cases_paths = _discover_files("cases")
    complaints_paths = _discover_files("complaints")
    fpa_paths = _discover_files("fpa")

    raw_cases = _read_many(cases_paths)
    raw_complaints = _read_many(complaints_paths)
    raw_fpa = _read_many(fpa_paths)

    # Normalize
    cases = _normalize_cases(raw_cases) if not raw_cases.empty else pd.DataFrame(columns=["id", "portfolio", "process", "date", "month"])
    complaints = _normalize_complaints(raw_complaints) if not raw_complaints.empty else pd.DataFrame(columns=["portfolio", "process", "case_link", "date", "month"])
    fpa = raw_fpa

    return {"cases": cases, "complaints": complaints, "fpa": fpa}
