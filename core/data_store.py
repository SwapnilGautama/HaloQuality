# core/data_store.py
from __future__ import annotations
import os
from pathlib import Path
import hashlib
import pandas as pd
import streamlit as st

DATA_DIR = Path("data")

# ---------- helpers

def _hash_file(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(2**20), b""):
            h.update(chunk)
    return h.hexdigest()

def _to_parquet_once(xlsx_path: Path, sheet_name: str | None = None) -> Path:
    """
    Convert an Excel file (or a specific sheet) to Parquet the first time we see it,
    and re-materialize if the XLSX changes. Returns the Parquet path.
    """
    pq_name = xlsx_path.with_suffix(f".{sheet_name or 'sheet'}.parquet")
    sig_path = xlsx_path.with_suffix(f".{sheet_name or 'sheet'}.sig")
    x_sig = _hash_file(xlsx_path) + (f":{sheet_name}" if sheet_name else "")

    current_sig = sig_path.read_text() if sig_path.exists() else ""
    if not pq_name.exists() or current_sig != x_sig:
        df = pd.read_excel(xlsx_path, sheet_name=sheet_name, engine="openpyxl")
        df.to_parquet(pq_name, index=False)
        sig_path.write_text(x_sig)
    return pq_name

def _read_xlsx_fast(path: Path, sheet_name: str | None = None) -> pd.DataFrame:
    # Materialize to parquet and read fast
    pq = _to_parquet_once(path, sheet_name)
    return pd.read_parquet(pq)

def _mmmyy(dt: pd.Series) -> pd.Series:
    return dt.dt.strftime("%b %y")

def _coerce_date(series: pd.Series, dayfirst: bool = True) -> pd.Series:
    return pd.to_datetime(series, errors="coerce", dayfirst=dayfirst)

# ---------- loaders (each returns a DF with 'month_dt' and 'month')

def load_cases(path: Path) -> pd.DataFrame:
    # Load all XLSX files under data/cases (newest pattern you’re using)
    files = sorted(Path(path).glob("*.xlsx"))
    dfs = []
    for f in files:
        df = _read_xlsx_fast(f)
        # Required columns we’ll use repeatedly
        # If columns differ slightly per file, .get with fallback keeps it robust
        if "Create Date" in df.columns:
            dt = _coerce_date(df["Create Date"], dayfirst=True)
        elif "Create_Date" in df.columns:
            dt = _coerce_date(df["Create_Date"], dayfirst=True)
        else:
            dt = pd.NaT

        df["month_dt"] = dt.dt.to_period("M").dt.to_timestamp()
        df["month"] = _mmmyy(df["month_dt"])

        # common dims (take as-is if they exist)
        for c in [
            "Case ID","Portfolio_std","Portfolio","Process Name","Parent Case Type",
            "Team Name","Process Group","Onshore/Offshore","Manual/RPA","Location",
            "ClientName","Scheme"
        ]:
            if c not in df.columns:
                df[c] = pd.NA

        # make sure Case ID is string to avoid dtype mismatches
        if "Case ID" in df.columns:
            df["Case ID"] = df["Case ID"].astype(str)

        dfs.append(df[[
            "Case ID","month_dt","month",
            "Portfolio_std","Portfolio","Process Name","Parent Case Type",
            "Team Name","Process Group","Onshore/Offshore","Manual/RPA","Location",
            "ClientName","Scheme"
        ]])
    if not dfs:
        return pd.DataFrame(columns=[
            "Case ID","month_dt","month","Portfolio_std","Portfolio","Process Name",
            "Parent Case Type","Team Name","Process Group","Onshore/Offshore",
            "Manual/RPA","Location","ClientName","Scheme"
        ])
    out = pd.concat(dfs, ignore_index=True)
    return out

def load_complaints(path: Path) -> pd.DataFrame:
    # one or more complaint workbooks
    files = sorted(Path(path).glob("*.xlsx"))
    dfs = []
    for f in files:
        df = _read_xlsx_fast(f)
        # date per your spec: "Date Complaint Received - DD/MM/YY"
        date_cols = [
            "Date Complaint Received - DD/MM/YY",
            "Date Complaint Received",
            "Date_Complaint_Received"
        ]
        date_col = next((c for c in date_cols if c in df.columns), None)
        dt = _coerce_date(df[date_col], dayfirst=True) if date_col else pd.NaT

        df["month_dt"] = dt.dt.to_period("M").dt.to_timestamp()
        df["month"] = _mmmyy(df["month_dt"])

        # id column (if present)
        if "Case ID" not in df.columns:
            df["Case ID"] = pd.NA

        # align dims we join on
        for c in ["Portfolio_std","Portfolio","Process Name","Parent Case Type"]:
            if c not in df.columns:
                df[c] = pd.NA

        dfs.append(df[["Case ID","month_dt","month","Portfolio_std","Portfolio",
                       "Process Name","Parent Case Type"]].copy())

    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame(
        columns=["Case ID","month_dt","month","Portfolio_std","Portfolio",
                 "Process Name","Parent Case Type"]
    )

def load_fpa(path: Path) -> pd.DataFrame:
    files = sorted(Path(path).glob("*.xlsx"))
    dfs = []
    for f in files:
        df = _read_xlsx_fast(f)
        date_cols = ["Activity Date","Activity_Date"]
        date_col = next((c for c in date_cols if c in df.columns), None)
        dt = _coerce_date(df[date_col], dayfirst=True) if date_col else pd.NaT
        df["month_dt"] = dt.dt.to_period("M").dt.to_timestamp()
        df["month"] = _mmmyy(df["month_dt"])
        for c in ["Portfolio_std","Portfolio","Process Name","Team Name","Team Manager","Scheme","Location","Review Result","Case Comment"]:
            if c not in df.columns:
                df[c] = pd.NA
        dfs.append(df[["month_dt","month","Portfolio_std","Portfolio","Process Name","Team Name","Team Manager","Scheme","Location","Review Result","Case Comment"]])
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame(
        columns=["month_dt","month","Portfolio_std","Portfolio","Process Name","Team Name","Team Manager","Scheme","Location","Review Result","Case Comment"]
    )

def load_checker(path: Path) -> pd.DataFrame:
    files = sorted(Path(path).glob("*.xlsx"))
    dfs = []
    for f in files:
        df = _read_xlsx_fast(f)
        date_cols = ["Date Completed","Review Date","Date_Completed","Review_Date"]
        date_col = next((c for c in date_cols if c in df.columns), None)
        dt = _coerce_date(df[date_col], dayfirst=True) if date_col else pd.NaT
        df["month_dt"] = dt.dt.to_period("M").dt.to_timestamp()
        df["month"] = _mmmyy(df["month_dt"])
        dfs.append(df[["month_dt","month"] + [c for c in df.columns if c not in ("month_dt","month")]])
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame(columns=["month_dt","month"])

def load_surveys(path: Path) -> pd.DataFrame:
    files = sorted(Path(path).glob("*.xlsx"))
    dfs = []
    for f in files:
        df = _read_xlsx_fast(f)
        # Some sheets will already have month text; normalize to Timestamp → MMM YY
        s = df["Month_received"] if "Month_received" in df.columns else pd.Series(pd.NaT, index=df.index)
        # Try both text and numeric
        dt = pd.to_datetime(s, errors="coerce")
        if dt.isna().all():
            # if it’s like "Jun-25" or "06/2025" in another col, we could add extra heuristics here
            pass
        df["month_dt"] = dt.dt.to_period("M").dt.to_timestamp()
        df["month"] = _mmmyy(df["month_dt"])
        dfs.append(df[["month_dt","month"] + [c for c in df.columns if c not in ("month_dt","month")]])
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame(columns=["month_dt","month"])

# ---------- public (cached) entry point

@st.cache_resource(show_spinner=False)
def load_store() -> dict:
    """
    Returns a dict of DFs with unified monthly keys:
      * month_dt  (Timestamp at month start)
      * month     (MMM YY)
    Only called once per session thanks to cache_resource.
    """
    cases = load_cases(DATA_DIR / "cases")
    complaints = load_complaints(DATA_DIR / "complaints")
    fpa = load_fpa(DATA_DIR / "first_pass_accuracy")
    checker = load_checker(DATA_DIR / "checker_accuracy")
    surveys = load_surveys(DATA_DIR / "surveys")

    # Light slimming
    for name, df in (("cases", cases), ("complaints", complaints), ("fpa", fpa), ("checker", checker), ("surveys", surveys)):
        if "month_dt" in df.columns:
            df.sort_values("month_dt", inplace=True)

    return {
        "cases": cases,
        "complaints": complaints,
        "fpa": fpa,
        "checker": checker,
        "surveys": surveys,
    }
