# core/data_store.py
from __future__ import annotations
from pathlib import Path
import hashlib
import pandas as pd
import streamlit as st

DATA_DIR = Path("data")

# ---------- generic helpers

def _hash_file(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(2**20), b""):
            h.update(chunk)
    return h.hexdigest()

def _sanitize_for_parquet(df: pd.DataFrame) -> pd.DataFrame:
    """Make a DataFrame Arrow-safe: drop empty cols, cast mixed/object to string."""
    out = df.copy()

    # drop fully empty columns
    empty_cols = [c for c in out.columns if out[c].isna().all()]
    if empty_cols:
        out = out.drop(columns=empty_cols)

    # cast object / mixed columns to pandas 'string' dtype
    for c in out.columns:
        if (
            pd.api.types.is_object_dtype(out[c])
            or pd.api.types.infer_dtype(out[c], skipna=True) in
               ("mixed", "mixed-integer", "mixed-integer-float", "mixed-string", "string", "unicode")
        ):
            out[c] = out[c].astype("string")

    return out

def _to_parquet_once(xlsx_path: Path, sheet_name: int | str = 0) -> Path:
    """
    Convert an Excel sheet to Parquet once (or if XLSX changes).
    Defaults to first sheet so pandas returns a DataFrame (not a dict).
    """
    df = pd.read_excel(xlsx_path, sheet_name=sheet_name, engine="openpyxl")

    # If caller supplied something that produced multiple sheets, choose first non-empty.
    chosen_key = str(sheet_name)
    if isinstance(df, dict):
        try:
            chosen_key, df = next(
                (k, v) for k, v in df.items()
                if isinstance(v, pd.DataFrame) and not v.empty
            )
        except StopIteration:
            first_key = next(iter(df.keys()))
            chosen_key, df = first_key, df[first_key]

    pq_name = xlsx_path.with_suffix(f".{chosen_key}.parquet")
    sig_path = xlsx_path.with_suffix(f".{chosen_key}.sig")
    x_sig = _hash_file(xlsx_path) + f":{chosen_key}"
    current_sig = sig_path.read_text() if sig_path.exists() else ""

    if not pq_name.exists() or current_sig != x_sig:
        safe_df = _sanitize_for_parquet(df)
        try:
            safe_df.to_parquet(pq_name, index=False)
        except Exception:
            # last resort: cast everything to string so Arrow will accept it
            safe_df = safe_df.astype("string")
            safe_df.to_parquet(pq_name, index=False)
        sig_path.write_text(x_sig)

    return pq_name

def _read_xlsx_fast(path: Path, sheet_name: int | str = 0) -> pd.DataFrame:
    pq = _to_parquet_once(path, sheet_name=sheet_name)
    return pd.read_parquet(pq)

def _mmmyy(dt: pd.Series) -> pd.Series:
    return dt.dt.strftime("%b %y")

def _coerce_date(series: pd.Series, dayfirst: bool = True) -> pd.Series:
    return pd.to_datetime(series, errors="coerce", dayfirst=dayfirst)

# ---------- specific loaders (all create month_dt + month)

def load_cases(path: Path) -> pd.DataFrame:
    files = sorted(Path(path).glob("*.xlsx"))
    dfs = []
    for f in files:
        df = _read_xlsx_fast(f, sheet_name=0)
        # Create Date (cases)
        if "Create Date" in df.columns:
            dt = _coerce_date(df["Create Date"], dayfirst=True)
        elif "Create_Date" in df.columns:
            dt = _coerce_date(df["Create_Date"], dayfirst=True)
        else:
            dt = pd.NaT

        df["month_dt"] = dt.dt.to_period("M").dt.to_timestamp()
        df["month"] = _mmmyy(df["month_dt"])

        for c in [
            "Case ID","Portfolio_std","Portfolio","Process Name","Parent Case Type",
            "Team Name","Process Group","Onshore/Offshore","Manual/RPA","Location",
            "ClientName","Scheme"
        ]:
            if c not in df.columns:
                df[c] = pd.NA

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
    return pd.concat(dfs, ignore_index=True)

def load_complaints(path: Path) -> pd.DataFrame:
    files = sorted(Path(path).glob("*.xlsx"))
    dfs = []
    for f in files:
        df = _read_xlsx_fast(f, sheet_name=0)
        # Complaints date: "Date Complaint Received - DD/MM/YY"
        date_cols = [
            "Date Complaint Received - DD/MM/YY",
            "Date Complaint Received",
            "Date_Complaint_Received"
        ]
        date_col = next((c for c in date_cols if c in df.columns), None)
        dt = _coerce_date(df[date_col], dayfirst=True) if date_col else pd.NaT
        df["month_dt"] = dt.dt.to_period("M").dt.to_timestamp()
        df["month"] = _mmmyy(df["month_dt"])

        if "Case ID" not in df.columns:
            df["Case ID"] = pd.NA
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
        df = _read_xlsx_fast(f, sheet_name=0)
        # FPA date: Activity Date
        date_col = next((c for c in ["Activity Date","Activity_Date"] if c in df.columns), None)
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
        df = _read_xlsx_fast(f, sheet_name=0)
        # Checker date: Date Completed / Review Date
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
        df = _read_xlsx_fast(f, sheet_name=0)
        # Surveys month: Month_received (already monthly)
        s = df["Month_received"] if "Month_received" in df.columns else pd.Series(pd.NaT, index=df.index)
        dt = pd.to_datetime(s, errors="coerce")
        df["month_dt"] = dt.dt.to_period("M").dt.to_timestamp()
        df["month"] = _mmmyy(df["month_dt"])
        dfs.append(df[["month_dt","month"] + [c for c in df.columns if c not in ("month_dt","month")]])
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame(columns=["month_dt","month"])

# ---------- cached store

@st.cache_resource(show_spinner=False)
def load_store() -> dict:
    cases = load_cases(DATA_DIR / "cases")
    complaints = load_complaints(DATA_DIR / "complaints")
    fpa = load_fpa(DATA_DIR / "first_pass_accuracy")
    checker = load_checker(DATA_DIR / "checker_accuracy")
    surveys = load_surveys(DATA_DIR / "surveys")

    for df in (cases, complaints, fpa, checker, surveys):
        if "month_dt" in df.columns:
            df.sort_values("month_dt", inplace=True)

    return {
        "cases": cases,
        "complaints": complaints,
        "fpa": fpa,
        "checker": checker,
        "surveys": surveys,
    }
