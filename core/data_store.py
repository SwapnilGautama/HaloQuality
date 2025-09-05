# core/data_store.py
# Loads all datasets and standardizes months across tables
# Creates two columns on every table:
#   Month        -> pd.Timestamp at first day of month
#   Month_label  -> "MMM YY" (e.g., "Jun 25")

from __future__ import annotations
import os
import re
import pandas as pd
from typing import List, Dict, Optional

DATA_ROOT = "data"

# ---------- small utils ----------
def _canon(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).lower())

def _find_first_col(df: pd.DataFrame, preferred_names: List[str]) -> Optional[str]:
    """Find the first matching column by canonical name (case/space/underscore/dash insensitive).
    Falls back to substring search with the first preferred name."""
    if df is None or df.empty:
        return None
    canon_map = {_canon(c): c for c in df.columns}
    for name in preferred_names:
        key = _canon(name)
        if key in canon_map:
            return canon_map[key]
    # fallback: substring search on the first pref token
    token = _canon(preferred_names[0])
    for c in df.columns:
        if token in _canon(c):
            return c
    return None

def _add_month(df: pd.DataFrame, date_col: str, *, dayfirst: bool = True) -> pd.DataFrame:
    """Adds Month (Timestamp) and Month_label (MMM YY) columns based on date_col."""
    if df is None or df.empty:
        return df
    # For UK-style dates, dayfirst=True helps (DD/MM/YY in complaints)
    dt = pd.to_datetime(df[date_col], errors="coerce", dayfirst=dayfirst)
    m = dt.dt.to_period("M").dt.to_timestamp()
    df["Month"] = m
    df["Month_label"] = df["Month"].dt.strftime("%b %y")
    return df

def _read_folder(folder: str) -> pd.DataFrame:
    """Read all CSV/XLS/XLSX under folder and concat. Empty if folder missing."""
    if not os.path.isdir(folder):
        return pd.DataFrame()
    parts = []
    for fname in os.listdir(folder):
        fpath = os.path.join(folder, fname)
        if not os.path.isfile(fpath):
            continue
        ext = os.path.splitext(fname)[1].lower()
        try:
            if ext in [".csv"]:
                parts.append(pd.read_csv(fpath))
            elif ext in [".xls", ".xlsx", ".xlsm"]:
                parts.append(pd.read_excel(fpath))
        except Exception:
            # if a single file fails, skip it but keep others
            continue
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


# ---------- loaders (enforce your date rules) ----------
def load_cases(path: str = os.path.join(DATA_ROOT, "cases")) -> pd.DataFrame:
    df = _read_folder(path)
    if df.empty:
        return df
    # Create Date
    col = _find_first_col(df, ["Create Date", "Create_Date", "createdate", "create"])
    if not col:
        raise ValueError("Cases: required date column 'Create Date' (or close variant) not found.")
    _add_month(df, col, dayfirst=True)
    return df


def load_complaints(path: str = os.path.join(DATA_ROOT, "complaints")) -> pd.DataFrame:
    df = _read_folder(path)
    if df.empty:
        return df
    # Date Complaint Received - DD/MM/YY
    col = _find_first_col(df, ["Date Complaint Received - DD/MM/YY", "Date Complaint Received", "complaint received"])
    if not col:
        raise ValueError("Complaints: required date column 'Date Complaint Received - DD/MM/YY' not found.")
    _add_month(df, col, dayfirst=True)
    return df


def load_fpa(path: str = os.path.join(DATA_ROOT, "first_pass_accuracy")) -> pd.DataFrame:
    df = _read_folder(path)
    if df.empty:
        return df
    # Activity Date
    col = _find_first_col(df, ["Activity Date", "activity_date"])
    if not col:
        raise ValueError("FPA: required date column 'Activity Date' not found.")
    _add_month(df, col, dayfirst=True)
    return df


def load_checker_accuracy(path: str = os.path.join(DATA_ROOT, "checker_accuracy")) -> pd.DataFrame:
    df = _read_folder(path)
    if df.empty:
        return df
    # Date Completed OR Review Date (prefer Date Completed)
    col = _find_first_col(df, ["Date Completed", "date completed", "Review Date", "review date"])
    if not col:
        raise ValueError("Checker accuracy: required date column 'Date Completed' or 'Review Date' not found.")
    _add_month(df, col, dayfirst=True)
    return df


def load_surveys(path: str = os.path.join(DATA_ROOT, "surveys")) -> pd.DataFrame:
    df = _read_folder(path)
    if df.empty:
        return df
    # Month_received
    col = _find_first_col(df, ["Month_received", "month_received", "month received"])
    if not col:
        raise ValueError("Surveys: required date column 'Month_received' not found.")
    # Surveys could already be month-like; treat generically
    _add_month(df, col, dayfirst=True)
    return df


# ---------- bundle loader ----------
def load_store(
    cases_dir: str = os.path.join(DATA_ROOT, "cases"),
    complaints_dir: str = os.path.join(DATA_ROOT, "complaints"),
    fpa_dir: str = os.path.join(DATA_ROOT, "first_pass_accuracy"),
    checker_dir: str = os.path.join(DATA_ROOT, "checker_accuracy"),
    surveys_dir: str = os.path.join(DATA_ROOT, "surveys"),
) -> Dict[str, pd.DataFrame]:
    cases = load_cases(cases_dir)
    complaints = load_complaints(complaints_dir)
    fpa = load_fpa(fpa_dir)
    checker = load_checker_accuracy(checker_dir)
    surveys = load_surveys(surveys_dir)

    return {
        "cases": cases,
        "complaints": complaints,
        "fpa": fpa,
        "checker": checker,
        "surveys": surveys,
    }
