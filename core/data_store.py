# data_store.py
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import pandas as pd
import streamlit as st


DATA_DIRS = [
    Path.cwd() / "data",
    Path.cwd() / "core" / "data",
]


# ---------- utilities ----------
def _first_existing(paths: Iterable[Path]) -> Optional[Path]:
    for p in paths:
        if p.exists():
            return p
    return None


def _read_any(f: Path) -> pd.DataFrame:
    if f.suffix.lower() in [".parquet"]:
        return pd.read_parquet(f)
    if f.suffix.lower() in [".csv"]:
        return pd.read_csv(f)
    # excel
    return pd.read_excel(f)


def _load_folder(sub: str) -> pd.DataFrame:
    base = _first_existing([d / sub for d in DATA_DIRS])
    if not base or not base.exists():
        return pd.DataFrame()
    out = []
    for f in sorted(base.glob("**/*")):
        if f.is_file() and f.suffix.lower() in [".xlsx", ".xls", ".csv", ".parquet"]:
            try:
                out.append(_read_any(f))
            except Exception:
                # ignore bad files
                pass
    if not out:
        return pd.DataFrame()
    df = pd.concat(out, ignore_index=True) if len(out) > 1 else out[0]
    return df


def _norm_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    # strip whitespace in string columns
    for c in df.columns:
        if pd.api.types.is_string_dtype(df[c]):
            df[c] = df[c].astype(str).str.strip()
    return df


def _rename_cases(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = _norm_cols(df)
    ren = {
        "case id": "id",
        "unique identifier": "id",
        "unique identi": "id",
        "process name": "process",
        "process": "process",
        "portfolio": "portfolio",
        "create date": "date",
        "report_date": "date",
        "report date": "date",
        "created date": "date",
    }
    df = df.rename(columns={k: v for k, v in ren.items() if k in df.columns})
    # portfolio normalize
    if "portfolio" in df.columns:
        df["portfolio"] = df["portfolio"].astype(str).str.strip().str.title()
    # parse date
    if "date" not in df.columns:
        # try to find something that looks like date
        for c in df.columns:
            if "date" in c:
                df = df.rename(columns={c: "date"})
                break
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce", dayfirst=True)
        df["_month"] = df["date"].dt.to_period("M")
    else:
        df["_month"] = pd.NaT
    return df


def _rename_complaints(df: pd.DataFrame, assume_year_for_complaints: int) -> pd.DataFrame:
    if df.empty:
        return df
    df = _norm_cols(df)
    ren = {
        "original process affected case id": "case_id",
        "parent case type": "process",
        "portfolio": "portfolio",
        "date complaint received - dd/mm/yy": "date",
        "date complaint received": "date",
        "month": "month_text",
    }
    df = df.rename(columns={k: v for k, v in ren.items() if k in df.columns})

    if "portfolio" in df.columns:
        df["portfolio"] = df["portfolio"].astype(str).str.strip().str.title()

    # Date: if explicit date column -> parse; else if month_text, assume given year
    if "date" in df.columns and df["date"].notna().any():
        df["date"] = pd.to_datetime(df["date"], errors="coerce", dayfirst=True)
        df["_month"] = df["date"].dt.to_period("M")
    else:
        # Expect month_text like 'June' or 'Jun'
        if "month_text" in df.columns:
            # normalize text to 'Mon' abbreviations
            m = (
                df["month_text"]
                .astype(str)
                .str.strip()
                .str.capitalize()
                .str.slice(0, 3)
            )
            # build date = first of that month in assumed year
            df["_month"] = pd.to_datetime(
                m + f" {assume_year_for_complaints}", format="%b %Y", errors="coerce"
            ).dt.to_period("M")
            df["date"] = df["_month"].dt.to_timestamp()
        else:
            df["_month"] = pd.NaT
            df["date"] = pd.NaT

    # if process missing, keep but don't fail later
    return df


# ---------- public loader ----------
@st.cache_data(show_spinner="Reading Excel / parquet sources", ttl=3600)
def load_store(assume_year_for_complaints: int = 2025) -> Dict[str, Any]:
    cases = _load_folder("cases")
    complaints = _load_folder("complaints")
    fpa = _load_folder("fpa")

    cases = _rename_cases(cases)
    complaints = _rename_complaints(complaints, assume_year_for_complaints)

    # expose simple counts
    store: Dict[str, Any] = {
        "cases": cases,
        "complaints": complaints,
        "fpa": fpa,
        "cases_rows": len(cases),
        "complaints_rows": len(complaints),
        "fpa_rows": len(fpa),
    }
    return store
