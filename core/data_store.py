# data_store.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Optional, List

import pandas as pd
import streamlit as st


DATA_DIRS = [
    Path.cwd() / "data",
    Path.cwd() / "core" / "data",
]


# ---------- small utils ----------
def _first_existing(paths: Iterable[Path]) -> Optional[Path]:
    for p in paths:
        if p.exists():
            return p
    return None


def _read_any(f: Path) -> pd.DataFrame:
    suffix = f.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(f)
    if suffix == ".csv":
        return pd.read_csv(f)
    # default Excel: Sheet 1 (index 0) as requested
    return pd.read_excel(f, sheet_name=0)


def _load_folder(sub: str) -> pd.DataFrame:
    base = _first_existing([d / sub for d in DATA_DIRS])
    if not base or not base.exists():
        return pd.DataFrame()
    frames: List[pd.DataFrame] = []
    for f in sorted(base.glob("**/*")):
        if f.is_file() and f.suffix.lower() in {".xlsx", ".xls", ".csv", ".parquet"}:
            try:
                frames.append(_read_any(f))
            except Exception:
                # ignore bad files; keep loading others
                pass
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]


def _norm_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    for c in df.columns:
        if pd.api.types.is_string_dtype(df[c]):
            df[c] = df[c].astype(str).str.strip()
    return df


def _choose_first_existing(cols: Iterable[str], candidates: Iterable[str]) -> Optional[str]:
    """Return the first candidate present in cols (case-insensitive, normalized)."""
    s = {c.lower(): c for c in cols}
    for c in candidates:
        if c in s:
            return s[c]
    return None


# ---------- canonicalizers ----------
def _rename_cases(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = _norm_cols(df)

    # id
    if "id" not in df.columns:
        for c in ["case id", "unique identifier", "unique identi", "caseid", "id"]:
            if c in df.columns:
                df = df.rename(columns={c: "id"})
                break

    # process
    if "process" not in df.columns:
        for c in ["process name", "process"]:
            if c in df.columns:
                df = df.rename(columns={c: "process"})
                break

    # portfolio
    if "portfolio" not in df.columns:
        for c in ["portfolio"]:
            if c in df.columns:
                df = df.rename(columns={c: "portfolio"})
                break
    if "portfolio" in df.columns:
        df["portfolio"] = df["portfolio"].astype(str).str.strip().str.title()

    # date
    date_candidates = [
        "create date",
        "report date",
        "created date",
        "start date",
        "report_date",
        "createddate",
        "create_date",
        "startdate",
    ]
    chosen = _choose_first_existing(df.columns, date_candidates)
    if not chosen:
        others = [c for c in df.columns if "date" in c and c not in {"_month", "month"}]
        chosen = others[0] if others else None

    if chosen:
        df["date"] = pd.to_datetime(df[chosen], errors="coerce", dayfirst=True)
        df["_month"] = df["date"].dt.to_period("M")
    else:
        df["_month"] = pd.NaT  # nothing we can do

    return df


def _rename_complaints(df: pd.DataFrame, assume_year_for_complaints: int) -> pd.DataFrame:
    if df.empty:
        return df
    df = _norm_cols(df)

    # portfolio
    if "portfolio" not in df.columns:
        if "portfolio" in df.columns:
            df = df.rename(columns={"portfolio": "portfolio"})
    if "portfolio" in df.columns:
        df["portfolio"] = df["portfolio"].astype(str).str.strip().str.title()

    # process if present
    for c in ["parent case type", "process"]:
        if c in df.columns:
            df = df.rename(columns={c: "process"})
            break

    # date (explicit first; else Month text + assumed year)
    comp_date_col = _choose_first_existing(
        df.columns,
        ["date complaint received - dd/mm/yy", "date complaint received", "complaint date"],
    )

    if comp_date_col:
        df["date"] = pd.to_datetime(df[comp_date_col], errors="coerce", dayfirst=True)
        df["_month"] = df["date"].dt.to_period("M")
    else:
        month_text_col = _choose_first_existing(df.columns, ["month"])
        if month_text_col:
            m = (
                df[month_text_col]
                .astype(str)
                .str.strip()
                .str.capitalize()
                .str.slice(0, 3)
            )
            df["_month"] = pd.to_datetime(
                m + f" {assume_year_for_complaints}", format="%b %Y", errors="coerce"
            ).dt.to_period("M")
            df["date"] = df["_month"].dt.to_timestamp()
        else:
            df["_month"] = pd.NaT
            df["date"] = pd.NaT

    # case id linkage if present
    for c in ["original process affected case id", "original process affected caseid", "case id", "caseid"]:
        if c in df.columns:
            df = df.rename(columns={c: "case_id"})
            break

    return df


def _rename_surveys(df: pd.DataFrame) -> pd.DataFrame:
    """Light normalization so the question can work consistently."""
    if df.empty:
        return df
    df = _norm_cols(df)

    # portfolio
    if "portfolio" in df.columns:
        df["portfolio"] = df["portfolio"].astype(str).str.strip().str.title()

    # Month_received -> _month (leave full parsing to the question for robustness)
    # we still try a best-effort here for quick checks
    month_col = None
    for c in ["month_received", "month received", "received_month", "month"]:
        if c in df.columns:
            month_col = c
            break
    if month_col:
        temp = pd.to_datetime(df[month_col], errors="coerce", dayfirst=True, infer_datetime_format=True)
        df["_month"] = temp.dt.to_period("M")
        df["date"] = temp
    else:
        df["_month"] = pd.NaT
        df["date"] = pd.NaT

    return df


# ---------- public loader ----------
@st.cache_data(show_spinner="Reading Excel / parquet sources", ttl=3600)
def load_store(assume_year_for_complaints: int = 2025) -> Dict[str, Any]:
    cases = _load_folder("cases")
    complaints = _load_folder("complaints")
    fpa = _load_folder("fpa")
    surveys = _load_folder("surveys")  # << NEW

    cases = _rename_cases(cases)
    complaints = _rename_complaints(complaints, assume_year_for_complaints)
    surveys = _rename_surveys(surveys)  # << NEW

    return {
        "cases": cases,
        "complaints": complaints,
        "fpa": fpa,
        "surveys": surveys,          # << NEW
        "cases_rows": len(cases),
        "complaints_rows": len(complaints),
        "fpa_rows": len(fpa),
        "surveys_rows": len(surveys) # << NEW
    }
