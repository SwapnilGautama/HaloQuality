# core/data_store.py
# -----------------------------------------------------------------------------
# Centralized data loader + normalizer for Halo Quality
# - Loads: cases, complaints, fpa (via your existing loader_* modules)
# - Normalizes to canonical columns used by questions:
#       month_key  -> pandas.Period[M] (joins/filters)
#       month      -> 'Jun 2025' (display only)
#       Portfolio_std -> canonical portfolio
#       Case ID    -> unique case identifier in cases
# - Optionally runs RCA and FPA labellers if present
# - Returns a dict-like "store" with dataframes and a small info block
# -----------------------------------------------------------------------------

from __future__ import annotations

from pathlib import Path
import re
import pandas as pd
import streamlit as st

# --- import your existing loaders (already in this repo) ----------------------
from .loader_cases import load_cases              # -> DataFrame
from .loader_complaints import load_complaints    # -> DataFrame
try:
    from .loader_fpa import load_fpa              # optional
except Exception:
    load_fpa = None

# --- optional labellers (run if available) -----------------------------------
try:
    from .rca_labeller import label_complaints_rca
except Exception:
    label_complaints_rca = None

try:
    from .fpa_labeller import label_fpa_comments
except Exception:
    label_fpa_comments = None

# ---------- Canonical columns used everywhere by questions / joins -----------
CANON_MONTH        = "month_key"        # pandas.Period[M]
CANON_MONTH_LABEL  = "month"            # 'Jun 2025' (for display)
CANON_PORT         = "Portfolio_std"    # standardized portfolio
CANON_CASE_ID      = "Case ID"          # unique case identifier

__all__ = [
    "load_store",
    "normalize_cases", "normalize_complaints", "normalize_fpa",
    "CANON_MONTH", "CANON_MONTH_LABEL", "CANON_PORT", "CANON_CASE_ID",
]

# Default dataset folders (used when loaders require a path)
DEFAULT_DIRS = {
    "cases": Path("data/cases"),
    "complaints": Path("data/complaints"),
    "fpa": Path("data/first_pass_accuracy"),
}

# -----------------------------------------------------------------------------


def _std_text(x):
    if pd.isna(x):
        return x
    x = str(x).strip()
    x = re.sub(r"\s+", " ", x)
    return x.title()


def _find_first(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _ensure_month_cols(df: pd.DataFrame, date_candidates: list[str]) -> pd.DataFrame:
    """
    Ensure df has:
        - month_key  (Period[M]) for joins/filters
        - month      ('Jun 2025') for display
    """
    if df is None or df.empty:
        return df

    date_col = _find_first(df, date_candidates)
    if date_col is None:
        return df

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df[CANON_MONTH] = df[date_col].dt.to_period("M")
    df[CANON_MONTH_LABEL] = df[CANON_MONTH].astype(str).map(
        lambda s: pd.Period(s, "M").strftime("%b %Y")
    )
    return df


def _ensure_portfolio(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df

    if CANON_PORT not in df.columns:
        cand = _find_first(df, ["Portfolio_std", "Portfolio", "portfolio"])
        if cand:
            df[CANON_PORT] = df[cand].astype(str).map(_std_text)
        else:
            df[CANON_PORT] = pd.NA
    else:
        df[CANON_PORT] = df[CANON_PORT].astype(str).map(_std_text)
    return df


# ------------------------- Public normalizers --------------------------------

def normalize_cases(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df

    # Case ID
    cid = _find_first(df, [CANON_CASE_ID, "CaseID", "Case Id", "case_id"])
    if cid and cid != CANON_CASE_ID:
        df = df.rename(columns={cid: CANON_CASE_ID})

    # Month from Create Date (preferred)
    df = _ensure_month_cols(df, ["Create Date", "Create_Date", "Created", "Created On"])

    # Portfolio
    df = _ensure_portfolio(df)

    return df


def normalize_complaints(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df

    # Month from Report Date (preferred)
    df = _ensure_month_cols(df, ["Report Date", "Report_Date", "Date", "Reported On"])

    # Portfolio
    df = _ensure_portfolio(df)

    return df


def normalize_fpa(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df

    df = _ensure_month_cols(
        df,
        [
            "Review Date", "Reviewed Date", "Reviewed On", "Sampling Date",
            "Completed On", "Completed Date", "Report Date", "Date",
        ],
    )
    df = _ensure_portfolio(df)
    return df


# ----------------------------- Safe loader calls ------------------------------

def _safe_load(loader, default_path: Path):
    """
    Try loader() first; if it requires a 'path' positional arg, retry with default_path.
    """
    try:
        # attempt no-arg call (some loaders support this)
        return loader()
    except TypeError as e:
        # Typical error we saw: "missing 1 required positional argument: 'path'"
        msg = str(e)
        needs_path = "positional argument" in msg and "path" in msg
        if not needs_path:
            # re-raise any other TypeError
            raise
        # retry with default directory
        return loader(default_path)


# ----------------------------- Main loader -----------------------------------

@st.cache_data(show_spinner=False)
def load_store(cache_bust: str = "v1") -> dict:
    """
    Loads all datasets, applies normalization, runs optional labellers,
    and returns a dict-like store used by the questions.
    """
    # 1) Load raw (compat with both arg/no-arg loaders)
    cases = _safe_load(load_cases, DEFAULT_DIRS["cases"])
    complaints = _safe_load(load_complaints, DEFAULT_DIRS["complaints"])
    fpa = None
    if callable(load_fpa):
        try:
            fpa = _safe_load(load_fpa, DEFAULT_DIRS["fpa"])
        except Exception as e:
            st.warning(f"FPA loader failed: {e}")
            fpa = None

    # 2) Normalize
    cases = normalize_cases(cases)
    complaints = normalize_complaints(complaints)
    if fpa is not None and not fpa.empty:
        fpa = normalize_fpa(fpa)

    # 3) Optional labellers
    if callable(label_complaints_rca) and complaints is not None and not complaints.empty:
        try:
            complaints = label_complaints_rca(complaints)
        except Exception as e:
            st.warning(f"RCA labeller failed: {e}")

    if fpa is not None and callable(label_fpa_comments) and not fpa.empty:
        try:
            fpa = label_fpa_comments(fpa)
        except Exception as e:
            st.warning(f"FPA labeller failed: {e}")

    # 4) Info block for sidebar
    def _latest_label(df: pd.DataFrame) -> str | None:
        if df is None or df.empty or CANON_MONTH not in df.columns:
            return None
        p = df[CANON_MONTH].dropna().max()
        if pd.isna(p):
            return None
        return pd.Period(p, "M").strftime("%b %Y")

    info = {
        "cases_rows": 0 if cases is None else len(cases),
        "complaints_rows": 0 if complaints is None else len(complaints),
        "fpa_rows": 0 if (fpa is None or fpa.empty) else len(fpa),
        "latest": {
            "cases": _latest_label(cases),
            "complaints": _latest_label(complaints),
            "fpa": _latest_label(fpa) if fpa is not None else None,
        },
    }

    return {
        "cases": cases,
        "complaints": complaints,
        "fpa": fpa,
        "info": info,
    }
