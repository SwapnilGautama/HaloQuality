# core/data_store.py
# -----------------------------------------------------------------------------
# Centralized data loader + normalizer for Halo Quality
# - Loads: cases, complaints, fpa (via your existing loader_* modules)
# - Normalizes to canonical columns used by questions:
#       month_key  -> pandas.Period[M] for joins/filters
#       month      -> 'Jun 2025' label for display
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
from .loader_cases import load_cases            # expected to return a DataFrame
from .loader_complaints import load_complaints  # expected to return a DataFrame
try:
    from .loader_fpa import load_fpa            # optional
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
CANON_MONTH_LABEL  = "month"            # 'Jun 2025' (display only)
CANON_PORT         = "Portfolio_std"    # standardized portfolio
CANON_CASE_ID      = "Case ID"          # unique case identifier

# export for other modules (e.g., questions/_utils)
__all__ = [
    "load_store",
    "normalize_cases", "normalize_complaints", "normalize_fpa",
    "CANON_MONTH", "CANON_MONTH_LABEL", "CANON_PORT", "CANON_CASE_ID",
]

# -----------------------------------------------------------------------------


def _std_text(x):
    if pd.isna(x):
        return x
    x = str(x).strip()
    x = re.sub(r"\s+", " ", x)
    return x.title()


def _find_first(df: pd.DataFrame, candidates: list[str] ) -> str | None:
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
        # allow downstream to warn; don't raise here
        return df

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    # Period month key and user-friendly label
    df[CANON_MONTH] = df[date_col].dt.to_period("M")
    df[CANON_MONTH_LABEL] = df[CANON_MONTH].astype(str).map(
        lambda s: pd.Period(s, "M").strftime("%b %Y")
    )
    return df


def _ensure_portfolio(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure we have a standardized Portfolio_std column for slicing/joins.
    """
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
    """
    Make the cases table schema consistent for questions:
      - Case ID present
      - month_key / month from Create Date
      - Portfolio_std standardized
    """
    if df is None or df.empty:
        return df

    # Case ID
    cid = _find_first(df, [CANON_CASE_ID, "CaseID", "Case Id", "case_id"])
    if cid and cid != CANON_CASE_ID:
        df = df.rename(columns={cid: CANON_CASE_ID})

    # Month from "Create Date" (preferred) or close variants
    df = _ensure_month_cols(df, ["Create Date", "Create_Date", "Created", "Created On"])

    # Portfolio
    df = _ensure_portfolio(df)

    return df


def normalize_complaints(df: pd.DataFrame) -> pd.DataFrame:
    """
    Make the complaints table schema consistent for questions:
      - month_key / month from Report Date (preferred)
      - Portfolio_std standardized
      - (RCA1/2 left as-is; labeller may add)
    """
    if df is None or df.empty:
        return df

    # Month from "Report Date" (preferred) or close variants
    df = _ensure_month_cols(df, ["Report Date", "Report_Date", "Date", "Reported On"])

    # Portfolio
    df = _ensure_portfolio(df)

    return df


def normalize_fpa(df: pd.DataFrame) -> pd.DataFrame:
    """
    Make the FPA table schema consistent for questions:
      - month_key / month from review/completion date
      - Portfolio_std standardized (if present)
    """
    if df is None or df.empty:
        return df

    # Accept a wider set of candidates we saw in the wild
    df = _ensure_month_cols(
        df,
        [
            "Review Date", "Reviewed Date", "Reviewed On", "Sampling Date",
            "Completed On", "Completed Date", "Report Date", "Date"
        ],
    )
    df = _ensure_portfolio(df)
    return df


# ----------------------------- Main loader -----------------------------------

@st.cache_data(show_spinner=False)
def load_store(cache_bust: str = "v1") -> dict:
    """
    Loads all datasets, applies normalization, runs optional labellers,
    and returns a dict-like store used by the questions.

    Returns
    -------
    store : dict
        {
          "cases": pd.DataFrame,
          "complaints": pd.DataFrame,
          "fpa": pd.DataFrame | None,
          "info": {
              "cases_rows": int,
              "complaints_rows": int,
              "fpa_rows": int,
              "latest": {
                  "cases": "Jun 2025" | None,
                  "complaints": "Jun 2025" | None,
                  "fpa": "Aug 2025" | None,
              }
          }
        }
    """
    # 1) Load raw
    cases = load_cases()
    complaints = load_complaints()
    fpa = load_fpa() if callable(load_fpa) else None

    # 2) Normalize
    cases = normalize_cases(cases)
    complaints = normalize_complaints(complaints)
    if fpa is not None and not fpa.empty:
        fpa = normalize_fpa(fpa)

    # 3) Optional labellers
    if callable(label_complaints_rca):
        try:
            complaints = label_complaints_rca(complaints)
        except Exception as e:
            # Non-fatal; questions that need RCA will explain if missing
            st.warning(f"RCA labeller failed: {e}")

    if fpa is not None and callable(label_fpa_comments):
        try:
            fpa = label_fpa_comments(fpa)
        except Exception as e:
            st.warning(f"FPA labeller failed: {e}")

    # 4) Small info block for the sidebar
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

    # 5) Return the store
    store = {
        "cases": cases,
        "complaints": complaints,
        "fpa": fpa,
        "info": info,
    }
    return store
