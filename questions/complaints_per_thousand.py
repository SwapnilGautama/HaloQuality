# questions/complaints_per_thousand.py
# Complaints per 1,000 cases — by process (with optional portfolio & month filters)

from __future__ import annotations
import re
from typing import Optional, Tuple, List

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px


TITLE = "Complaints per 1,000 cases"


# ----------------------------- column candidates ------------------------------

DATE_CANDIDATES_CASES = [
    "Create Date", "Created", "Created Date", "create_date",
    "create_dt", "Created_On", "Create_Date"
]
DATE_CANDIDATES_COMPLAINTS = [
    "Report Date", "Report_Date", "Reported On", "Date", "Created", "Create Date"
]

PROCESS_CANDIDATES_CASES = [
    "Process Name", "Process", "Process_Name", "ProcessName"
]
PROCESS_CANDIDATES_COMPLAINTS = [
    "Parent Case Type", "Parent_Case_Type", "Process Name", "Process"
]

PORTFOLIO_CANDIDATES = [
    "Portfolio_std", "Portfolio", "portfolio", "Portfolio Name", "Portfolio_Name"
]

CASE_ID_CANDIDATES = [
    "Case ID", "CaseID", "case_id", "Case Number", "Case No", "Case Ref",
    "Case Reference", "Case Reference ID", "CaseNumber",
    "Unique Identifier (NINO Encrypted)"
]


# ------------------------------ helpers ---------------------------------------

def _first_existing(df: pd.DataFrame, opts: List[str]) -> Optional[str]:
    for c in opts:
        if c in df.columns:
            return c
    return None


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(s).lower()).strip("_")


def _find_case_id_col(df: pd.DataFrame) -> Optional[str]:
    # 1) exacts
    c = _first_existing(df, CASE_ID_CANDIDATES)
    if c:
        return c
    # 2) heuristic
    for col in df.columns:
        n = _norm(col)
        if "case" in n and any(k in n for k in ["id", "no", "num", "number", "ref", "reference"]):
            return col
    # 3) fallback: high-cardinality looks like an id
    try:
        nun = df.nunique(dropna=True).sort_values(ascending=False)
        for col in nun.index:
            if nun[col] >= 0.5 * len(df):
                return col
    except Exception:
        pass
    return None


def _find_col(df: pd.DataFrame, primary: List[str], secondary: Optional[List[str]] = None) -> Optional[str]:
    c = _first_existing(df, primary)
    if c:
        return c
    if secondary:
        return _first_existing(df, secondary)
    return None


def _parse_month_token(x) -> Optional[pd.Timestamp]:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    try:
        # accept “Jun 2025”, “2025-06”, “2025/06/01”, etc.
        ts = pd.to_datetime(str(x), errors="coerce", dayfirst=False)
        if pd.isna(ts):
            return None
        # normalize to first of month
        ts = ts.to_period("M").to_timestamp()
        return ts
    except Exception:
        return None


def _ensure_month(df: pd.DataFrame, date_candidates: List[str], out_col: str = "month") -> Tuple[pd.DataFrame, str]:
    c = _first_existing(df, date_candidates)
    if c is None:
        # last resort: try any column that parses to datetime for most rows
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                c = col
                break
            try:
                test = pd.to_datetime(df[col], errors="coerce")
                if test.notna().mean() > 0.7:
                    c = col
                    break
            except Exception:
                continue
    if c is None:
        raise KeyError("No date column found to create a month field.")

    if not pd.api.types.is_datetime64_any_dtype(df[c]):
        df = df.copy()
        df[c] = pd.to_datetime(df[c], errors="coerce")

    df[out_col] = df[c].dt.to_period("M").dt.to_timestamp()
    return df, out_col


def _caption_from_filters(portfolio: Optional[str], start_m: Optional[pd.Timestamp], end_m: Optional[pd.Timestamp]) -> str:
    parts = []
    if portfolio:
        parts.append(f"Portfolio: **{portfolio}**")
    if start_m or end_m:
        lo = start_m.strftime("%b %Y") if start_m is not None else "…"
        hi = end_m.strftime("%b %Y")   if end_m is not None else "…"
        parts.append(f"Months: **{lo} → {hi}**")
    return " · ".join(parts) if parts else "All data"


# ------------------------------ main question ---------------------------------

def run(store, params=None, query: str | None = None, user_text: str | None = None):
    """
    Render 'Complaints per 1,000 cases' by process, with optional filters.
    Expected store attributes:
      - store.cases (pd.DataFrame)
      - store.complaints (pd.DataFrame)
    Accepted params (all optional):
      - portfolio / portfolio_std (str)
      - month_from / start_month (str-like)
      - month_to / end_month (str-like)
    """
    st.subheader(TITLE)

    params = params or {}

    cases = getattr(store, "cases", None)
    complaints = getattr(store, "complaints", None)

    if cases is None or complaints is None or len(cases) == 0 or len(complaints) == 0:
        st.warning("No overlapping data for cases and complaints.")
        return

    # --- canonical columns
    case_id_col = _find_case_id_col(cases)
    if case_id_col is None:
        raise KeyError("Cases: missing Case ID column")

    proc_cases = _find_col(cases, PROCESS_CANDIDATES_CASES)
    if proc_cases is None:
        raise KeyError("Cases: missing process column")

    proc_comp = _find_col(complaints, PROCESS_CANDIDATES_COMPLAINTS, PROCESS_CANDIDATES_CASES)
    if proc_comp is None:
        raise KeyError("Complaints: missing process/parent case type column")

    portfolio_col_cases = _find_col(cases, PORTFOLIO_CANDIDATES)
    portfolio_col_comp  = _find_col(complaints, PORTFOLIO_CANDIDATES)

    # --- month fields
    cases, cases_month_col = _ensure_month(cases, DATE_CANDIDATES_CASES, "month")
    complaints, comp_month_col = _ensure_month(complaints, DATE_CANDIDATES_COMPLAINTS, "month")

    # --- filters from params
    portfolio = params.get("portfolio") or params.get("portfolio_std") or params.get("Portfolio") or params.get("Portfolio_std")
    start_m = _parse_month_token(params.get("month_from") or params.get("start_month") or params.get("from_month"))
    end_m   = _parse_month_token(params.get("month_to")   or params.get("end_month")   or params.get("to_month"))

    # show filters caption
    st.caption(_caption_from_filters(portfolio, start_m, end_m))

    # apply month filters
    if start_m is not None:
        cases = cases.loc[cases[cases_month_col] >= start_m]
        complaints = complaints.loc[complaints[comp_month_col] >= start_m]
    if end_m is not None:
        cases = cases.loc[cases[cases_month_col] <= end_m]
        complaints = complaints.loc[complaints[comp_month_col] <= end_m]

    # portfolio filter
    if portfolio and portfolio_col_cases:
        cases = cases.loc[cases[portfolio_col_cases].astype(str).str.casefold() == str(portfolio).casefold()]
    if portfolio and portfolio_col_comp:
        complaints = complaints.loc[complaints[portfolio_col_comp].astype(str).str.casefold() == str(portfolio).casefold()]

    if cases.empty or complaints.empty:
        st.info("No data after applying filters.")
        return

    # --- aggregate
    # unique cases by process
    cases_agg = (
        cases
        .dropna(subset=[proc_cases, case_id_col])
        .groupby(proc_cases, dropna=False)[case_id_col]
        .nunique()
        .rename("Unique_Cases")
        .reset_index()
    )

    # complaints by process
    comp_agg = (
        complaints
        .dropna(subset=[proc_comp])
        .groupby(proc_comp, dropna=False)
        .size()
        .rename("Complaints")
        .reset_index()
    )
    comp_agg = comp_agg.rename(columns={proc_comp: proc_cases})  # align key

    # join & compute rate
    df = pd.merge(comp_agg, cases_agg, on=proc_cases, how="outer")
    df["Complaints"] = df["Complaints"].fillna(0).astype(int)
    df["Unique_Cases"] = df["Unique_Cases"].fillna(0).astype(int)
    df["Complaints_per_1000"] = np.where(
        df["Unique_Cases"] > 0, df["Complaints"] / df["Unique_Cases"] * 1000.0, np.nan
    )
    df = df.sort_values("Complaints_per_1000", ascending=False)

    # friendly names
    df = df.rename(columns={proc_cases: "Process"})

    # --- render
    if df["Complaints_per_1000"].notna().any():
        fig = px.bar(
            df.sort_values("Complaints_per_1000", ascending=True),
            x="Complaints_per_1000",
            y="Process",
            orientation="h",
            title=None,
        )
        fig.update_layout(
            height=max(360, 28 * len(df)),
            xaxis_title="Complaints per 1,000 cases",
            yaxis_title="Process",
            margin=dict(l=10, r=10, t=10, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No processes have both complaints and cases in the selected window.")

    # show table
    pretty = df.copy()
    pretty["Complaints_per_1000"] = pretty["Complaints_per_1000"].round(2)
    st.dataframe(pretty, use_container_width=True)
