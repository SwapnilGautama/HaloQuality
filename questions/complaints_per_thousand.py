import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
from typing import Any, Dict, Optional, Iterable


# ----------------------------- helpers ---------------------------------

DATE_CANDIDATES = [
    "Report Date", "Create Date", "Created", "Created Date", "create_date",
    "report_date", "create_dt", "Created_On"
]

PROCESS_CANDIDATES = [
    "Process Name", "Process", "Process_Name", "ProcessName", "Parent Case Type"
]

PORTFOLIO_CANDIDATES = [
    "Portfolio_std", "Portfolio", "portfolio", "Portfolio Name"
]

CASE_ID_CANDIDATES = [
    "Case ID", "CaseID", "case_id", "Unique Identifier (NINO Encrypted)"
]


def _first_existing(df: pd.DataFrame, options: Iterable[str]) -> Optional[str]:
    for c in options:
        if c in df.columns:
            return c
    return None


def _ensure_month(df: pd.DataFrame, prefer: Optional[str] = None) -> pd.Series:
    """
    Return a 'month' Series (Timestamp normalized to month start),
    deriving it from one of the known date columns.
    """
    col = prefer if (prefer and prefer in df.columns) else _first_existing(df, DATE_CANDIDATES)
    if col is None:
        raise KeyError("No usable date column found (looked for: %s)" % ", ".join(DATE_CANDIDATES))
    s = pd.to_datetime(df[col], errors="coerce")
    return s.dt.to_period("M").dt.to_timestamp()


def _get_df(store: Any, key: str) -> pd.DataFrame:
    if isinstance(store, dict):
        return store[key]
    if hasattr(store, key):
        return getattr(store, key)
    # last resort – some stores keep everything under .tables
    if hasattr(store, "tables"):
        return store.tables[key]
    raise KeyError(f"Could not find '{key}' in data store")


def _coerce_list(x):
    if x is None:
        return None
    if isinstance(x, (list, tuple, set)):
        return list(x)
    return [x]


def _month_from_arg(x: Any) -> Optional[pd.Timestamp]:
    if x in (None, "", np.nan):
        return None
    try:
        # allow "Jun 2025" / "2025-06"
        dt = pd.to_datetime(str(x), errors="coerce")
        if pd.isna(dt):
            return None
        return dt.to_period("M").to_timestamp()
    except Exception:
        return None


# ----------------------------- main -------------------------------------

def run(store: Any, params: Optional[Dict[str, Any]] = None, **kwargs):
    """
    Compute Complaints per 1,000 cases by month/process (optionally filtered by portfolio/process/month range).

    Accepts either a legacy 'params' dict or named kwargs. Supported keys:
      - portfolio
      - process or process_name
      - start, end (month endpoints, e.g., '2025-06' or 'Jun 2025')
    """
    # unify arguments
    p = {}
    if params:
        p.update(params)
    p.update(kwargs)

    portfolio = p.get("portfolio")
    process = p.get("process") or p.get("process_name")
    start_m = _month_from_arg(p.get("start") or p.get("from"))
    end_m   = _month_from_arg(p.get("end") or p.get("to"))

    # Load data
    cases = _get_df(store, "cases").copy()
    complaints = _get_df(store, "complaints").copy()

    # Identify columns
    case_id_col = _first_existing(cases, CASE_ID_CANDIDATES)
    if case_id_col is None:
        raise KeyError("Cases: missing Case ID column")

    case_proc_col = _first_existing(cases, PROCESS_CANDIDATES)
    if case_proc_col is None:
        raise KeyError("Cases: missing Process column")

    case_port_col = _first_existing(cases, PORTFOLIO_CANDIDATES)
    if case_port_col is None:
        # keep going without portfolio if truly absent
        case_port_col = None

    comp_proc_col = _first_existing(complaints, PROCESS_CANDIDATES)
    if comp_proc_col is None:
        # Some complaint files only have Parent Case Type; we already searched for it.
        # If nothing matched, give a clear error.
        raise KeyError("Complaints: missing Process / Parent Case Type column")

    comp_port_col = _first_existing(complaints, PORTFOLIO_CANDIDATES)
    if comp_port_col is None:
        comp_port_col = None

    # Months
    cases["month"] = _ensure_month(cases)
    complaints["month"] = _ensure_month(complaints)

    # Optional filters
    if portfolio and case_port_col:
        cases = cases[cases[case_port_col].astype(str).str.lower() == str(portfolio).lower()]
    if portfolio and comp_port_col:
        complaints = complaints[complaints[comp_port_col].astype(str).str.lower() == str(portfolio).lower()]

    if process:
        cases = cases[cases[case_proc_col].astype(str).str.lower() == str(process).lower()]
        complaints = complaints[complaints[comp_proc_col].astype(str).str.lower() == str(process).lower()]

    if start_m is not None:
        cases = cases[cases["month"] >= start_m]
        complaints = complaints[complaints["month"] >= start_m]
    if end_m is not None:
        cases = cases[cases["month"] <= end_m]
        complaints = complaints[complaints["month"] <= end_m]

    # Group and aggregate
    gb_keys = ["month", case_proc_col]
    if case_port_col:
        gb_keys.append(case_port_col)

    cases_agg = (
        cases.groupby(gb_keys)[case_id_col].nunique()
        .rename("unique_cases")
        .reset_index()
    )

    comp_keys = ["month", comp_proc_col]
    if comp_port_col:
        comp_keys.append(comp_port_col)

    comp_agg = (
        complaints.groupby(comp_keys).size()
        .rename("complaints")
        .reset_index()
    )

    # Normalize column names to join cleanly
    cases_agg = cases_agg.rename(columns={case_proc_col: "Process", case_port_col or "": "Portfolio"})
    comp_agg  = comp_agg.rename(columns={comp_proc_col: "Process", comp_port_col or "": "Portfolio"})

    # Some datasets won’t have portfolio; align columns for merge
    if "Portfolio" not in cases_agg.columns:
        cases_agg["Portfolio"] = "ALL"
    if "Portfolio" not in comp_agg.columns:
        comp_agg["Portfolio"] = "ALL"

    # Join on month + process + portfolio
    merged = pd.merge(
        cases_agg, comp_agg,
        on=["month", "Process", "Portfolio"],
        how="outer"
    ).fillna(0)

    if merged.empty or (merged["unique_cases"] == 0).all():
        st.info("No overlapping data after filters (or zero cases). Try widening filters.")
        return

    merged["complaints_per_1000"] = (merged["complaints"] * 1000.0) / merged["unique_cases"]
    merged = merged.sort_values(["month", "Process", "Portfolio"])

    # --------- UI ----------
    title_bits = ["Complaints per 1,000 cases"]
    if portfolio:
        title_bits.append(f"— Portfolio: {portfolio}")
    if process:
        title_bits.append(f"— Process: {process}")
    if start_m or end_m:
        sm = start_m.strftime("%b %Y") if start_m is not None else "…"
        em = end_m.strftime("%b %Y") if end_m is not None else "…"
        title_bits.append(f"— {sm} to {em}")

    st.subheader(" ".join(title_bits))

    # Line/Bar by month
    chart_df = merged.groupby("month", as_index=False)["complaints_per_1000"].mean()
    fig = px.bar(chart_df, x="month", y="complaints_per_1000",
                 title="Average complaints per 1,000 cases (all processes in selection)")
    fig.update_yaxes(title="Complaints / 1000 cases")
    fig.update_xaxes(title="Month")
    st.plotly_chart(fig, use_container_width=True)

    # Detail table (by process/portfolio/month)
    pretty = merged.copy()
    pretty["month"] = pretty["month"].dt.strftime("%b %Y")
    pretty = pretty[["month", "Portfolio", "Process", "unique_cases", "complaints", "complaints_per_1000"]]
    pretty = pretty.rename(columns={
        "month": "Month",
        "unique_cases": "Unique_Cases",
        "complaints": "Complaints",
        "complaints_per_1000": "Complaints_per_1000"
    })
    st.dataframe(pretty, use_container_width=True)
