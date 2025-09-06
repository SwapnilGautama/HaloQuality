# questions/unique_cases_mom.py
from __future__ import annotations
import re
from typing import Iterable, Optional
import pandas as pd
import streamlit as st


def _canon(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", s.lower())


def _pick_col(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    canon_cols = {_canon(c): c for c in df.columns}
    for want in candidates:
        c = canon_cols.get(_canon(want))
        if c:
            return c
    return None


def _ensure_month_cols(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    dt = pd.to_datetime(df[date_col], errors="coerce", dayfirst=True)
    out = df.copy()
    out["month_dt"] = dt.dt.to_period("M").dt.to_timestamp()
    out["month_label"] = out["month_dt"].dt.strftime("%b %y")
    return out


def run(store: dict, params: dict, user_text: Optional[str] = None):
    """
    Monthly unique **cases** (nunique Case ID) with flexible column names.
    Accepts optional filters in `params`: start_month, end_month, portfolio, process.
    """
    if "cases" not in store or store["cases"] is None or len(store["cases"]) == 0:
        st.info("Cases data not loaded.")
        return

    df = store["cases"]

    case_id_col = _pick_col(df, ["Case ID", "CaseId", "Case Number", "Case_Number"])
    created_col = _pick_col(df, ["Create Date", "Created Date", "Creation Date", "Created_On", "CRTD_DT", "Created"])
    portfolio_col = _pick_col(df, ["Portfolio", "Portfolio_std", "Portfolio Name", "PortfolioName"])
    process_col = _pick_col(df, ["Process", "Process Name", "ProcessName", "Parent case type", "Parent_Case_Type"])

    if not case_id_col or not created_col:
        st.info("Missing 'Case ID' and/or a 'Create Date' column in cases.")
        return

    # Standardise month cols
    df = _ensure_month_cols(df, created_col)

    # Optional filters
    if portfolio_col and params.get("portfolio"):
        want = str(params["portfolio"]).lower().strip()
        df = df[df[portfolio_col].astype(str).str.lower().str.strip() == want]

    if process_col and params.get("process"):
        want = str(params["process"]).lower().strip()
        df = df[df[process_col].astype(str).str.lower().str.strip() == want]

    # Month window
    start = params.get("start_month")
    end = params.get("end_month")
    if start:
        try:
            start_dt = pd.to_datetime(start).to_period("M").to_timestamp()
            df = df[df["month_dt"] >= start_dt]
        except Exception:
            pass
    if end:
        try:
            end_dt = pd.to_datetime(end).to_period("M").to_timestamp()
            df = df[df["month_dt"] <= end_dt]
        except Exception:
            pass

    if df.empty:
        st.info("No cases after filters.")
        return

    # Monthly uniques
    summary = (
        df.groupby("month_dt")[case_id_col]
        .nunique(dropna=True)
        .rename("unique_cases")
        .reset_index()
        .sort_values("month_dt")
    )
    summary["month_label"] = summary["month_dt"].dt.strftime("%b %y")
    summary = summary[["month_label", "unique_cases"]]

    st.subheader("Unique cases (MoM)")
    st.dataframe(summary, use_container_width=True)

    try:
        import altair as alt
        chart = (
            alt.Chart(summary)
            .mark_line(point=True)
            .encode(
                x=alt.X("month_label:N", title="Month", sort=list(summary["month_label"])),
                y=alt.Y("unique_cases:Q", title="Unique cases"),
                tooltip=["month_label", "unique_cases"],
            )
        )
        st.altair_chart(chart, use_container_width=True)
    except Exception:
        pass
