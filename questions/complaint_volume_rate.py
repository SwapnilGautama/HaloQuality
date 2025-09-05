# questions/complaint_volume_rate.py
import pandas as pd
import streamlit as st

def _first(df, names):
    for n in names:
        if n in df.columns:
            return n
    return None

def _month_from(df, names):
    col = _first(df, names)
    if col is None:
        raise ValueError(f"Missing any of {names}")
    return pd.to_datetime(df[col], errors="coerce").dt.to_period("M").dt.to_timestamp()

def _portfolio_col(df):
    return _first(df, ["Portfolio_std", "portfolio_std", "Portfolio", "portfolio"])

def _norm(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower()

def run(store, params, user_text=None):
    """
    Complaint volume (MoM) – counts complaints by month (Report_Date),
    with optional portfolio filter and date range.
    """
    complaints = store.get("complaints")
    if complaints is None or complaints.empty:
        st.info("Complaints data not loaded.")
        return

    complaints = complaints.copy()
    complaints["Month"] = _month_from(complaints, ["Report_Date", "Report Date", "ReportDate", "Report Dt"])
    p_col = _portfolio_col(complaints)

    portfolio = (params or {}).get("portfolio")
    date_from = (params or {}).get("date_from")
    date_to   = (params or {}).get("date_to")

    if portfolio and p_col is not None:
        complaints = complaints[_norm(complaints[p_col]).eq(_norm(pd.Series([portfolio])).iloc[0])]

    if date_from:
        complaints = complaints[complaints["Month"] >= pd.to_datetime(date_from)]
    if date_to:
        complaints = complaints[complaints["Month"] <= pd.to_datetime(date_to)]

    if complaints.empty:
        st.info("No complaints after filters.")
        return

    series = complaints.groupby("Month").size().rename("complaints").reset_index()
    series["MoM_%"] = series["complaints"].pct_change() * 100.0
    series["Month"] = series["Month"].dt.strftime("%Y-%m")

    st.subheader("Complaint volume (MoM)")
    st.dataframe(series, use_container_width=True)
