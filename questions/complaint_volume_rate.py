# questions/complaint_volume_rate.py
import pandas as pd
import streamlit as st

def run(store, params, user_text=None):
    complaints = store.get("complaints")
    if complaints is None or complaints.empty:
        st.info("Complaints data not loaded.")
        return

    if "Month" not in complaints:
        st.error("Month column missing in complaints; please refresh loaders.")
        return

    complaints = complaints.copy()

    portfolio = (params or {}).get("portfolio")
    date_from = (params or {}).get("date_from")
    date_to   = (params or {}).get("date_to")

    # Optional portfolio filter
    p_col = None
    for c in ["Portfolio_std", "Portfolio", "portfolio_std", "portfolio"]:
        if c in complaints.columns:
            p_col = c; break
    if portfolio and p_col:
        complaints = complaints[complaints[p_col].astype(str).str.strip().str.lower()
                                == str(portfolio).strip().lower()]

    if date_from:
        complaints = complaints[complaints["Month"] >= pd.to_datetime(date_from)]
    if date_to:
        complaints = complaints[complaints["Month"] <= pd.to_datetime(date_to)]

    if complaints.empty:
        st.info("No complaints after filters.")
        return

    series = complaints.groupby("Month").size().rename("complaints").reset_index()
    series["MoM_%"] = series["complaints"].pct_change() * 100.0
    series["Month_label"] = series["Month"].dt.strftime("%b %y")

    st.subheader("Complaint volume (MoM)")
    st.dataframe(series[["Month_label","complaints","MoM_%"]], use_container_width=True)
