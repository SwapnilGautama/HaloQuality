# questions/rca1_portfolio_process.py
from __future__ import annotations

import pandas as pd
import streamlit as st


def _coerce_month(s: str) -> pd.Timestamp:
    return pd.to_datetime(s).to_period("M").to_timestamp()


def run(store, params: dict, user_text: str):
    """
    Lightweight RCA1 view. If an 'rca1' (or similar) column exists in complaints,
    we show counts by portfolio x process for the last N months; otherwise we show
    complaints grouped by Parent Case Type as a proxy.
    """
    complaints = store["complaints"]
    if complaints.empty:
        st.info("No complaints available.")
        return

    # Default is "last 3 months" if not provided
    start_m = _coerce_month(params.get("start_month", pd.Timestamp.today()))
    end_m = _coerce_month(params.get("end_month", pd.Timestamp.today()))
    portfolio = params.get("portfolio")

    df = complaints[(complaints["month"] >= start_m) & (complaints["month"] <= end_m)].copy()
    if portfolio:
        df = df[df["portfolio"].astype(str).str.strip().str.casefold() == str(portfolio).strip().casefold()]

    use_col = None
    for cand in ["rca1", "RCA1", "rca_1", "root_cause_1"]:
        if cand in df.columns:
            use_col = cand
            break

    if use_col is None:
        # fallback to Parent Case Type
        if "process" not in df.columns:
            st.info("No RCA1 column and no Parent Case Type to proxy.")
            return
        use_col = "process"

    out = (
        df.groupby(["portfolio", "process"] if use_col == "rca1" else ["portfolio", use_col], as_index=False)
        .size()
        .rename(columns={"size": "complaints"})
        .sort_values("complaints", ascending=False)
    )

    if out.empty:
        st.info("No complaints for the selected filters.")
        return

    st.subheader("RCA1 by Portfolio × Process — last 3 months")
    st.dataframe(out, use_container_width=True, hide_index=True)
