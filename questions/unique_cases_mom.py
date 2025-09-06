# questions/unique_cases_mom.py
from __future__ import annotations

import pandas as pd
import streamlit as st


def _coerce_month(s: str) -> pd.Timestamp:
    return pd.to_datetime(s).to_period("M").to_timestamp()


def run(store, params: dict, user_text: str):
    cases = store["cases"]
    if cases.empty:
        st.info("No cases available.")
        return

    start_m = _coerce_month(params["start_month"])
    end_m = _coerce_month(params["end_month"])
    portfolio = params.get("portfolio")  # optional

    df = cases[(cases["month"] >= start_m) & (cases["month"] <= end_m)].copy()
    if portfolio:
        df = df[df["portfolio"].astype(str).str.strip().str.casefold() == str(portfolio).strip().casefold()]

    res = (
        df.groupby("month", as_index=False)
        .agg(unique_cases=("id", "nunique"))
        .assign(_month=lambda x: x["month"].dt.strftime("%b %y"))
        .loc[:, ["_month", "unique_cases"]]
    )

    if res.empty:
        st.info("No cases returned for the current filters.")
        return

    st.subheader("Unique cases (MoM)")
    st.dataframe(res, use_container_width=True, hide_index=True)
