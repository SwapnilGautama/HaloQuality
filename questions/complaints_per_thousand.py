# questions/complaints_per_thousand.py
from __future__ import annotations

import pandas as pd
import streamlit as st


def _coerce_month(s: str) -> pd.Timestamp:
    return pd.to_datetime(s).to_period("M").to_timestamp()


def run(store, params: dict, user_text: str):
    """
    Output a month-by-process table for the selected portfolio & month range.
    IMPORTANT: denominator joins by PORTFOLIO ONLY (not by process).
    """
    cases = store["cases"]
    complaints = store["complaints"]

    if cases.empty or complaints.empty:
        st.info("No overlapping data for cases and complaints.")
        return

    portfolio = params.get("portfolio", "London")
    start_m = _coerce_month(params["start_month"])
    end_m = _coerce_month(params["end_month"])

    # Filter by portfolio + month windows
    c_cases = cases[(cases["portfolio"].astype(str).str.strip().str.casefold() == str(portfolio).strip().casefold())]
    c_cases = c_cases[(c_cases["month"] >= start_m) & (c_cases["month"] <= end_m)]

    c_comp = complaints[(complaints["portfolio"].astype(str).str.strip().str.casefold() == str(portfolio).strip().casefold())]
    c_comp = c_comp[(c_comp["month"] >= start_m) & (c_comp["month"] <= end_m)]

    if c_cases.empty:
        st.info("No cases after applying the selected filters/date window.")
        return

    # Denominator: cases per month for the portfolio (NOT split by process)
    denom = (
        c_cases.groupby("month", as_index=False)
        .agg(cases=("id", "nunique"))
    )

    # Numerator: complaints per month * process (use Parent Case Type if present)
    c_comp = c_comp.copy()
    if "process" not in c_comp.columns or c_comp["process"].isna().all():
        c_comp["process"] = "Unknown"
    c_comp["process"] = c_comp["process"].fillna("Unknown").astype(str)

    num = (
        c_comp.groupby(["month", "process"], as_index=False)
        .size()
        .rename(columns={"size": "complaints"})
    )

    # Join by month only, then compute per_1000
    out = num.merge(denom, on="month", how="left")
    out["per_1000"] = (out["complaints"] / out["cases"].replace(0, pd.NA)) * 1000
    out["per_1000"] = out["per_1000"].fillna(0).round(2)

    # Friendly month label
    out = out.sort_values(["month", "process"]).reset_index(drop=True)
    out.insert(0, "month_label", out["month"].dt.strftime("%b %y"))

    st.dataframe(
        out[["month_label", "process", "cases", "complaints", "per_1000"]],
        use_container_width=True,
        hide_index=True,
    )

    # Also show a quick pivot (optional)
    with st.expander("Pivot (process x month)", expanded=False):
        pvt = out.pivot_table(index="process", columns="month_label", values="per_1000", aggfunc="first").fillna(0)
        st.dataframe(pvt, use_container_width=True)
