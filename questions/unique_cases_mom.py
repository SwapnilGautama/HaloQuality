# questions/unique_cases_mom.py
from __future__ import annotations
import pandas as pd
import streamlit as st

from ._utils import pick_col, ensure_month_series, available_values, fuzzy_pick

CASE_ID_COLS = ["Case ID", "CaseID", "Case Number", "Case_No", "Case"]
PORT_COLS    = ["Portfolio", "Portfolio Name", "LOB", "Business Unit"]
DATE_COLS    = ["month_dt", "Create Date", "Create_Date", "Created On", "Created"]

def run(store,
        start_month: str | None = None,
        end_month: str | None = None,
        portfolio: str | None = None,
        user_text: str | None = None):

    cases = store["cases"].copy()

    case_id = pick_col(cases, CASE_ID_COLS)
    portc   = pick_col(cases, PORT_COLS)
    if case_id is None:
        st.warning("Cases: missing Case ID column.")
        return
    if portc is None:
        st.warning("Cases: missing Portfolio column.")
        return

    month_s = ensure_month_series(cases, DATE_COLS)
    cases["_month"] = month_s

    # time window
    if end_month:
        end = pd.Timestamp(pd.to_datetime(end_month).year, pd.to_datetime(end_month).month, 1)
    else:
        end = cases["_month"].max()

    if start_month:
        start = pd.Timestamp(pd.to_datetime(start_month).year, pd.to_datetime(start_month).month, 1)
    else:
        start = end - pd.offsets.MonthBegin(2)  # last 3 months by default

    cases = cases[(cases["_month"] >= start) & (cases["_month"] <= end)]

    # portfolio filter (fuzzy)
    avail_ports = sorted(cases[portc].dropna().unique().tolist())
    port_sel, score = (None, 0)
    if portfolio and portfolio.lower() not in {"all"}:
        port_sel, score = fuzzy_pick(portfolio, avail_ports, cutoff=75)
        if port_sel:
            cases = cases[cases[portc] == port_sel]

    st.caption(f"start_month: {start.date()} | end_month: {end.date()}")
    st.caption(f"portfolio: {port_sel or 'All'}")

    if cases.empty:
        st.info(
            "No cases after filters.\n\n"
            f"Try one of these portfolios: {avail_ports[:10]}"
        )
        return

    # unique cases per month
    out = (cases.groupby("_month")[case_id]
                 .nunique()
                 .reset_index(name="unique_cases")
                 .sort_values("_month"))

    out["_month"] = out["_month"].dt.strftime("%b %y")
    st.dataframe(out, use_container_width=True)
