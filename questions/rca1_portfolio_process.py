# questions/rca1_portfolio_process.py
from __future__ import annotations
import pandas as pd
import streamlit as st

from ._utils import pick_col, ensure_month_series, fuzzy_pick, available_values

# column fallbacks for complaints data
PROC_COLS = ["Process", "Process Name", "Primary Process"]
PORT_COLS = ["Portfolio", "Portfolio Name", "LOB", "Business Unit"]
DATE_COLS = ["month_dt", "Report Date", "Report_Date", "ReportDate",
             "Date Complaint Received - DD/MM/YY"]

def run(store, relative_months: int = 3,
        start_month: str | None = None,
        end_month: str | None = None,
        portfolio: str | None = None,
        process: str | None = None,
        user_text: str | None = None):
    df = store["complaints"].copy()

    procc = pick_col(df, PROC_COLS)
    portc = pick_col(df, PORT_COLS)
    if procc is None or portc is None:
        st.warning("Required columns not found in complaints (need Process and Portfolio).")
        return

    # normalize month series
    month_s = ensure_month_series(df, DATE_COLS)
    df["_month"] = month_s

    # resolve time window
    if end_month:
        end = pd.to_datetime(end_month)  # yyyy-mm
        end = pd.Timestamp(end.year, end.month, 1)
    else:
        end = df["_month"].max()

    if start_month:
        start = pd.to_datetime(start_month)
        start = pd.Timestamp(start.year, start.month, 1)
    else:
        start = (end - pd.offsets.MonthBegin(relative_months-1))

    # apply time filter
    df = df[(df["_month"] >= start) & (df["_month"] <= end)]

    # fuzzy portfolio/process
    avail = available_values(df, PROC_COLS, PORT_COLS, DATE_COLS)

    port_sel, port_score = (None, 0)
    if portfolio and portfolio.lower() not in {"all", "for"}:
        port_sel, port_score = fuzzy_pick(portfolio, avail["portfolios"], cutoff=75)

    proc_sel, proc_score = (None, 0)
    if process:
        proc_sel, proc_score = fuzzy_pick(process, avail["processes"], cutoff=75)

    if proc_sel:
        df = df[df[procc] == proc_sel]
    if port_sel:
        df = df[df[portc] == port_sel]

    st.caption(f"relative_months: {relative_months}")
    st.caption(f"start_month: {start.date()} | end_month: {end.date()}")
    st.caption(f"portfolio: {port_sel or 'All'} (match {port_score}%)")
    st.caption(f"process: {proc_sel or 'All'} (match {proc_score}%)")

    if df.empty:
        st.info(
            "No complaints after filters.\n\n"
            f"Available months in window: "
            f"{pd.Series(avail['months']).dt.strftime('%b %y').tolist()}\n\n"
            f"Try one of these processes: {avail['processes'][:10]}\n"
            f"and portfolios: {avail['portfolios'][:10]}"
        )
        return

    # RCA1: show top drivers by Portfolio x Process (count of complaints)
    grp = (df.groupby([portc, procc])
             .size()
             .reset_index(name="count")
             .sort_values("count", ascending=False))

    if grp.empty:
        st.info("No grouped results.")
        return

    # simple pivot for readability
    pivot = grp.pivot_table(index=portc, columns=procc, values="count", fill_value=0)
    st.dataframe(pivot, use_container_width=True)
