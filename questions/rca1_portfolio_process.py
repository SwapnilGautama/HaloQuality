# questions/rca1_portfolio_process.py
from __future__ import annotations
import re
from typing import Iterable, Optional
import pandas as pd
import numpy as np
import streamlit as st


def _canon(s: str) -> str:
    """lower + strip non-alnum to match columns flexibly."""
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
    df = df.copy()
    df["month_dt"] = dt.dt.to_period("M").dt.to_timestamp()
    df["month_label"] = df["month_dt"].dt.strftime("%b %y")
    return df


def _extract_process(params: dict, user_text: Optional[str]) -> Optional[str]:
    proc = None
    if params:
        # parser may have put it under 'process' or 'process_name'
        proc = params.get("process") or params.get("process_name")
    if not proc and user_text:
        m = re.search(r"\bprocess\s+([A-Za-z0-9 &/_-]+?)(?:\slast|\sin\b|\sfor\b|$)", user_text, flags=re.I)
        if m:
            proc = m.group(1).strip()
    return proc


def run(store: dict, params: dict, user_text: Optional[str] = None):
    """
    Show RCA1 distribution by Portfolio for a given Process over the last 3 months.
    Requires complaints to have an RCA1 column (from your complaints labeller).
    """
    if "complaints" not in store or store["complaints"] is None or len(store["complaints"]) == 0:
        st.info("Complaints data not loaded.")
        return

    df = store["complaints"]

    # --- flexible column picking ---
    process_col = _pick_col(df, ["Process", "Process Name", "ProcessName", "Parent case type", "Parent_Case_Type"])
    portfolio_col = _pick_col(df, ["Portfolio", "Portfolio_std", "Portfolio Name", "PortfolioName"])
    date_col = _pick_col(
        df,
        [
            "month_dt",  # already prepared
            "Date Complaint Received - DD/MM/YY",
            "Date Complaint Received",
            "Report Date",
            "Report_Date",
            "Date",
        ],
    )
    rca_col = _pick_col(df, ["RCA1", "rca1", "RCA_1"])

    missing_bits = []
    if not process_col:
        missing_bits.append("Process")
    if not portfolio_col:
        missing_bits.append("Portfolio")
    if not date_col:
        missing_bits.append("a Date")
    if missing_bits:
        st.info(f"Required columns not found in complaints (need {', '.join(missing_bits)}).")
        return

    if not rca_col:
        st.info("RCA labels not found. Please run the complaints labeller so 'RCA1' exists.")
        return

    # Standardise month fields
    if _canon(date_col) != _canon("month_dt"):
        df = _ensure_month_cols(df, date_col)
    else:
        # month_dt already present — also make label
        out = df.copy()
        if "month_label" not in out.columns:
            out["month_label"] = out["month_dt"].dt.strftime("%b %y")
        df = out

    # --- process selection ---
    wanted_process = _extract_process(params, user_text)
    if not wanted_process:
        st.info("Please specify a process (e.g., *process Member Enquiry*).")
        return

    # case-insensitive match on process
    proc_l = wanted_process.lower().strip()
    mask = df[process_col].astype(str).str.lower().str.strip() == proc_l
    got = df.loc[mask].copy()

    if got.empty:
        st.info(f"No complaints for process **{wanted_process}**.")
        return

    # last 3 months by month_dt
    last3 = (
        got[["month_dt"]]
        .dropna()
        .drop_duplicates()
        .sort_values("month_dt")
        .tail(3)
        .squeeze()
        .tolist()
    )
    if not last3:
        st.info("No months available after date parsing.")
        return

    got = got[got["month_dt"].isin(last3)]
    if got.empty:
        st.info("No complaints in the last 3 months for the selected process.")
        return

    # Group to shares by portfolio × RCA1
    counts = (
        got.groupby([portfolio_col, rca_col], dropna=False)
        .size()
        .rename("count")
        .reset_index()
    )
    total_by_portfolio = counts.groupby(portfolio_col)["count"].transform("sum")
    counts["share_%"] = (counts["count"] / total_by_portfolio * 100).round(1)

    # Pivot for a clean table
    table = (
        counts.pivot(index=portfolio_col, columns=rca_col, values="share_%")
        .fillna(0.0)
        .sort_index()
    )
    table = table.reset_index().rename(columns={portfolio_col: "Portfolio"})

    st.subheader(f"RCA1 by Portfolio — **{wanted_process}** (last 3 months)")
    st.dataframe(table, use_container_width=True)

    # Simple stacked bar chart summary of top 6 portfolios
    try:
        import altair as alt

        melted = counts.copy()
        melted = melted.rename(columns={portfolio_col: "Portfolio", rca_col: "RCA1"})
        # Keep top portfolios by total volume (over last 3 months)
        top_ports = (
            got.groupby(portfolio_col)
            .size()
            .sort_values(ascending=False)
            .head(6)
            .index
            .tolist()
        )
        melted = melted[melted["Portfolio"].isin(top_ports)]

        chart = (
            alt.Chart(melted)
            .mark_bar()
            .encode(
                x=alt.X("sum(share_%):Q", title="Share %"),
                y=alt.Y("Portfolio:N", sort="-x"),
                color=alt.Color("RCA1:N"),
                tooltip=["Portfolio", "RCA1", "share_%"]
            )
        )
        st.altair_chart(chart, use_container_width=True)
    except Exception:
        # Altair not mandatory
        pass
