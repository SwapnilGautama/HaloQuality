# questions/rca1_portfolio_process.py
from __future__ import annotations

import pandas as pd
import streamlit as st

def _find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    cl = {c.lower(): c for c in df.columns}
    for cand in candidates:
        key = cand.lower()
        if key in cl:
            return cl[key]
        for lc, original in cl.items():
            if key == lc or key in lc:
                return original
    return None

def _ensure_month_col(df: pd.DataFrame, date_candidates: list[str]) -> str:
    if "month_dt" in df.columns:
        return "month_dt"
    col = _find_col(df, date_candidates)
    if not col:
        raise ValueError(f"Could not find a date column among: {date_candidates}")
    df["month_dt"] = pd.to_datetime(df[col], errors="coerce").dt.to_period("M").dt.to_timestamp()
    return "month_dt"

def run(store: dict, params: dict | None = None, args: dict | None = None, user_text: str = ""):
    p = (params or args or {}).copy()

    complaints = store.get("complaints")
    if complaints is None:
        st.error("Complaints data not loaded.")
        return

    # Columns
    c_process   = _find_col(complaints, ["Process", "Process Name", "Department", "Function"])
    c_portfolio = _find_col(complaints, ["Portfolio", "Product", "LOB", "Line of Business", "Account", "Brand"])
    c_rca1      = _find_col(complaints, ["RCA1", "RCA 1", "Root Cause 1", "Primary Cause"])
    month_col   = _ensure_month_col(complaints, ["Date Complaint Received - DD/MM/YY", "Date Complaint Received", "Complaint Date", "Created Date"])

    if c_process is None or c_portfolio is None:
        st.warning("Required columns not found in complaints (need Process and Portfolio).")
        return

    if c_rca1 is None:
        st.info("RCA labels not found. Please run the complaints labeller so 'RCA1' exists.")
        return

    df = complaints.copy()
    df[c_process]   = df[c_process].astype(str).str.strip()
    df[c_portfolio] = df[c_portfolio].astype(str).str.strip()
    df[c_rca1]      = df[c_rca1].astype(str).str.strip()

    # Filters
    flt_process   = p.get("process")      # e.g., "Member Enquiry"
    flt_portfolio = p.get("portfolio")
    rel_months    = p.get("months") or p.get("relative_months") or 3  # default last 3

    # Month range (last N months up to max available)
    maxm = df[month_col].max()
    if pd.isna(maxm):
        st.info("No dates in complaints to calculate last months range.")
        return
    startm = (maxm.to_period("M") - (int(rel_months) - 1)).to_timestamp()
    df = df[(df[month_col] >= startm) & (df[month_col] <= maxm)]

    if flt_process:
        df = df[df[c_process].str.contains(str(flt_process), case=False, na=False)]
    if flt_portfolio:
        df = df[df[c_portfolio].str.contains(str(flt_portfolio), case=False, na=False)]

    st.subheader(f"RCA1 by Portfolio × Process — last {int(rel_months)} months")
    st.caption(f"start_month: {startm.date()} | end_month: {maxm.date()}")

    if df.empty:
        st.info("No complaints for the current filters.")
        return

    # Aggregate: counts of RCA1; show table by Process × Portfolio
    agg = (
        df.groupby([c_process, c_portfolio])[c_rca1]
        .count()
        .reset_index(name="rca1_count")
        .sort_values("rca1_count", ascending=False)
    )

    # Pivot for display
    pivot = agg.pivot_table(
        index=c_process,
        columns=c_portfolio,
        values="rca1_count",
        aggfunc="sum",
        fill_value=0,
        dropna=False,
    )

    st.dataframe(pivot, use_container_width=True)
