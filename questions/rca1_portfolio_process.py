# questions/rca1_portfolio_process.py
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

def _norm(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower()

def _portfolio_col(df):
    return _first(df, ["Portfolio_std", "portfolio_std", "Portfolio", "portfolio"])

def _process_col_cases(df):
    return _first(df, ["Process Name", "ProcessName", "process_name", "process"])

def _parent_case_col(df):
    return _first(df, ["Parent Case Type", "Parent_Case_Type", "parent_case_type"])

def _rca1_col(df):
    return _first(df, ["RCA1", "RCA_1", "rca1", "rca_1", "RCA1_std", "RCA_1_std"])

def run(store, params, user_text=None):
    """
    RCA1 distribution by Portfolio × Process for last N months (default 3).
    Uses Report_Date for complaints' month; Create Date only used to map processes
    if you want to restrict to processes existing in cases.
    """
    complaints = store.get("complaints")
    cases = store.get("cases")

    if complaints is None or complaints.empty:
        st.info("Complaints data not loaded.")
        return

    # Build Month from Report_Date/Report Date
    complaints = complaints.copy()
    complaints["Month"] = _month_from(complaints, ["Report_Date", "Report Date", "ReportDate", "Report Dt"])
    p_col = _portfolio_col(complaints)
    parent_col = _parent_case_col(complaints)
    rca_col = _rca1_col(complaints)

    if p_col is None or parent_col is None:
        st.error("Required columns missing in complaints (Portfolio/Parent Case Type).")
        return
    if rca_col is None:
        st.error("RCA1 column not found in complaints.")
        return

    # Params
    months_back = (params or {}).get("months_back", 3)
    process_want = (params or {}).get("process")
    portfolio = (params or {}).get("portfolio")

    # Filter last N months from max
    max_m = complaints["Month"].max()
    if pd.isna(max_m):
        st.error("Complaints have no valid Report Date.")
        return
    cutoff = (max_m.to_period("M") - (months_back - 1)).to_timestamp()
    complaints = complaints[(complaints["Month"] >= cutoff) & (complaints["Month"] <= max_m)]

    if portfolio:
        complaints = complaints[_norm(complaints[p_col]).eq(str(portfolio).strip().lower())]

    # Process filter: Member Enquiry, etc.
    # Your dataset keeps the *process* equivalent in Parent Case Type.
    if process_want:
        complaints = complaints[_norm(complaints[parent_col]).eq(str(process_want).strip().lower())]

    if complaints.empty:
        st.info("No complaints after filters.")
        return

    # Aggregate RCA1 shares
    grp = complaints.groupby([p_col, parent_col, rca_col]).size().rename("count").reset_index()
    grp["share_%"] = grp["count"] / grp.groupby([p_col, parent_col])["count"].transform("sum") * 100.0

    st.subheader("RCA1 by Portfolio × Process")
    st.caption(f"Last {months_back} months (by Report Date).")

    # Nice pivot for display
    pivot = grp.pivot_table(index=[p_col, parent_col], columns=rca_col, values="share_%", aggfunc="sum", fill_value=0.0)
    pivot = pivot.reset_index()
    st.dataframe(pivot, use_container_width=True)
