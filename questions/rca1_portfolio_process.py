# questions/rca1_portfolio_process.py
import re
import pandas as pd
import streamlit as st

def _canon(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).lower())

def _find_col(df, candidates, *, required=True):
    if df is None or df.empty:
        if required: raise ValueError("Empty dataframe.")
        return None
    lookup = {_canon(c): c for c in df.columns}
    for cand in candidates:
        key = _canon(cand)
        if key in lookup:
            return lookup[key]
    token = _canon(candidates[0])
    for c in df.columns:
        if token in _canon(c):
            return c
    if required:
        raise ValueError(f"Missing any of {candidates}")
    return None

def _norm(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower()

def run(store, params, user_text=None):
    complaints = store.get("complaints")
    if complaints is None or complaints.empty:
        st.info("Complaints data not loaded.")
        return

    if "Month" not in complaints:
        st.error("Month column missing in complaints; please refresh loaders.")
        return

    complaints = complaints.copy()
    p_col   = _find_col(complaints, ["Portfolio_std", "Portfolio", "portfolio_std", "portfolio"])
    proc_c  = _find_col(complaints, ["Parent Case Type", "Parent_Case_Type", "parent_case_type"])
    rca_col = _find_col(complaints, ["RCA1", "RCA_1", "rca1", "rca_1", "RCA1_std", "RCA_1_std"], required=True)

    months_back = (params or {}).get("months_back", 3)
    want_proc   = (params or {}).get("process")
    want_port   = (params or {}).get("portfolio")

    # last N months window by Month
    max_m = complaints["Month"].max()
    if pd.isna(max_m):
        st.error("Complaints have no valid Month.")
        return
    cutoff = (max_m.to_period("M") - (months_back - 1)).to_timestamp()
    complaints = complaints[(complaints["Month"] >= cutoff) & (complaints["Month"] <= max_m)]

    if want_port:
        complaints = complaints[_norm(complaints[p_col]).eq(_norm(pd.Series([want_port])).iloc[0])]
    if want_proc:
        complaints = complaints[_norm(complaints[proc_c]).eq(_norm(pd.Series([want_proc])).iloc[0])]

    if complaints.empty:
        st.info("No complaints after filters.")
        return

    grp = (
        complaints.groupby([p_col, proc_c, rca_col], dropna=False)
        .size().rename("count").reset_index()
    )
    grp["share_%"] = grp["count"] / grp.groupby([p_col, proc_c])["count"].transform("sum") * 100.0

    st.subheader("RCA1 by Portfolio × Process")
    st.caption(f"Last {months_back} months (by Month).")

    pivot = grp.pivot_table(index=[p_col, proc_c], columns=rca_col, values="share_%", aggfunc="sum", fill_value=0.0)
    st.dataframe(pivot.reset_index(), use_container_width=True)
