# questions/complaints_per_thousand.py
import re
import numpy as np
import pandas as pd
import streamlit as st

try:
    from rapidfuzz import process as rf_process, fuzz as rf_fuzz
    _FUZZ = True
except Exception:
    _FUZZ = False

def _canon(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).lower())

def _find_col(df: pd.DataFrame, candidates, *, required=True):
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

def _portfolio_col(df) -> str:
    return _find_col(df, ["Portfolio_std", "Portfolio", "portfolio_std", "portfolio"])

def _process_col_cases(df) -> str:
    return _find_col(df, ["Process Name", "ProcessName", "process_name", "process"])

def _parent_case_col(df) -> str:
    return _find_col(df, ["Parent Case Type", "Parent_Case_Type", "parent_case_type"])

def _case_id_col(df) -> str:
    return _find_col(df, ["Case ID", "CaseID", "case_id"], required=False)

def _fuzzy_map_parents_to_process(parent_vals, process_vals):
    if not _FUZZ or len(process_vals) == 0:
        return {p: p for p in parent_vals}
    procs = sorted(set(process_vals))
    mp = {}
    for p in set(parent_vals):
        if p in procs:
            mp[p] = p
            continue
        match = rf_process.extractOne(p, procs, scorer=rf_fuzz.WRatio)
        if match and match[1] >= 88:
            mp[p] = match[0]
        else:
            mp[p] = p
    return mp

def run(store, params, user_text=None):
    cases = store.get("cases")
    complaints = store.get("complaints")
    if cases is None or complaints is None or cases.empty or complaints.empty:
        st.info("Cases and/or complaints are not loaded.")
        return

    # we expect Month + Month_label from data_store
    if "Month" not in cases.columns or "Month" not in complaints.columns:
        st.error("Month column missing; please refresh (loader sets Month).")
        return

    cases = cases.copy()
    complaints = complaints.copy()

    p_cases_col = _portfolio_col(cases)
    p_comp_col  = _portfolio_col(complaints)
    proc_col    = _process_col_cases(cases)
    parent_col  = _parent_case_col(complaints)

    portfolio = (params or {}).get("portfolio")
    date_from = (params or {}).get("date_from")
    date_to   = (params or {}).get("date_to")

    if portfolio:
        tgt = _norm(pd.Series([portfolio])).iloc[0]
        cases = cases[_norm(cases[p_cases_col]).eq(tgt)]
        complaints = complaints[_norm(complaints[p_comp_col]).eq(tgt)]

    if date_from:
        ts = pd.to_datetime(date_from)
        cases = cases[cases["Month"] >= ts]
        complaints = complaints[complaints["Month"] >= ts]
    if date_to:
        ts = pd.to_datetime(date_to)
        cases = cases[cases["Month"] <= ts]
        complaints = complaints[complaints["Month"] <= ts]

    if cases.empty or complaints.empty:
        st.info("No data after applying filters.")
        return

    cases["_p"]   = _norm(cases[p_cases_col])
    cases["_proc"] = _norm(cases[proc_col])
    complaints["_p"] = _norm(complaints[p_comp_col])
    complaints["_parent"] = _norm(complaints[parent_col])

    # Map Parent Case Type (complaints) -> nearest Process Name (cases)
    mapper = _fuzzy_map_parents_to_process(
        complaints["_parent"].unique().tolist(),
        cases["_proc"].unique().tolist()
    )
    complaints["_proc_mapped"] = complaints["_parent"].map(mapper)

    # aggregate
    cid = _case_id_col(cases)
    if cid:
        grp_cases = cases.groupby(["Month", "_p", "_proc"], dropna=False)[cid].nunique().rename("cases").reset_index()
    else:
        grp_cases = cases.groupby(["Month", "_p", "_proc"], dropna=False).size().rename("cases").reset_index()

    grp_complaints = (
        complaints.groupby(["Month", "_p", "_proc_mapped"], dropna=False)
        .size().rename("complaints").reset_index()
        .rename(columns={"_proc_mapped": "_proc"})
    )

    joined = pd.merge(grp_cases, grp_complaints, on=["Month", "_p", "_proc"], how="inner")

    if joined.empty:
        st.caption("No process-level overlap found. Showing portfolio-level rate instead.")
        c_by = grp_cases.groupby(["Month", "_p"], as_index=False)["cases"].sum()
        q_by = grp_complaints.groupby(["Month", "_p"], as_index=False)["complaints"].sum()
        j2 = pd.merge(c_by, q_by, on=["Month", "_p"], how="inner")
        if j2.empty:
            st.info("No overlapping data for cases and complaints.")
            return
        j2["complaints_per_1000"] = (j2["complaints"] / j2["cases"].replace(0, np.nan)) * 1000
        j2["Month_label"] = j2["Month"].dt.strftime("%b %y")
        st.subheader("Complaints per 1,000 cases (Portfolio level)")
        st.dataframe(j2.rename(columns={"_p": "Portfolio"})[["Month_label","Portfolio","cases","complaints","complaints_per_1000"]],
                     use_container_width=True)
        return

    joined["complaints_per_1000"] = (joined["complaints"] / joined["cases"].replace(0, np.nan)) * 1000
    joined["Month_label"] = joined["Month"].dt.strftime("%b %y")

    st.subheader("Complaints per 1,000 cases")
    st.dataframe(
        joined.sort_values(["Month", "_p", "_proc"])[
            ["Month_label", "_p", "_proc", "cases", "complaints", "complaints_per_1000"]
        ].rename(columns={"_p": "Portfolio", "_proc": "Process"}),
        use_container_width=True,
    )
