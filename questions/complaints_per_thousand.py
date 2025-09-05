# questions/complaints_per_thousand.py
import pandas as pd
import numpy as np
import streamlit as st

try:
    from rapidfuzz import process as rf_process, fuzz as rf_fuzz
    _FUZZ = True
except Exception:
    _FUZZ = False


# ------------ helpers ------------
def _first(df, names):
    for n in names:
        if n in df.columns:
            return n
    return None

def _norm_series(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower()

def _month_from(df, pref_names) -> pd.Series:
    col = _first(df, pref_names)
    if col is None:
        raise ValueError(f"Missing any of {pref_names}")
    return pd.to_datetime(df[col], errors="coerce").dt.to_period("M").dt.to_timestamp()

def _portfolio_col(df) -> str:
    return _first(df, ["Portfolio_std", "portfolio_std", "Portfolio", "portfolio"])

def _process_col_cases(df) -> str:
    return _first(df, ["Process Name", "ProcessName", "process_name", "process"])

def _parent_case_col(df) -> str:
    return _first(df, ["Parent Case Type", "Parent_Case_Type", "parent_case_type"])

def _case_id_col(df) -> str:
    return _first(df, ["Case ID", "CaseID", "case_id"])

def _fmt_month(s: pd.Series) -> pd.Series:
    return s.dt.strftime("%Y-%m")

def _fuzzy_map_parents_to_process(parent_vals, process_vals):
    """Map complaints' parent case types to the closest process in cases."""
    if not _FUZZ or len(process_vals) == 0:
        # Best-effort identity / title-case
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
            mp[p] = p  # leave as-is; may not overlap
    return mp


# ------------ main ------------
def run(store, params, user_text=None):
    """
    KPI: Complaints per 1,000 cases by process (Create Date for cases; Report Date for complaints).
    Expects in store:
      - store["cases"]: case-level rows
      - store["complaints"]: complaint rows
    """
    cases = store.get("cases")
    complaints = store.get("complaints")

    if cases is None or complaints is None or len(cases) == 0 or len(complaints) == 0:
        st.info("Cases and/or complaints are not loaded.")
        return

    # Normalize months
    cases = cases.copy()
    complaints = complaints.copy()

    cases["Month"] = _month_from(cases, ["Create Date", "Create_Date", "CreateDate", "Create Dat"])
    complaints["Month"] = _month_from(complaints, ["Report_Date", "Report Date", "ReportDate", "Report Dt"])

    # Normalize portfolio
    p_cases_col = _portfolio_col(cases)
    p_comp_col = _portfolio_col(complaints)
    if p_cases_col is None or p_comp_col is None:
        st.error("Portfolio column not found in cases/complaints.")
        return

    # Normalize process fields
    proc_col = _process_col_cases(cases)
    parent_col = _parent_case_col(complaints)
    if proc_col is None or parent_col is None:
        st.error("Process (cases) or Parent Case Type (complaints) column missing.")
        return

    # Filters from params
    portfolio = (params or {}).get("portfolio")
    date_from = (params or {}).get("date_from")  # pandas Timestamp or string ok
    date_to   = (params or {}).get("date_to")

    if portfolio:
        mask_cases = _norm_series(cases[p_cases_col]).eq(str(portfolio).strip().lower()) if p_cases_col == "portfolio" else _norm_series(cases[p_cases_col]).eq(_norm_series(pd.Series([portfolio])).iloc[0])
        mask_comp  = _norm_series(complaints[p_comp_col]).eq(_norm_series(pd.Series([portfolio])).iloc[0])
        cases = cases[mask_cases]
        complaints = complaints[mask_comp]

    if date_from:
        date_from = pd.to_datetime(date_from)
        cases = cases[cases["Month"] >= date_from]
        complaints = complaints[complaints["Month"] >= date_from]
    if date_to:
        date_to = pd.to_datetime(date_to)
        cases = cases[cases["Month"] <= date_to]
        complaints = complaints[complaints["Month"] <= date_to]

    if cases.empty or complaints.empty:
        st.warning("No data after applying filters.")
        return

    # Standardize keys
    cases["_p"] = _norm_series(cases[p_cases_col])
    cases["_proc"] = _norm_series(cases[proc_col])
    complaints["_p"] = _norm_series(complaints[p_comp_col])
    complaints["_parent"] = _norm_series(complaints[parent_col])

    # Build mapping Parent -> Process from the processes actually present in filtered cases
    parent_vals = complaints["_parent"].unique().tolist()
    process_vals = cases["_proc"].unique().tolist()
    mapper = _fuzzy_map_parents_to_process(parent_vals, process_vals)
    complaints["_proc_mapped"] = complaints["_parent"].map(mapper)

    # Aggregate
    cid = _case_id_col(cases)
    if cid is None:
        # if we don't have distinct case IDs, fall back to counting rows
        grp_cases = cases.groupby(["Month", "_p", "_proc"], dropna=False).size().rename("cases").reset_index()
    else:
        grp_cases = (
            cases.groupby(["Month", "_p", "_proc"], dropna=False)[cid]
            .nunique()
            .rename("cases").reset_index()
        )

    grp_complaints = (
        complaints.groupby(["Month", "_p", "_proc_mapped"], dropna=False)
        .size().rename("complaints").reset_index()
        .rename(columns={"_proc_mapped": "_proc"})
    )

    joined = pd.merge(
        grp_cases, grp_complaints,
        on=["Month", "_p", "_proc"],
        how="inner",
        validate="many_to_many"
    )

    # Portfolio/month fallback if no overlap
    if joined.empty:
        st.caption("No process-level overlap found. Showing portfolio-level rate instead.")
        c_by = grp_cases.groupby(["Month", "_p"], as_index=False)["cases"].sum()
        q_by = grp_complaints.groupby(["Month", "_p"], as_index=False)["complaints"].sum()
        j2 = pd.merge(c_by, q_by, on=["Month", "_p"], how="inner")
        if j2.empty:
            st.info("No overlapping data for cases and complaints.")
            return
        j2["complaints_per_1000"] = (j2["complaints"] / j2["cases"].replace(0, np.nan)) * 1000
        j2["Month_str"] = _fmt_month(j2["Month"])
        st.subheader("Complaints per 1,000 cases (Portfolio level)")
        st.dataframe(j2[["Month_str", "_p", "cases", "complaints", "complaints_per_1000"]].rename(
            columns={"Month_str": "Month", "_p": "Portfolio"}
        ), use_container_width=True)
        return

    # Compute KPI
    joined["complaints_per_1000"] = (joined["complaints"] / joined["cases"].replace(0, np.nan)) * 1000
    joined["Month_str"] = _fmt_month(joined["Month"])

    title_bits = ["Complaints per 1,000 cases"]
    if portfolio:
        title_bits.append(f"— {portfolio}")

    st.subheader(" ".join(title_bits))
    st.dataframe(
        joined.sort_values(["Month", "_p", "_proc"])[
            ["Month_str", "_p", "_proc", "cases", "complaints", "complaints_per_1000"]
        ].rename(columns={"Month_str": "Month", "_p": "Portfolio", "_proc": "Process"}),
        use_container_width=True,
    )
