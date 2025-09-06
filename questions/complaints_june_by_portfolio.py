# questions/complaints_june_by_portfolio.py
from __future__ import annotations

import pandas as pd
import streamlit as st


def run(store, params=None, user_text=None):
    """
    Complaint analysis for June 2025:
      - Join ONLY by Portfolio + Month (no Process)
      - Cases month: from Create/Report/Start date (prepared by data_store as _month)
      - Complaints month: from 'Date Complaint Received' or Month + assumed year (2025)
      - Output: portfolio | cases | complaints | per_1000  (+ overall total)
    """
    params = params or {}
    target_period = pd.Period("2025-06", freq="M")

    cases = store.get("cases")
    complaints = store.get("complaints")

    if cases is None or cases.empty:
        st.info("No cases data loaded.")
        return
    if complaints is None or complaints.empty:
        st.info("No complaints data loaded.")
        return

    # normalise portfolio and ensure _month is present
    for df in (cases, complaints):
        if "portfolio" in df.columns:
            df["portfolio"] = df["portfolio"].astype(str).str.strip().str.title()
        if "_month" not in df.columns:
            if "date" in df.columns:
                df["_month"] = pd.to_datetime(df["date"], errors="coerce").dt.to_period("M")
            else:
                df["_month"] = pd.NaT

    cases_m = cases[cases["_month"] == target_period]
    comp_m = complaints[complaints["_month"] == target_period]

    # Optional portfolio filter if user typed "for London"
    pf = (params or {}).get("portfolio")
    if pf:
        pf = str(pf).strip().str.title()
        cases_m = cases_m[cases_m["portfolio"] == pf]
        comp_m = comp_m[complaints["portfolio"] == pf]

    # Aggregate by portfolio only
    cases_by = cases_m.groupby("portfolio", dropna=False).size().rename("cases")
    comp_by = comp_m.groupby("portfolio", dropna=False).size().rename("complaints")

    out = pd.concat([cases_by, comp_by], axis=1).fillna(0)
    if out.empty:
        st.info("No rows returned for the current filters.")
        with st.expander("Parsed filters", expanded=False):
            st.write(f"month: 2025-06 | portfolio: {pf or 'All'}")
        return

    out = out.astype({"cases": int, "complaints": int}).reset_index()
    out["per_1000"] = (out["complaints"] / out["cases"].where(out["cases"] != 0, pd.NA) * 1000).round(1)
    out = out.sort_values("portfolio", na_position="last")

    # Totals row
    total = pd.DataFrame({
        "portfolio": ["All"],
        "cases": [cases_m.shape[0]],
        "complaints": [comp_m.shape[0]],
    })
    total["per_1000"] = (total["complaints"] / total["cases"].where(total["cases"] != 0, pd.NA) * 1000).round(1)

    st.subheader("Complaints per 1,000 cases — June 2025")
    with st.expander("Parsed filters", expanded=False):
        st.write(f"month: 2025-06 | portfolio: {pf or 'All'}")

    st.dataframe(out[["portfolio", "cases", "complaints", "per_1000"]], use_container_width=True)
    st.dataframe(total[["portfolio", "cases", "complaints", "per_1000"]], use_container_width=True)
