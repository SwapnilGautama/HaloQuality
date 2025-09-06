# questions/complaints_june_by_portfolio.py
from __future__ import annotations

import pandas as pd
import streamlit as st


ASSUME_YEAR = 2025
TARGET_PERIOD = pd.Period(f"{ASSUME_YEAR}-06", freq="M")


def _coerce_portfolio(df: pd.DataFrame) -> pd.DataFrame:
    # Normalize to a single 'portfolio' column, title-cased
    if "portfolio" not in df.columns:
        # find a case-insensitive match
        match = next((c for c in df.columns if c.strip().lower() == "portfolio"), None)
        if match:
            df = df.rename(columns={match: "portfolio"})
    if "portfolio" in df.columns:
        df["portfolio"] = df["portfolio"].astype(str).str.strip().str.title()
    return df


def _ensure_month(df: pd.DataFrame, kind: str) -> pd.DataFrame:
    """
    Ensure a monthly period column '_month' exists.

    kind = "cases"       -> derive from a create/report/start date
    kind = "complaints"  -> derive from 'Date Complaint Received - DD/MM/YY' or Month + ASSUME_YEAR
    """
    if "_month" in df.columns:
        return df

    dfx = df.copy()
    if kind == "cases":
        # common candidates in your files
        candidates = [
            "Create Date", "Create Dt", "Report Date", "Start Date", "CreateDate", "Created",
            "Create Dt.", "Create_Dt", "Create_Dt.", "Create_Date",
            "Date",
        ]
        col = next((c for c in dfx.columns if c.strip() in candidates), None)
        if col is None:
            # fallback: pick first column that looks like a date
            col = next((c for c in dfx.columns if "date" in c.lower()), None)

        if col is not None:
            dfx["_month"] = pd.to_datetime(dfx[col], errors="coerce").dt.to_period("M")
        else:
            dfx["_month"] = pd.NaT

    else:  # complaints
        col_date = next((c for c in dfx.columns if c.strip().lower() == "date complaint received - dd/mm/yy"), None)
        if col_date:
            dfx["_month"] = pd.to_datetime(dfx[col_date], errors="coerce", dayfirst=True).dt.to_period("M")
        else:
            # Try a 'Month' column (e.g., "June", "Jun"); assume ASSUME_YEAR
            col_month = next((c for c in dfx.columns if c.strip().lower() == "month"), None)
            if col_month:
                dfx["_month"] = pd.to_datetime(dfx[col_month].astype(str).str.strip() + f" {ASSUME_YEAR}",
                                               errors="coerce").dt.to_period("M")
            else:
                dfx["_month"] = pd.NaT

    return dfx


def run(store, params=None, user_text=None):
    """
    Complaint analysis for June 2025:
      - Join ONLY by Portfolio + Month (no Process)
      - Cases month: Create/Report/Start date (or prebuilt _month)
      - Complaints month: 'Date Complaint Received - DD/MM/YY' or Month + ASSUME_YEAR
      - Output: portfolio | cases | complaints | per_1000  (+ overall total)
    """
    params = params or {}

    cases = store.get("cases")
    complaints = store.get("complaints")

    if cases is None or cases.empty:
        st.info("No cases data loaded.")
        return
    if complaints is None or complaints.empty:
        st.info("No complaints data loaded.")
        return

    # normalize schema
    cases = _coerce_portfolio(cases)
    complaints = _coerce_portfolio(complaints)
    cases = _ensure_month(cases, kind="cases")
    complaints = _ensure_month(complaints, kind="complaints")

    # focus on June 2025
    cases_m = cases[cases["_month"] == TARGET_PERIOD].copy()
    comp_m = complaints[complaints["_month"] == TARGET_PERIOD].copy()

    # Optional portfolio filter from user text ("for London")
    pf = (params or {}).get("portfolio")
    if pf:
        pf = str(pf).strip().title()
        if "portfolio" in cases_m.columns:
            cases_m = cases_m[cases_m["portfolio"] == pf]
        if "portfolio" in comp_m.columns:
            comp_m = comp_m[comp_m["portfolio"] == pf]

    # Aggregate by portfolio only
    if "portfolio" not in cases_m.columns or "portfolio" not in comp_m.columns:
        st.warning("Missing 'Portfolio' in cases or complaints after normalization.")
        with st.expander("Parsed filters", expanded=False):
            st.write(f"month: {TARGET_PERIOD} | portfolio: {pf or 'All'}")
        return

    cases_by = cases_m.groupby("portfolio", dropna=False).size().rename("cases")
    comp_by = comp_m.groupby("portfolio", dropna=False).size().rename("complaints")

    out = pd.concat([cases_by, comp_by], axis=1).fillna(0)
    if out.empty:
        st.info("No rows returned for the current filters.")
        with st.expander("Parsed filters", expanded=False):
            st.write(f"month: {TARGET_PERIOD} | portfolio: {pf or 'All'}")
        return

    out = out.astype({"cases": int, "complaints": int}).reset_index()
    out["per_1000"] = (out["complaints"] / out["cases"].where(out["cases"] != 0, pd.NA) * 1000).round(1)
    out = out.sort_values("portfolio", na_position="last")

    # Total row
    total = pd.DataFrame({
        "portfolio": ["All"],
        "cases": [cases_m.shape[0]],
        "complaints": [comp_m.shape[0]],
    })
    total["per_1000"] = (total["complaints"] / total["cases"].where(total["cases"] != 0, pd.NA) * 1000).round(1)

    st.subheader("Complaints per 1,000 cases — June 2025")
    with st.expander("Parsed filters", expanded=False):
        st.write(f"month: {TARGET_PERIOD} | portfolio: {pf or 'All'}")

    st.dataframe(out[["portfolio", "cases", "complaints", "per_1000"]], use_container_width=True)
    st.dataframe(total[["portfolio", "cases", "complaints", "per_1000"]], use_container_width=True)
