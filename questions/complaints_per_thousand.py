# questions/complaints_per_thousand.py
from __future__ import annotations

import pandas as pd
import numpy as np
import streamlit as st

# ---------- helpers

def _find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    cl = {c.lower(): c for c in df.columns}
    for cand in candidates:
        key = cand.lower()
        if key in cl:
            return cl[key]
        # try loose contains match
        for lc, original in cl.items():
            if key == lc or key in lc:
                return original
    return None

def _ensure_month(dt_series: pd.Series) -> pd.Series:
    # coerce to datetime, then to month start (timestamp)
    s = pd.to_datetime(dt_series, errors="coerce")
    return s.dt.to_period("M").dt.to_timestamp()

def _get_month_col(df: pd.DataFrame, preferred: list[str]) -> str:
    # use existing month_dt if it’s there; else build from preferred date columns
    if "month_dt" in df.columns:
        return "month_dt"
    col = _find_col(df, preferred)
    if not col:
        raise ValueError(f"Could not find a date column among: {preferred}")
    df["month_dt"] = _ensure_month(df[col])
    return "month_dt"

def _normalise_text(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip()

def _require_any(df: pd.DataFrame, bundles: dict[str, list[str]]) -> dict[str, str]:
    """Returns a dict mapping canonical -> actual column names found.
    Tries multiple candidates for each canonical key."""
    out = {}
    for canonical, candidates in bundles.items():
        found = _find_col(df, candidates)
        if not found:
            raise KeyError(f"Required column not found for '{canonical}'. Tried: {candidates}")
        out[canonical] = found
    return out

# ---------- main

def run(store: dict, params: dict | None = None, args: dict | None = None, user_text: str = ""):
    """
    Complaints per 1,000 cases by Process (monthly).
    Accepts either `params` or `args` (both supported by app.py).
    """
    p = (params or args or {}).copy()

    # Optional filters from matcher:
    # portfolio, process, city/site/region, start_month, end_month
    flt_portfolio = p.get("portfolio")
    flt_process   = p.get("process")
    flt_city      = p.get("city") or p.get("site") or p.get("location") or p.get("region")

    start_month = p.get("start_month")  # YYYY-MM or YYYY-MM-01; flexible
    end_month   = p.get("end_month")

    # ----- pull data
    cases = store.get("cases")
    complaints = store.get("complaints")
    if cases is None or complaints is None:
        st.error("Cases or Complaints data not loaded.")
        return

    # ----- column normalisation
    # Cases columns
    case_cols = _require_any(
        cases,
        {
            "id": ["Case ID", "CaseID", "Case Number", "Case_No", "Reference", "Ref ID"],
            "process": ["Process", "Process Name", "Case Type", "Workflow"],
            "portfolio": ["Portfolio", "Product", "LOB", "Line of Business", "Account", "Brand"],
            "city": ["Site", "Location", "City", "Centre", "Center", "Office", "Region"],
        },
    )
    case_month_col = _get_month_col(cases, ["Create Date", "Created Date", "Created On", "Report Date", "Report_Date"])
    cases[case_cols["process"]] = _normalise_text(cases[case_cols["process"]])
    cases[case_cols["portfolio"]] = _normalise_text(cases[case_cols["portfolio"]])

    # Complaints columns
    comp_cols = _require_any(
        complaints,
        {
            "id": ["Complaint ID", "Complaint Ref", "Reference Number", "ID"],
            "process": ["Process", "Process Name", "Department", "Function", "Complaint Type"],
            "portfolio": ["Portfolio", "Product", "LOB", "Line of Business", "Account", "Brand"],
            "city": ["Site", "Location", "City", "Centre", "Center", "Office", "Region"],
        },
    )
    comp_month_col = _get_month_col(
        complaints,
        ["Date Complaint Received - DD/MM/YY", "Date Complaint Received", "Date Received", "Complaint Date", "Created Date"],
    )
    complaints[comp_cols["process"]] = _normalise_text(complaints[comp_cols["process"]])
    complaints[comp_cols["portfolio"]] = _normalise_text(complaints[comp_cols["portfolio"]])

    # ----- apply filters
    def _apply_filters(df, cols):
        if flt_portfolio:
            df = df[df[cols["portfolio"]].str.contains(str(flt_portfolio), case=False, na=False)]
        if flt_process:
            df = df[df[cols["process"]].str.contains(str(flt_process), case=False, na=False)]
        if flt_city:
            city_col = cols.get("city")
            if city_col and city_col in df.columns:
                df = df[df[city_col].astype(str).str.contains(str(flt_city), case=False, na=False)]
        if start_month:
            sm = pd.to_datetime(str(start_month), errors="coerce")
            if not pd.isna(sm):
                df = df[df[_get_month_col(df, [])] >= sm.to_period("M").to_timestamp()]
        if end_month:
            em = pd.to_datetime(str(end_month), errors="coerce")
            if not pd.isna(em):
                df = df[df[_get_month_col(df, [])] <= em.to_period("M").to_timestamp()]
        return df

    cases_f = _apply_filters(cases, case_cols)
    complaints_f = _apply_filters(complaints, comp_cols)

    # ----- month range default (if none provided)
    # sync complaints/cases overlapping range
    if not start_month or not end_month:
        cmin = cases_f[case_month_col].min()
        cmax = cases_f[case_month_col].max()
        pmin = complaints_f[comp_month_col].min()
        pmax = complaints_f[comp_month_col].max()
        lo = max(cmin, pmin)
        hi = min(cmax, pmax)
        # apply if present
        if pd.notna(lo):
            cases_f = cases_f[cases_f[case_month_col] >= lo]
            complaints_f = complaints_f[complaints_f[comp_month_col] >= lo]
        if pd.notna(hi):
            cases_f = cases_f[cases_f[case_month_col] <= hi]
            complaints_f = complaints_f[complaints_f[comp_month_col] <= hi]

    # ----- aggregate
    cases_g = (
        cases_f.groupby([case_month_col, case_cols["process"]])[case_cols["id"]]
        .nunique()
        .reset_index(name="cases")
    )
    comp_g = (
        complaints_f.groupby([comp_month_col, comp_cols["process"]])[comp_cols["id"]]
        .nunique()
        .reset_index(name="complaints")
    )

    # align month col names
    comp_g = comp_g.rename(columns={comp_month_col: "month", comp_cols["process"]: "Process"})
    cases_g = cases_g.rename(columns={case_month_col: "month", case_cols["process"]: "Process"})

    # join + rate
    df = pd.merge(cases_g, comp_g, on=["month", "Process"], how="outer").fillna(0.0)
    df["cases"] = df["cases"].astype(float)
    df["complaints"] = df["complaints"].astype(float)
    df["per_1000"] = np.where(df["cases"] > 0, (df["complaints"] / df["cases"]) * 1000.0, np.nan)
    df = df.sort_values(["month", "Process"])

    # display
    title_bits = []
    if flt_portfolio: title_bits.append(f"Portfolio: {flt_portfolio}")
    if flt_process:   title_bits.append(f"Process filter: {flt_process}")
    if start_month or end_month:
        title_bits.append(f"Range: {start_month or '…'} → {end_month or '…'}")

    st.subheader("Complaints per 1,000 cases")
    if title_bits:
        st.caption(" | ".join(title_bits))

    if df.empty:
        st.info("No overlapping data for cases and complaints with the current filters.")
        return

    tidy = df.copy()
    tidy["_month"] = tidy["month"].dt.strftime("%b %y")
    tidy = tidy[["_month", "Process", "cases", "complaints", "per_1000"]]
    tidy = tidy.rename(columns={"_month": "month"})

    st.dataframe(tidy, use_container_width=True)
