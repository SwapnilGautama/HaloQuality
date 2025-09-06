# questions/complaints_per_thousand.py
from __future__ import annotations

from typing import Any, Dict, Optional

import pandas as pd
import streamlit as st

MONTH_FMT = "%b %Y"


def _normalize_month(s: Optional[str]) -> Optional[pd.Period]:
    if not s:
        return None
    try:
        dt = pd.to_datetime(s, format="%b %Y")
        return dt.to_period("M")
    except Exception:
        return None


def run(store: Dict[str, Any], params: Dict[str, Any], user_text: str = ""):
    cases = store.get("cases", pd.DataFrame()).copy()
    complaints = store.get("complaints", pd.DataFrame()).copy()

    # guard rails
    if cases.empty or complaints.empty:
        st.info("No overlapping data for cases and complaints.")
        return

    # required canonical columns (data_store.py guarantees these when present)
    miss_cases = [c for c in ["portfolio", "_month"] if c not in cases.columns]
    miss_comps = [c for c in ["portfolio", "_month"] if c not in complaints.columns]
    if miss_cases:
        st.info(f"Missing columns in cases: {miss_cases!r}")
        return
    if miss_comps:
        st.info(f"Missing columns in complaints: {miss_comps!r}")
        return

    # parse filters
    portfolio = (params or {}).get("portfolio")
    m_start = _normalize_month((params or {}).get("start"))
    m_end = _normalize_month((params or {}).get("end"))

    # fallback: compute last 3 months in intersection
    if not (m_start and m_end):
        inter = sorted(set(cases["_month"].dropna()) & set(complaints["_month"].dropna()))
        if len(inter) >= 3:
            m_end = inter[-1]
            m_start = inter[-3]
        elif inter:
            m_start = inter[0]
            m_end = inter[-1]

    # filter
    if portfolio:
        cases = cases[cases["portfolio"].eq(str(portfolio).title())]
        complaints = complaints[complaints["portfolio"].eq(str(portfolio).title())]

    if m_start is not None:
        cases = cases[cases["_month"] >= m_start]
        complaints = complaints[complaints["_month"] >= m_start]
    if m_end is not None:
        cases = cases[cases["_month"] <= m_end]
        complaints = complaints[complaints["_month"] <= m_end]

    if cases.empty or complaints.empty:
        st.info("No rows returned for the current filters.")
        return

    # We join by month + portfolio. Process is OPTIONAL: if both have it, include it; else aggregate on portfolio only.
    grp_cols = ["portfolio", "_month"]
    if "process" in cases.columns and "process" in complaints.columns:
        grp_cols_proc = grp_cols + ["process"]
        cases_g = cases.groupby(grp_cols_proc, dropna=False).size().rename("cases")
        comps_g = complaints.groupby(grp_cols_proc, dropna=False).size().rename("complaints")
        df = pd.concat([cases_g, comps_g], axis=1).fillna(0).reset_index()
    else:
        cases_g = cases.groupby(grp_cols, dropna=False).size().rename("cases")
        comps_g = complaints.groupby(grp_cols, dropna=False).size().rename("complaints")
        df = pd.concat([cases_g, comps_g], axis=1).fillna(0).reset_index()

    if df.empty:
        st.info("No overlapping data for cases and complaints.")
        return

    df["per_1000"] = df.apply(
        lambda r: (r["complaints"] / r["cases"] * 1000.0) if r["cases"] else 0.0, axis=1
    )
    df["_month_label"] = df["_month"].dt.to_timestamp().dt.strftime(MONTH_FMT)

    order_cols = [c for c in ["_month_label", "portfolio", "process"] if c in df.columns] + [
        "cases",
        "complaints",
        "per_1000",
    ]
    out = df[order_cols].sort_values(["_month_label", "portfolio"] + (["process"] if "process" in df.columns else []))

    st.dataframe(out, use_container_width=True)
