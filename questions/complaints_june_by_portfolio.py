# questions/complaints_june_by_portfolio.py
from __future__ import annotations

from typing import Any, Dict

import pandas as pd
import streamlit as st


def run(store: Dict[str, Any], params: Dict[str, Any], user_text: str = ""):
    cases = store.get("cases", pd.DataFrame()).copy()
    complaints = store.get("complaints", pd.DataFrame()).copy()

    if cases.empty or complaints.empty:
        st.info("No overlapping data for cases and complaints.")
        return

    # data_store guarantees canonical columns when present
    need_cases = [c for c in ["portfolio", "_month"] if c not in cases.columns]
    need_comps = [c for c in ["portfolio", "_month"] if c not in complaints.columns]
    if need_cases:
        st.info(f"Missing columns in cases: {need_cases!r}")
        return
    if need_comps:
        st.info(f"Missing columns in complaints: {need_comps!r}")
        return

    # force June 2025 (complaints Month->_month handled in data_store via assume_year)
    target = pd.Period("2025-06")

    cases_m = cases[cases["_month"] == target]
    comps_m = complaints[complaints["_month"] == target]

    if cases_m.empty or comps_m.empty:
        st.info("No rows returned for June 2025.")
        return

    c_cases = cases_m.groupby(["portfolio"], dropna=False).size().rename("cases")
    c_comps = comps_m.groupby(["portfolio"], dropna=False).size().rename("complaints")
    df = pd.concat([c_cases, c_comps], axis=1).fillna(0).reset_index()
    df["per_1000"] = df.apply(lambda r: (r["complaints"] / r["cases"] * 1000.0) if r["cases"] else 0.0, axis=1)

    # totals row
    totals = pd.DataFrame(
        {
            "portfolio": ["Total"],
            "cases": [df["cases"].sum()],
            "complaints": [df["complaints"].sum()],
            "per_1000": [(df["complaints"].sum() / df["cases"].sum() * 1000.0) if df["cases"].sum() else 0.0],
        }
    )
    out = pd.concat([df.sort_values("portfolio"), totals], ignore_index=True)

    st.dataframe(out, use_container_width=True)
