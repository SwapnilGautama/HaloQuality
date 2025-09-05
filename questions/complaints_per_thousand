# questions/question_q1.py
from __future__ import annotations
import math
import pandas as pd
import streamlit as st

def _ensure_month(df: pd.DataFrame, date_col: str, out_col="month"):
    out = df.copy()
    if out_col not in out.columns:
        out[out_col] = pd.to_datetime(out[date_col]).dt.strftime("%Y-%m")
    return out

def _month_range(rr):
    s,e = rr
    s2 = pd.to_datetime(s+"-01"); e2 = pd.to_datetime(e+"-01")
    return [d.strftime("%Y-%m") for d in pd.date_range(s2, e2, freq="MS")]

def run(store, params):
    cases = store["cases"].copy()
    comps = store["complaints"].copy()

    cases = _ensure_month(cases, "Create Date" if "Create Date" in cases.columns else "Create_Date")
    comps = _ensure_month(comps, "Report_Date" if "Report_Date" in comps.columns else "Report Date")

    if params.portfolio:
        cases = cases[cases["Portfolio_std"].str.casefold()==params.portfolio.casefold()]
        comps = comps[comps["Portfolio_std"].str.casefold()==params.portfolio.casefold()]
    if params.process:
        if "ProcessName" in cases.columns:
            cases = cases[cases["ProcessName"].str.casefold()==params.process.casefold()]
        if "Parent_Case_Type" in comps.columns:
            comps = comps[comps["Parent_Case_Type"].str.casefold()==params.process.casefold()]

    if params.month_range:
        rng = set(_month_range(params.month_range))
        cases = cases[cases["month"].isin(rng)]
        comps = comps[comps["month"].isin(rng)]
    elif params.last_n:
        inter = sorted(set(cases["month"]).intersection(set(comps["month"])))
        inter = inter[-params.last_n:]
        cases = cases[cases["month"].isin(inter)]
        comps  = comps[comps["month"].isin(inter)]

    # unique case ids per month
    case_key = "Case ID" if "Case ID" in cases.columns else "CaseID"
    case_m = (cases.drop_duplicates([case_key]).groupby("month").size()
              .rename("unique_cases").reset_index())
    comp_m = comps.groupby("month").size().rename("complaints").reset_index()
    joined = pd.merge(case_m, comp_m, on="month", how="inner")
    if joined.empty:
        st.info("No overlapping months for the filters.")
        return

    joined["complaints_per_1000"] = joined["complaints"] / joined["unique_cases"] * 1000
    st.subheader("Complaints per 1,000 cases (MoM)")
    st.line_chart(joined.set_index("month")[["complaints_per_1000"]])

    with st.expander("Data", expanded=False):
        st.dataframe(joined.sort_values("month"))
