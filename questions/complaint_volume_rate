# questions/question_q4.py
from __future__ import annotations
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

    if params.month_range:
        rng = set(_month_range(params.month_range))
        cases = cases[cases["month"].isin(rng)]
        comps = comps[comps["month"].isin(rng)]
    elif params.last_n:
        inter = sorted(set(cases["month"]).intersection(set(comps["month"])))
        inter = inter[-params.last_n:]
        cases = cases[cases["month"].isin(inter)]
        comps  = comps[comps["month"].isin(inter)]

    # choose dimension based on filter presence or default to portfolio
    dim = "Portfolio_std"
    if params.process: dim = "ProcessName" if "ProcessName" in cases.columns else dim
    if dim not in cases.columns and "Portfolio_std" in cases.columns:
        dim = "Portfolio_std"

    # restrict if user already filtered by portfolio/process
    if params.portfolio and "Portfolio_std" in cases.columns:
        cases = cases[cases["Portfolio_std"].str.casefold()==params.portfolio.casefold()]
        comps = comps[comps["Portfolio_std"].str.casefold()==params.portfolio.casefold()]
    if params.process:
        if "ProcessName" in cases.columns:
            cases = cases[cases["ProcessName"].str.casefold()==params.process.casefold()]
        if "Parent_Case_Type" in comps.columns:
            comps  = comps[comps["Parent_Case_Type"].str.casefold()==params.process.casefold()]

    case_key = "Case ID" if "Case ID" in cases.columns else "CaseID"
    c_cases = (cases.drop_duplicates([case_key]).groupby([dim,"month"]).size()
               .rename("unique_cases").reset_index())
    c_comp  = comps.groupby([dim,"month"]).size().rename("complaints").reset_index()

    j = pd.merge(c_cases, c_comp, on=[dim,"month"], how="outer").fillna(0)
    if j.empty:
        st.info("No data for the selection.")
        return
    j["complaints_per_1000"] = j["complaints"]/j["unique_cases"].where(j["unique_cases"]>0, pd.NA)*1000
    st.subheader(f"Complaints & complaints/1000 by {dim}")
    st.dataframe(j.sort_values([dim,"month"]), use_container_width=True)
