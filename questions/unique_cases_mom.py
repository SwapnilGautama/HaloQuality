# questions/question_q3.py
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
    cases = _ensure_month(cases, "Create Date" if "Create Date" in cases.columns else "Create_Date")

    if params.portfolio:
        cases = cases[cases["Portfolio_std"].str.casefold()==params.portfolio.casefold()]
    if params.process and "ProcessName" in cases.columns:
        cases = cases[cases["ProcessName"].str.casefold()==params.process.casefold()]

    if params.month_range:
        cases = cases[cases["month"].isin(_month_range(params.month_range))]
    elif params.last_n:
        cases = cases[cases["month"].isin(sorted(cases["month"].unique())[-params.last_n:])]

    key = ["month"]
    if "ProcessName" in cases.columns: key.append("ProcessName")
    if "Portfolio_std" in cases.columns: key.append("Portfolio_std")

    case_key = "Case ID" if "Case ID" in cases.columns else "CaseID"
    out = (cases.drop_duplicates([case_key]).groupby(key).size()
           .rename("unique_cases").reset_index().sort_values(key))
    st.subheader("Unique cases (MoM)")
    st.dataframe(out, use_container_width=True)
