# questions/question_q2.py
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
    df = store["complaints"].copy()
    df = _ensure_month(df, "Report_Date" if "Report_Date" in df.columns else "Report Date")

    if params.process and "Parent_Case_Type" in df.columns:
        df = df[df["Parent_Case_Type"].str.casefold()==params.process.casefold()]
    if params.month_range:
        df = df[df["month"].isin(_month_range(params.month_range))]
    elif params.last_n:
        months = sorted(df["month"].unique())[-params.last_n:]
        df = df[df["month"].isin(months)]

    if "RCA1" not in df.columns:
        st.info("RCA1 not found in complaints.")
        return

    piv = (df.pivot_table(index="Portfolio_std", columns="RCA1",
                          values="Parent_Case_Type", aggfunc="count", fill_value=0)
           .sort_index())
    if piv.empty:
        st.info("No complaints for the selected filters.")
        return

    st.subheader("RCA1 mix by portfolio")
    st.bar_chart((piv.T / piv.sum(axis=1)).T.fillna(0.0))
    with st.expander("Data", expanded=False):
        st.dataframe((piv.T / piv.sum(axis=1)).T.round(3))
