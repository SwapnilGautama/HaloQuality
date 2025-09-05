# questions/question_q5.py
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
    fpa = store.get("fpa")
    if fpa is None or fpa.empty:
        st.info("No FPA data loaded.")
        return

    # detect the date column present in FPA
    date_col = "Review_Date" if "Review_Date" in fpa.columns else fpa.columns[0]
    fpa = _ensure_month(fpa, date_col)

    rr_col = "Review_Result" if "Review_Result" in fpa.columns else "review_result"
    if rr_col not in fpa.columns:
        st.error("FPA is missing 'Review_Result'.")
        return

    # pick dimension requested or default
    dim_map = {
        "team": ["Team","Parent Team","TeamName","Parent_Team"],
        "manager": ["Team_Manager","Manager","Team Manager"],
        "location": ["Location","Site"],
        "scheme": ["Scheme","ClientName Scheme","ClientName_Scheme"],
        "portfolio": ["Portfolio_std","Portfolio"],
        "process": ["ProcessName","Parent_Case_Type"],
    }
    dim = "Portfolio_std"
    if params.by_dim:
        for c in dim_map.get(params.by_dim, []):
            if c in fpa.columns:
                dim = c; break

    if params.portfolio and "Portfolio_std" in fpa.columns:
        fpa = fpa[fpa["Portfolio_std"].str.casefold()==params.portfolio.casefold()]
    if params.process and "ProcessName" in fpa.columns:
        fpa = fpa[fpa["ProcessName"].str.casefold()==params.process.casefold()]

    if params.month_range:
        fpa = fpa[fpa["month"].isin(_month_range(params.month_range))]
    elif params.last_n:
        fpa = fpa[fpa["month"].isin(sorted(fpa["month"].unique())[-params.last_n:])]

    grp = fpa.groupby(dim)
    out = grp.apply(lambda d: (d[rr_col].str.lower()=="fail").mean()).rename("fail_rate").reset_index()
    out = out.sort_values("fail_rate", ascending=False)

    st.subheader(f"FPA fail rate by {dim}")
    st.bar_chart(out.set_index(dim)["fail_rate"])
    with st.expander("Data", expanded=False):
        st.dataframe(out)
