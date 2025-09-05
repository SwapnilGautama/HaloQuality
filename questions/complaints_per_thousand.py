import streamlit as st
import pandas as pd

TITLE = "Complaints per 1,000 cases"

def run(store: dict, params: dict):
    st.subheader(TITLE)

    cases = store["cases"]
    comp  = store["complaints"]

    # ---- filter early by user intent (optional)
    portfolio = params.get("portfolio")
    process   = params.get("process")
    date_from = params.get("date_from")   # Timestamp or None
    date_to   = params.get("date_to")     # Timestamp or None

    if portfolio and "Portfolio_std" in cases:
        cases = cases.loc[cases["Portfolio_std"].astype(str).str.casefold().eq(str(portfolio).casefold())]
        comp  = comp.loc[comp["Portfolio_std"].astype(str).str.casefold().eq(str(portfolio).casefold())]

    if process and "Process Name" in cases:
        cases = cases.loc[cases["Process Name"].astype(str).str.casefold().eq(str(process).casefold())]
        comp  = comp.loc[comp["Process Name"].astype(str).str.casefold().eq(str(process).casefold())]

    if date_from is not None:
        cases = cases.loc[cases["month_dt"] >= date_from]
        comp  = comp.loc[comp["month_dt"]  >= date_from]
    if date_to is not None:
        cases = cases.loc[cases["month_dt"] <= date_to]
        comp  = comp.loc[comp["month_dt"]  <= date_to]

    # ---- monthly agg
    k = ["month_dt","month"]
    cases_m = cases.groupby(k, dropna=False)["Case ID"].nunique().rename("cases").reset_index()
    comp_m  = comp.groupby(k,  dropna=False)["Case ID"].nunique().rename("complaints").reset_index()

    # inner join by month only — prevents cartesian
    df = pd.merge(cases_m, comp_m, on=k, how="inner")
    if df.empty:
        st.info("No overlapping data for cases and complaints.")
        return

    df["complaints_per_1000"] = (df["complaints"] / df["cases"]) * 1000

    st.line_chart(
        df.sort_values("month_dt").set_index("month_dt")[["complaints_per_1000"]],
        height=280
    )
    st.dataframe(
        df.sort_values("month_dt")[["month","cases","complaints","complaints_per_1000"]].rename(columns={"month":"Month"}),
        use_container_width=True
    )
