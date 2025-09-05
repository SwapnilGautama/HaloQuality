import streamlit as st
import pandas as pd

TITLE = "Complaint volume (MoM)"

def run(store: dict, params: dict):
    st.subheader(TITLE)
    comp = store["complaints"]

    date_from = params.get("date_from")
    date_to   = params.get("date_to")
    if date_from is not None:
        comp = comp.loc[comp["month_dt"] >= date_from]
    if date_to is not None:
        comp = comp.loc[comp["month_dt"] <= date_to]

    k = ["month_dt","month"]
    df = comp.groupby(k, dropna=False)["Case ID"].nunique().rename("complaints").reset_index()
    if df.empty:
        st.info("No complaints in the selected period.")
        return

    st.bar_chart(df.sort_values("month_dt").set_index("month_dt")["complaints"], height=280)
    st.dataframe(df.sort_values("month_dt").rename(columns={"month":"Month"}), use_container_width=True)
