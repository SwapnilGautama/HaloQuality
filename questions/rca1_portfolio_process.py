import streamlit as st
import pandas as pd

TITLE = "RCA1 by Portfolio × Process"

def run(store: dict, params: dict):
    st.subheader(TITLE)

    comp = store["complaints"].copy()
    # Expect RCA fields already labeled during ingestion; if not present, show hint
    rca_col = "RCA1"
    if rca_col not in comp.columns:
        st.info("RCA labels not found. Please run the complaints labeller so 'RCA1' exists.")
        return

    # filter by last N months if provided
    date_from = params.get("date_from")
    date_to   = params.get("date_to")
    if date_from is not None:
        comp = comp.loc[comp["month_dt"] >= date_from]
    if date_to is not None:
        comp = comp.loc[comp["month_dt"] <= date_to]

    # group
    k = ["Portfolio_std","Process Name", rca_col]
    df = (comp
          .dropna(subset=["Portfolio_std","Process Name", rca_col])
          .groupby(k, dropna=False)
          .size()
          .rename("complaints")
          .reset_index())

    if df.empty:
        st.info("No complaint RCA found in the selected period.")
        return

    # show top RCAs per portfolio × process
    top = (df.sort_values("complaints", ascending=False)
             .groupby(["Portfolio_std","Process Name"], as_index=False)
             .head(5))
    st.dataframe(top, use_container_width=True)
