from __future__ import annotations
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

def run(store: dict[str, pd.DataFrame], month: str, group_by: list[str] = ["Portfolio_std"]):
    complaints = store["complaints"]
    cases      = store["cases"]
    survey     = store["survey"]

    # guards
    if complaints.empty or not {"month","Portfolio_std"}.issubset(complaints.columns):
        st.warning("Complaints data with `month` and a Portfolio column is required under `data/complaints/`.")
        return
    if cases.empty or not {"Case ID","month","Portfolio_std"}.issubset(cases.columns):
        st.warning("Cases data (with `Case ID`) is required under `data/cases/`.")
        return
    if survey.empty or not {"month","Portfolio_std","NPS"}.issubset(survey.columns):
        st.warning("Survey/NPS data is required under `data/surveys/` (direct NPS or Promoters/Passives/Detractors).")
        return

    comp_g = (complaints[complaints["month"] == month]
              .groupby(group_by, dropna=False).size().reset_index(name="Complaints"))

    cas_m = cases[cases["month"] == month].copy()
    cas_g = cas_m.groupby(group_by, dropna=False)["Case ID"].nunique().reset_index(name="Unique_Cases")

    base = comp_g.merge(cas_g, on=group_by, how="inner")
    base = base[base["Unique_Cases"] > 0]
    if base.empty:
        st.warning(f"No overlapping data for {month}.")
        return

    base["Complaints_per_1000"] = (base["Complaints"] / base["Unique_Cases"]) * 1000.0

    nps_g = survey[survey["month"] == month].groupby(group_by, dropna=False)["NPS"].mean().reset_index()
    df = base.merge(nps_g, on=group_by, how="inner").dropna(subset=["NPS"])
    if df.empty:
        st.warning("No groups have both Complaints rate and NPS for the chosen month.")
        return

    r = np.corrcoef(df["Complaints_per_1000"], df["NPS"])[0, 1]
    slope, intercept = np.polyfit(df["Complaints_per_1000"], df["NPS"], 1)
    xs = np.linspace(df["Complaints_per_1000"].min(), df["Complaints_per_1000"].max(), 50)
    ys = slope * xs + intercept
    direction = "negative" if r < 0 else "positive"
    strength = "strong" if abs(r) >= 0.7 else "moderate" if abs(r) >= 0.4 else "weak"

    st.subheader(f"Complaints vs NPS — {month}")
    st.write(f"**Correlation:** {r:.2f} ({strength}, {direction}) • Groups: {len(df)}")

    fig = px.scatter(
        df, x="Complaints_per_1000", y="NPS",
        hover_data=group_by, labels={"Complaints_per_1000":"Complaints per 1,000 cases"}
    )
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name="trend"))
    fig.update_layout(height=460, margin=dict(l=6,r=6,t=6,b=6))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("**By portfolio**")
    st.dataframe(df.sort_values("Complaints_per_1000", ascending=False),
                 hide_index=True, use_container_width=True)

