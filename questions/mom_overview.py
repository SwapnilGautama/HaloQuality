from __future__ import annotations
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

# ---------- helpers ----------
def _common_months(*dfs: pd.DataFrame) -> list[str]:
    def mset(df):
        if df is None or df.empty or "month" not in df.columns: return set()
        return set(df["month"].dropna().astype(str))
    sets = [mset(d) for d in dfs if not (d is None or d.empty)]
    sets = [s for s in sets if s]
    if not sets: return []
    common = set.intersection(*sets) if len(sets) > 1 else sets[0]
    return sorted(common)

def _pivot_long_to_wide(df: pd.DataFrame, index="month", columns="Portfolio_std", values="value"):
    if df.empty: 
        return pd.DataFrame()
    out = df.pivot_table(index=index, columns=columns, values=values, aggfunc="first")
    out = out.sort_index()
    return out

def _available_dims(complaints: pd.DataFrame, cases: pd.DataFrame, survey: pd.DataFrame) -> list[str]:
    # Only show tabs for dims that exist in (at least) complaints & cases (needed for per-1000)
    candidates = ["Portfolio_std", "Team", "Work_Type", "Processor_std", "Checker_std", "Location", "Portfolio"]  # flexible
    cols_in_complaints = set(complaints.columns) if not complaints.empty else set()
    cols_in_cases      = set(cases.columns) if not cases.empty else set()
    # Need dim in both complaints and cases for per-1000 calc
    return [c for c in candidates if (c in cols_in_complaints and c in cols_in_cases)]

# ---------- core ----------
def run(store: dict[str, pd.DataFrame], dims: list[str] | None = None):
    complaints = store["complaints"]
    cases      = store["cases"]
    survey     = store["survey"]

    # Basic presence checks
    needed_c = {"month"}
    needed_k = {"Case ID","month"}
    needed_s = {"month","NPS"}
    if complaints.empty or not needed_c.issubset(complaints.columns):
        st.warning("Complaints data (with a `month` column) is required.")
        return
    if cases.empty or not needed_k.issubset(cases.columns):
        st.warning("Cases data (with `Case ID` and `month`) is required.")
        return
    if survey.empty or not needed_s.issubset(survey.columns):
        st.warning("Survey data (with `NPS` and `month`) is required.")
        return

    # Limit to months common across all three datasets
    commons = _common_months(complaints, cases, survey)
    if not commons:
        st.warning("No overlapping months across complaints, cases, and survey.")
        return
    complaints = complaints[complaints["month"].astype(str).isin(commons)].copy()
    cases      = cases[cases["month"].astype(str).isin(commons)].copy()
    survey     = survey[survey["month"].astype(str).isin(commons)].copy()

    # ---------- OVERALL month-on-month ----------
    comp_m = complaints.groupby("month", dropna=False).size().reset_index(name="Complaints")
    uniq_m = cases.groupby("month", dropna=False)["Case ID"].nunique().reset_index(name="Unique_Cases")
    nps_m  = survey.groupby("month", dropna=False)["NPS"].mean().reset_index()

    overall = comp_m.merge(uniq_m, on="month", how="outer").merge(nps_m, on="month", how="outer").fillna(0)
    overall["Complaints_per_1000"] = np.where(
        overall["Unique_Cases"] > 0,
        (overall["Complaints"] / overall["Unique_Cases"]) * 1000.0,
        np.nan
    )
    overall = overall.sort_values("month")

    st.subheader("Overall month-on-month")
    # combined bar (complaints/1000) + line (NPS)
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=overall["month"], y=overall["Complaints_per_1000"],
        name="Complaints per 1,000", yaxis="y1"
    ))
    fig.add_trace(go.Scatter(
        x=overall["month"], y=overall["NPS"],
        name="NPS", mode="lines+markers", yaxis="y2"
    ))
    fig.update_layout(
        height=420, margin=dict(l=6,r=6,t=6,b=6),
        yaxis=dict(title="Complaints per 1,000"),
        yaxis2=dict(title="NPS", overlaying="y", side="right"),
        xaxis=dict(title="Month")
    )
    st.plotly_chart(fig, use_container_width=True)

    # ---------- Tables (overall) ----------
    st.markdown("**Tables — overall**")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.caption("Complaints (sum)")
        st.dataframe(overall[["month","Complaints"]].set_index("month"))
    with c2:
        st.caption("NPS (mean)")
        st.dataframe(overall[["month","NPS"]].set_index("month"))
    with c3:
        st.caption("Complaints / 1,000")
        st.dataframe(overall[["month","Complaints_per_1000"]].set_index("month"))

    # ---------- By dimension tabs ----------
    dims_in_data = dims or _available_dims(complaints, cases, survey)
    if not dims_in_data:
        return

    st.subheader("Breakdown by dimension")
    tabs = st.tabs(dims_in_data)

    for tab, dim in zip(tabs, dims_in_data):
        with tab:
            # Complaints by month x dim
            comp_g = complaints.groupby(["month", dim], dropna=False).size().reset_index(name="Complaints")
            comp_w = _pivot_long_to_wide(comp_g.rename(columns={"Complaints":"value"}), index="month", columns=dim, values="value")

            # Unique cases by month x dim
            uniq_g = cases.groupby(["month", dim], dropna=False)["Case ID"].nunique().reset_index(name="Unique_Cases")

            # NPS by month x dim
            if dim in survey.columns:
                nps_g = survey.groupby(["month", dim], dropna=False)["NPS"].mean().reset_index()
            else:
                # If dim missing in survey, fallback to overall NPS (same for every dim)
                nps_g = survey.groupby(["month"], dropna=False)["NPS"].mean().reset_index()
                nps_g[dim] = "Overall"

            # Complaints per 1000 by month x dim
            base = comp_g.merge(uniq_g, on=["month", dim], how="outer").fillna(0)
            base["Complaints_per_1000"] = np.where(
                base["Unique_Cases"] > 0,
                (base["Complaints"] / base["Unique_Cases"]) * 1000.0,
                np.nan
            )
            cpk_w = _pivot_long_to_wide(
                base[["month", dim, "Complaints_per_1000"]].rename(columns={"Complaints_per_1000":"value"}),
                index="month", columns=dim, values="value"
            )

            # NPS wide
            nps_w = _pivot_long_to_wide(
                nps_g.rename(columns={"NPS":"value"}),
                index="month", columns=dim, values="value"
            )

            # Render three tables
            t1, t2, t3 = st.columns(3)
            with t1:
                st.caption(f"Complaints — by {dim}")
                st.dataframe(comp_w, use_container_width=True)
            with t2:
                st.caption(f"NPS — by {dim}")
                st.dataframe(nps_w, use_container_width=True)
            with t3:
                st.caption(f"Complaints / 1,000 — by {dim}")
                st.dataframe(cpk_w, use_container_width=True)

            # Optional: a per-dimension trend chart (stacked complaints/1000 + NPS overall line)
            # Quick overall for this dim (weighted by cases)
            monthly_tot = base.groupby("month", dropna=False)[["Complaints", "Unique_Cases"]].sum().reset_index()
            monthly_tot["Complaints_per_1000"] = np.where(
                monthly_tot["Unique_Cases"] > 0,
                (monthly_tot["Complaints"] / monthly_tot["Unique_Cases"]) * 1000.0,
                np.nan
            )
            nps_overall = survey.groupby("month", dropna=False)["NPS"].mean().reset_index()

            merged = monthly_tot.merge(nps_overall, on="month", how="left").sort_values("month")
            fig2 = go.Figure()
            fig2.add_trace(go.Bar(x=merged["month"], y=merged["Complaints_per_1000"], name="Complaints per 1,000", yaxis="y1"))
            fig2.add_trace(go.Scatter(x=merged["month"], y=merged["NPS"], name="NPS", mode="lines+markers", yaxis="y2"))
            fig2.update_layout(
                height=380, margin=dict(l=6,r=6,t=6,b=6),
                yaxis=dict(title="Complaints per 1,000"),
                yaxis2=dict(title="NPS", overlaying="y", side="right"),
                xaxis=dict(title="Month"),
                title=f"Trend — {dim} (overall)"
            )
            st.plotly_chart(fig2, use_container_width=True)
