# questions/first_pass_accuracy.py
from __future__ import annotations

import os
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

# << NEW >>
from core.reason_labeller import classify_comments, OPENAI_READY

# -------------------------------------------------------------
# Helpers (unchanged behaviour from your last working version)
# -------------------------------------------------------------
def _load_excel(root: str) -> pd.DataFrame:
    """
    Loads the FirstPassAccuracy workbook the same way your app did before.
    Expecting columns at least:
      - 'Activity Date' (date)
      - 'Review Result' (Pass/Fail)
      - 'Portfolio'
      - 'Scheme'
      - 'Case Comment' (free text)
    """
    # Your existing path convention was data/first_pass_accuracy/FirstPassAccuracy_*.xlsx
    # but we preserve the last working single-file behaviour as well.
    candidate_dirs = [
        os.path.join(root, "data", "first_pass_accuracy"),
        os.path.join(root, "data"),
    ]
    excel_path = None
    for d in candidate_dirs:
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            if f.lower().startswith("firstpassaccuracy") and f.lower().endswith(".xlsx"):
                excel_path = os.path.join(d, f)
    if not excel_path:
        # fall back to the root if someone put it there
        for f in os.listdir(root):
            if f.lower().startswith("firstpassaccuracy") and f.lower().endswith(".xlsx"):
                excel_path = os.path.join(root, f)

    if not excel_path:
        st.error("Could not find a FirstPassAccuracy workbook (FirstPassAccuracy*.xlsx).")
        return pd.DataFrame()

    try:
        df = pd.read_excel(excel_path)
    except Exception as e:
        st.error(f"Failed to read workbook: {excel_path}\n{e}")
        return pd.DataFrame()

    # Normalize columns
    rename = {c.lower().strip(): c for c in df.columns}
    # make a lower-name to original map
    lower_map = {c.lower().strip(): c for c in df.columns}

    def pick(name: str) -> str:
        # try exact, then case-insensitive
        if name in df.columns:
            return name
        if name.lower() in lower_map:
            return lower_map[name.lower()]
        return name

    # Standardize
    act_col = pick("Activity Date")
    res_col = pick("Review Result")
    port_col = pick("Portfolio")
    scheme_col = pick("Scheme")
    comm_col = pick("Case Comment")

    # coerce types
    if act_col in df.columns:
        df[act_col] = pd.to_datetime(df[act_col], errors="coerce")
    if res_col in df.columns:
        df[res_col] = df[res_col].astype(str).str.strip()
    for c in [port_col, scheme_col, comm_col]:
        if c in df.columns:
            df[c] = df[c].astype(str).fillna("")

    # keep only essential cols
    keep = [c for c in [act_col, res_col, port_col, scheme_col, comm_col] if c in df.columns]
    return df[keep].copy()

def _month_floor(dt: pd.Series) -> pd.Series:
    return (dt.values.astype("datetime64[M]")).astype("datetime64[ns]")

def _pass_percent(v: pd.Series) -> float:
    if v.empty:
        return 0.0
    return (v.str.lower().eq("pass").sum() / len(v)) * 100.0

# -------------------------------------------------------------
# Pareto calcs (unchanged)
# -------------------------------------------------------------
def _pareto_top80(counts: pd.Series) -> pd.DataFrame:
    if counts.empty:
        return pd.DataFrame(columns=["reason", "count", "percent", "cum_percent"])
    df = counts.reset_index()
    df.columns = ["reason", "count"]
    df = df.sort_values("count", ascending=False).reset_index(drop=True)
    total = df["count"].sum()
    df["percent"] = (df["count"] / total) * 100
    df["cum_percent"] = df["percent"].cumsum()
    # Keep rows up to 80%, then a single "Other" row (if needed)
    head = df[df["cum_percent"] <= 80.0]
    tail = df[df["cum_percent"] > 80.0]
    if not tail.empty:
        other_row = pd.DataFrame(
            [["Other", tail["count"].sum(), (tail["count"].sum() / total) * 100, 100.0]],
            columns=["reason", "count", "percent", "cum_percent"],
        )
        out = pd.concat([head, other_row], ignore_index=True)
    else:
        out = df.copy()
    return out

# -------------------------------------------------------------
# R E N D E R
# -------------------------------------------------------------
def run(store: Dict[str, Any], params: Dict[str, Any], q: str) -> None:
    """
    Question 2 runner (compatible with your current engine).
    Only the fail reason classification block was replaced.
    """
    root = store.get("root", os.getcwd())
    df = _load_excel(root)
    if df.empty:
        return

    # Column pickers (as loaded)
    act_col = df.columns[0]  # Activity Date
    res_col = df.columns[1]  # Review Result
    port_col = df.columns[2]  # Portfolio
    scheme_col = df.columns[3]  # Scheme
    comm_col = df.columns[4]  # Case Comment

    # Month-level MoM
    df["_month"] = _month_floor(df[act_col])
    mom = (
        df.groupby("_month")[res_col]
        .apply(_pass_percent)
        .reset_index(name="pass_%")
        .sort_values("_month")
    )

    st.subheader("First-Pass Accuracy — Jan–Most Recent")
    c1, c2 = st.columns([2, 3], gap="large")

    with c1:
        if not mom.empty:
            fig = px.line(
                mom, x="_month", y="pass_%", markers=True,
                labels={"_month": "", "pass_%": "Pass %"}
            )
            fig.update_traces(mode="lines+markers")
            fig.update_yaxes(showgrid=False, visible=False)
            fig.update_xaxes(showgrid=False)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No month-level data available.")

    # Pass % by Portfolio × Scheme (latest month)
    latest_month = mom["_month"].max() if not mom.empty else None
    if latest_month is not None:
        latest_df = df[df["_month"].eq(latest_month)].copy()
    else:
        latest_df = df.copy()

    with c2:
        if not latest_df.empty:
            grp = latest_df.groupby([port_col, scheme_col])[res_col].agg(
                cases="count", pass_="sum"
            )
            # pass_ above is wrong — fix to count "Pass"
            grp = (
                latest_df
                .assign(_pass=latest_df[res_col].str.lower().eq("pass").astype(int))
                .groupby([port_col, scheme_col])["_pass"]
                .agg(cases="count", pass_="sum")
                .reset_index()
            )
            grp["pass_%"] = (grp["pass_"] / grp["cases"]) * 100
            grp = grp.drop(columns=["pass_"]).sort_values(["pass_%", "cases"], ascending=[False, False])
            st.caption(f"Pass % by Portfolio × Scheme — {pd.to_datetime(latest_month).strftime('%b-%y') if latest_month is not None else ''}")
            st.dataframe(grp, use_container_width=True, height=360)
        else:
            st.info("No latest-month data to show by Portfolio × Scheme.")

    st.markdown("---")
    st.subheader(f"Reasons for Fail — {pd.to_datetime(latest_month).strftime('%b-%y') if latest_month is not None else ''}")

    if not OPENAI_READY:
        st.info("OpenAI labelling inactive (no OPENAI_API_KEY). Using keyword model only; results may include 'Other'.")

    # --------- NEW: classification (keyword first, GPT refine optionally) ----------
    fails = latest_df[latest_df[res_col].str.lower().eq("fail")].copy()
    if fails.empty:
        st.info("No failed cases in the selected period.")
        return

    # Classify comments
    reasons = classify_comments(fails[comm_col])

    # Pareto (top 80% + Other)
    counts = reasons.value_counts(dropna=False)
    pareto_df = _pareto_top80(counts)

    # Side-by-side chart + table (unchanged)
    b1, b2 = st.columns([3, 2], gap="large")
    with b1:
        fig2 = px.bar(
            pareto_df, x="reason", y="count", text="count",
            labels={"reason": "", "count": ""},
            title="Fail reasons — Pareto (top 80% + Other)"
        )
        fig2.update_traces(textposition="outside")
        fig2.update_yaxes(showgrid=False, visible=False)
        fig2.update_xaxes(showgrid=False)
        st.plotly_chart(fig2, use_container_width=True)

    with b2:
        st.dataframe(
            pareto_df[["reason", "count", "percent", "cum_percent"]],
            use_container_width=True,
            height=380
        )
