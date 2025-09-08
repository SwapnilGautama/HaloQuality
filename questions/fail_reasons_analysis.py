# questions/fail_reasons_analysis.py
# New FRA question – analyses Fail reasons using the hybrid labeller.
# Keeps the same visual style and "Top 80% + Other" Pareto.

from __future__ import annotations

import re
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# Use the shared labeller
from core.reason_labeller import (
    label_dataframe,
    get_or_fit_model,
)

# ------------------------------------------------------------
# Helpers: resilient access to the "First Pass Accuracy" frame
# ------------------------------------------------------------

def _first(df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    return df if isinstance(df, pd.DataFrame) and not df.empty else None

def _get_df_from_store(store: Dict) -> Optional[pd.DataFrame]:
    """
    Try common locations/shapes used by the app for FPA data.
    This is deliberately permissive so Q1/Q2 remain isolated.
    """
    for k in ("fpa", "fpa_df", "df_fpa", "first_pass_accuracy", "fpa_data"):
        if k in store and isinstance(store[k], pd.DataFrame):
            return _first(store[k])

    # Fallback: if a single file was loaded into store["files"] or similar
    for k in store.keys():
        v = store.get(k)
        if isinstance(v, pd.DataFrame):
            cols = set(map(str.lower, v.columns))
            # Basic signature check
            if {"activity date", "review result"}.issubset(cols):
                return _first(v)

    return None


# ------------------------------------------------------------
# Business helpers
# ------------------------------------------------------------

_FAIL_REVIEW_VALUES = {"fail", "failed", "no"}  # include common spellings if present

def _is_fail(val: str) -> bool:
    if not isinstance(val, str):
        return False
    return val.strip().lower() in _FAIL_REVIEW_VALUES

def _month_key(dt: pd.Timestamp) -> str:
    return f"{dt.year}-{dt.month:02d}"

def _coerce_date(s: pd.Series) -> pd.Series:
    try:
        return pd.to_datetime(s, errors="coerce", dayfirst=True)
    except Exception:
        return pd.to_datetime(s, errors="coerce")


def _pareto_top80(df_counts: pd.DataFrame) -> Tuple[pd.DataFrame, plt.Figure]:
    """
    Build Top 80% + Other and render a bar chart figure.
    """
    if df_counts.empty:
        return df_counts, plt.figure()

    df = df_counts.copy()
    df = df.sort_values("count", ascending=False).reset_index(drop=True)
    total = df["count"].sum()
    df["percent"] = (df["count"] / total * 100).round(1)
    df["cum_percent"] = df["percent"].cumsum().round(1)

    # Split at 80%
    head = df[df["cum_percent"] <= 80.0]
    if head.empty:
        head = df.iloc[:1, :]
    tail = df.loc[~df.index.isin(head.index), :]
    if not tail.empty:
        other_row = pd.DataFrame([{
            "reason": "Other",
            "count": int(tail["count"].sum()),
            "percent": round(tail["percent"].sum(), 1),
            "cum_percent": 100.0
        }])
        pareto = pd.concat([head, other_row], ignore_index=True)
    else:
        pareto = head.copy()
        pareto.loc[pareto.index[-1], "cum_percent"] = 100.0

    # Plot
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(pareto["reason"], pareto["count"])
    ax.set_title("Fail reasons — Pareto (top 80% + Other)")
    ax.set_ylabel("count")
    ax.set_xlabel("")
    ax.set_ylim(0, pareto["count"].max() * 1.15)
    for i, v in enumerate(pareto["count"].tolist()):
        ax.text(i, v + max(1, pareto["count"].max() * 0.02), f"{v}", ha="center", va="bottom")
    ax.tick_params(axis='x', rotation=90)
    fig.tight_layout()
    return pareto, fig


# ------------------------------------------------------------
# Streamlit entry point
# ------------------------------------------------------------

def run(store: Dict, params: Dict, q: str) -> None:
    """
    Render "Fail reasons analysis" (FRA) – independent of Q1/Q2.
    """
    st.header("Reasons for Fail — No data")

    df_all = _get_df_from_store(store)
    if df_all is None:
        st.info("Could not find a First-Pass Accuracy dataset in the store.")
        return

    # Required columns
    needed = {"Activity Date", "Review Result", "Case Comment"}
    missing = [c for c in needed if c not in df_all.columns]
    if missing:
        st.error(f"Dataset is missing required columns: {', '.join(missing)}")
        return

    # Prepare
    df = df_all.copy()
    df["Activity Date"] = _coerce_date(df["Activity Date"])
    df = df.dropna(subset=["Activity Date"])
    df["month"] = df["Activity Date"].dt.to_period("M").astype(str)
    df["is_fail"] = df["Review Result"].apply(_is_fail)

    # Focus on most recent month available
    if df["month"].empty:
        st.info("No dated rows available.")
        return
    latest_month = df["month"].max()
    df_m = df[df["month"] == latest_month]
    df_fail = df_m[df_m["is_fail"]]

    st.header(f"Reasons for Fail — {pd.Period(latest_month).strftime('%b-%y').title()}")

    if df_fail.empty:
        st.info(f"No Fail rows found for {latest_month}.")
        return

    # Label reasons (Rules → RCA2 → ML)
    model_bundle = get_or_fit_model(df_all, text_col="Case Comment", rca2_col="RCA2")
    reasons = label_dataframe(df_fail, text_col="Case Comment", rca2_col="RCA2", model_bundle=model_bundle)
    df_fail = df_fail.assign(reason=reasons)

    # Build counts & Pareto
    counts = df_fail.groupby("reason", dropna=False).size().reset_index(name="count")
    pareto_df, fig = _pareto_top80(counts)

    # Layout: chart + table
    col1, col2 = st.columns([2, 1])
    with col1:
        st.pyplot(fig, clear_figure=True)
    with col2:
        st.dataframe(pareto_df, use_container_width=True)
