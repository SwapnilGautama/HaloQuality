# -*- coding: utf-8 -*-
# questions/first_pass_accuracy.py
#
# Renders First Pass Accuracy analysis (Jan-2025 .. latest).
# - Loads the *real* file from data/first_pass_accuracy/
#   by globbing for "FirstPassAccuracy*.xlsx"
# - Keeps fast thanks to st.cache_data
# - Charts are clean (no gridlines / y-axis)
# - Robust column handling
#
from __future__ import annotations
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# --------------------------------------------------------------------------------------
# File discovery (robust to apostrophes, spaces and different suffixes)
# --------------------------------------------------------------------------------------
def _find_fpa_workbook(root: Path) -> Path:
    folder = root / "data" / "first_pass_accuracy"
    if not folder.exists():
        raise FileNotFoundError(f"Folder missing: {folder}")

    # Try best-pattern first
    cand = sorted(folder.glob("FirstPassAccuracy*.*"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not cand:
        # Fallback: any xlsx
        cand = sorted(folder.glob("*.xlsx"), key=lambda p: p.stat().st_mtime, reverse=True)

    if not cand:
        raise FileNotFoundError("Could not find a FirstPassAccuracy workbook (FirstPassAccuracy*.*).")

    return cand[0]


# --------------------------------------------------------------------------------------
# Load & prepare
# --------------------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def _load_fpa(root: Path) -> pd.DataFrame:
    f = _find_fpa_workbook(root)
    df = pd.read_excel(f, engine="openpyxl")  # engine robust

    # Column mapping (case-insensitive)
    cols = {c.lower().strip(): c for c in df.columns}

    def _col(*cands: str) -> str:
        for c in cands:
            key = c.lower().strip()
            if key in cols:
                return cols[key]
        raise KeyError(f"Missing required column; tried {cands}")

    # Typical names (adjusted to your file)
    col_date = _col("Activity Date", "Activity_Date", "ActivityDate", "Date")
    col_result = _col("Review Result", "Result", "Review_Result")
    col_comment = _col("Case Comment", "Comment", "Case_Comment")
    col_portfolio = _col("Portfolio", "portfolio")
    col_scheme = _col("Scheme", "scheme", "Scheme Name")

    # Normalise
    df["_date"] = pd.to_datetime(df[col_date], errors="coerce")
    df = df[~df["_date"].isna()].copy()
    df["_ym"] = df["_date"].dt.to_period("M").astype(str)  # 'YYYY-MM'

    # Result as pass/fail (very robust)
    rr = df[col_result].astype(str).str.strip().str.lower()
    df["_pass"] = rr.isin(["pass", "passed", "p", "ok", "correct"])

    # Keep relevant fields
    keep = [col_portfolio, col_scheme, col_comment, "_date", "_ym", "_pass"]
    return df[keep].rename(columns={col_portfolio: "portfolio", col_scheme: "scheme", col_comment: "comment"})


# --------------------------------------------------------------------------------------
# Label reasons (keyword buckets) – fast & transparent
# --------------------------------------------------------------------------------------
_PATTERNS = [
    ("Incorrect form", ["form", "incomplete", "missing field", "signature", "incorrect form"]),
    ("Bank / payment", ["bank", "bacs", "payment", "standing order", "account", "sort code"]),
    ("Waiting on member/TPA", ["await", "waiting", "member reply", "chaser", "tpa reply", "response"]),
    ("Manual calculation", ["manual calc", "calculation", "calc error", "recalc"]),
    ("Postal delay", ["post", "postal", "mail", "royal mail"]),
    ("Data entry error", ["data entry", "keying", "typo", "transposed"]),
    ("System", ["system", "it issue", "workflow", "bug"]),
]
def _label_reason(s: pd.Series) -> pd.Series:
    text = s.fillna("").astype(str).str.lower()
    lab = pd.Series(np.full(len(text), "Other"), index=text.index)
    for bucket, kws in _PATTERNS:
        hit = False
        for kw in kws:
            hit = hit | text.str.contains(rf"\b{pd.re.escape(kw)}\b", regex=True)
        lab[hit] = bucket
    return lab

# --------------------------------------------------------------------------------------
# Small chart helpers (clean axes)
# --------------------------------------------------------------------------------------
def _clean_axis(ax: plt.Axes):
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5, integer=False))
    ax.set_yticklabels([])  # remove y axis ticks/labels
    ax.tick_params(axis='y', left=False)

# --------------------------------------------------------------------------------------
# View
# --------------------------------------------------------------------------------------
def run(store: Dict, params: Dict, q: str):
    """
    Required signature: run(store, params, q)
    """
    root: Path = store["root"]
    df = _load_fpa(root)

    # Time window: Jan-2025 .. latest in data
    start = pd.Period("2025-01", freq="M")
    end = pd.Period(df["_ym"].max(), freq="M")
    months = pd.period_range(start, end, freq="M").astype(str)

    # Monthly pass%
    mon = (df.groupby("_ym")["_pass"].mean() * 100.0).round(1)
    mon = mon.reindex(months, fill_value=0.0)

    st.header("First-Pass Accuracy — Jan–{}"
              .format(pd.to_datetime(end.start_time).strftime("%b %y")))

    # Row: MoM Line
    left, right = st.columns([1.15, 1])
    with left:
        fig, ax = plt.subplots(figsize=(7.5, 2.5))
        ax.plot(pd.to_datetime(mon.index), mon.values, marker="o", linewidth=2.0, color="#5B8DEF")
        _clean_axis(ax)
        # soft X axis line
        ax.axhline(0, color="#E5E7EB", linewidth=1.2)
        # x ticks as MMM-YY
        ax.set_xticks(pd.to_datetime(mon.index))
        ax.set_xticklabels([pd.to_datetime(x).strftime("%b-%y") for x in mon.index], rotation=0)
        ax.set_title("Pass % — MoM", loc="left")
        # labels
        for x, y in zip(pd.to_datetime(mon.index), mon.values):
            ax.text(x, y + 1.5, f"{y:.0f}%", ha="center", va="bottom", fontsize=9, color="#374151")
        st.pyplot(fig, use_container_width=True)

    # Right: pass% by portfolio × scheme for latest month (Jun-25 or most recent)
    with right:
        latest = months[-1]
        df_latest = df[df["_ym"] == latest].copy()
        if df_latest.empty:
            st.info("No rows for the latest month in this file.")
        else:
            grp = df_latest.groupby(["portfolio", "scheme"])["_pass"].mean().mul(100).round(1)
            tbl = (grp.reset_index()
                      .sort_values(["portfolio", "_pass"], ascending=[True, False])
                      .rename(columns={"_pass": "pass_%"}))
            st.subheader(f"Pass % by Portfolio × Scheme — {pd.to_datetime(latest+'-01').strftime('%b-%y')}")
            st.dataframe(tbl, hide_index=True, use_container_width=True)

    st.divider()

    # Row: RCA1 bar (Pareto) at left + RCA2 table (top 80%) at right for latest month
    left, right = st.columns([1.05, 1])

    df_latest = df[df["_ym"] == months[-1]].copy()
    reasons = _label_reason(df_latest["comment"])
    rc1 = (reasons.value_counts().sort_values(ascending=False)).rename("count")
    if rc1.empty:
        st.info("No fails detected for the latest month.")
        return

    # Pareto data
    pareto = rc1.copy()
    cum = pareto.cumsum() / pareto.sum() * 100.0

    with left:
        fig, ax = plt.subplots(figsize=(7.5, 3.2))
        bars = ax.bar(pareto.index, pareto.values, color=["#9ECBF7", "#90D494", "#C1C1C1", "#F6B98A", "#B9B9DF", "#E5A3A3", "#C9F0FF"])
        _clean_axis(ax)
        # x tick vertical for readability
        ax.set_xticklabels(pareto.index, rotation=90)
        ax.axhline(0, color="#E5E7EB", linewidth=1.2)
        ax.set_title(f"RCA1 — {pd.to_datetime(months[-1]+'-01').strftime('%b %Y')} (Pareto)", loc="left")
        # Cumulative line (single axis to keep y simple)
        ax2 = ax.twinx()
        ax2.plot(range(len(cum)), cum.values, marker="o", linewidth=2, color="#5B8DEF")
        ax2.set_ylim(0, 105)
        ax2.set_yticklabels([])  # hide numbers
        ax2.grid(False)
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)
        for i, (b, cp) in enumerate(zip(bars, cum.values)):
            ax2.text(i, cp + 2, f"{cp:.0f}%", ha="center", va="bottom", fontsize=9, color="#374151")
        st.pyplot(fig, use_container_width=True)

    with right:
        # RCA2 (deeper) from the same keyword labelling, show top 80%
        rc2 = (reasons.value_counts(normalize=False)
                        .rename_axis("RCA2")
                        .reset_index(name="count"))
        rc2["percent"] = (rc2["count"] / rc2["count"].sum() * 100).round(1)
        rc2["cum_percent"] = rc2["percent"].cumsum().round(1)
        rc2 = rc2[rc2["cum_percent"] <= 80.0].copy()
        st.subheader(f"RCA2 — Top 80% ({pd.to_datetime(months[-1]+'-01').strftime('%b %Y')})")
        st.dataframe(rc2, hide_index=True, use_container_width=True)

    return "ok", None
