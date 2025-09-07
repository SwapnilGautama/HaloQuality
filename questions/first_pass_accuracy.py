# -*- coding: utf-8 -*-
# questions/first_pass_accuracy.py
from __future__ import annotations

import re
from io import BytesIO
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# -------------------------------------------------------------------
# Config / constants
# -------------------------------------------------------------------

# Data locations we auto-discover (kept as in your working version)
CANDIDATE_FPA_PATHS = [
    # your canonical folder
    "data/first_pass_accuracy/FirstPassAccuracy_Aug25.xlsx",
    # older / alternate names people have uploaded
    "data/first_pass_accuracy/FirstPassAccuracy_Aug'25.xlsx",
    "data/first_pass_accuracy/FirstPassAccuracy_Aug’25.xlsx",
    "data/first_pass_accuracy/FirstPassAccuracy.xlsx",
]

MONTH_FMT = "%b-%y"  # MMM-YY on charts

# Reason patterns (RCA for fails) – extend freely
REASON_KEYWORDS: Dict[str, List[str]] = {
    "Document missing": [
        "missing form", "missing document", "doc missing", "no form", "incomplete form",
        "unsent form", "not received form", "id missing", "proof missing"
    ],
    "Data entry / setup": [
        "data entry", "keying error", "typo", "miskey", "wrong field", "wrong entry",
        "incorrect entry", "set up", "setup", "set-up", "scheme setup", "pension set up"
    ],
    "Bank / payment": [
        "bank", "sort code", "account number", "payment", "bacs", "returned payment",
        "cheque", "check", "bank detail"
    ],
    "Waiting on member/TPA": [
        "waiting on member", "awaiting member", "waiting on tpa", "awaiting tpa",
        "chaser sent", "member yet to respond", "no response", "awaiting documents"
    ],
    "Postal / dispatch": [
        "post", "postal", "dispatch", "sent by mail", "royal mail", "courier", "returned mail"
    ],
    "Manual calculation": [
        "manual calc", "manual calculation", "calc error", "recalc", "re-calculation"
    ],
    "Trustee / AVC": [
        "trustee", "avc", "additional voluntary contribution"
    ],
    "System": [
        "system", "workflow", "it issue", "technical issue", "system error", "system down"
    ],
}

# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------

def _find_workbook() -> Optional[str]:
    """Return the first existing workbook path from known candidates."""
    for p in CANDIDATE_FPA_PATHS:
        try:
            with open(p, "rb"):
                return p
        except Exception:
            continue
    return None


def _load_data(path: str) -> pd.DataFrame:
    """
    Expected columns:
      - 'Activity Date' (date)
      - 'Review Result' ('Pass'/'Fail' or similar)
      - 'Portfolio'
      - 'Scheme'
      - 'Case Comment'  (free text, optional)
    """
    df = pd.read_excel(path, engine="openpyxl")
    # Normalise column names (case-insensitive)
    cols = {c: c.strip() for c in df.columns}
    df.rename(columns=cols, inplace=True)

    # Be tolerant to various spellings
    col_date = next((c for c in df.columns if c.lower().startswith("activity date")), None)
    col_result = next((c for c in df.columns if "review" in c.lower() and "result" in c.lower()), None)
    col_portfolio = next((c for c in df.columns if c.lower() == "portfolio"), None)
    col_scheme = next((c for c in df.columns if c.lower() == "scheme"), None)
    col_comment = next((c for c in df.columns if "comment" in c.lower()), None)

    if not all([col_date, col_result]):
        raise ValueError("Required columns missing (need at least 'Activity Date' and 'Review Result').")

    # Keep known columns only
    keep = [col_date, col_result, col_portfolio, col_scheme, col_comment]
    df = df[[c for c in keep if c is not None]].copy()

    df[col_date] = pd.to_datetime(df[col_date], errors="coerce")
    df["month"] = df[col_date].dt.to_period("M").dt.to_timestamp()

    # Normalize result
    df["result"] = df[col_result].astype(str).str.strip().str.lower()
    df["is_pass"] = df["result"].isin(["pass", "passed", "p", "true", "1", "yes"])

    # Friendly names
    if col_portfolio:
        df.rename(columns={col_portfolio: "portfolio"}, inplace=True)
    else:
        df["portfolio"] = "(Unknown)"

    if col_scheme:
        df.rename(columns={col_scheme: "scheme"}, inplace=True)
    else:
        df["scheme"] = "(Unknown)"

    if col_comment:
        df.rename(columns={col_comment: "comment"}, inplace=True)
    else:
        df["comment"] = ""

    # Filter sensible window: Jan-2025 -> latest (keeps your requirement)
    df = df[df["month"] >= pd.Timestamp("2025-01-01")]
    return df


def _pass_mom(df: pd.DataFrame) -> pd.DataFrame:
    by_m = df.groupby("month", dropna=False).agg(
        total=("is_pass", "size"),
        passed=("is_pass", "sum"),
    )
    by_m["pass_pct"] = np.where(by_m["total"] > 0, by_m["passed"] / by_m["total"] * 100.0, 0.0)
    # Ensure all months Jan..latest exist
    idx = pd.period_range(by_m.index.min(), by_m.index.max(), freq="M").to_timestamp()
    by_m = by_m.reindex(idx, fill_value=0)
    by_m["label"] = by_m.index.strftime(MONTH_FMT)
    return by_m.reset_index(drop=True)


def _build_reason_patterns(keywords_map: Dict[str, List[str]]) -> Dict[str, str]:
    """
    Build case-insensitive regex patterns with word boundaries for each reason.
    """
    pats: Dict[str, str] = {}
    for reason, kws in keywords_map.items():
        kws = [k.strip() for k in kws if k and isinstance(k, str)]
        if not kws:
            continue
        # Escape each kw, then join with alternation. Use word boundaries.
        escaped = [re.escape(k) for k in kws]
        pats[reason] = r"(?i)\b(?:%s)\b" % "|".join(escaped)
    return pats


def _label_reason(series_comment: pd.Series) -> pd.DataFrame:
    """
    Vectorised classification of fail reasons on the *latest month* comments.
    Returns a summary table with count/percent/cum_percent.
    """
    s = series_comment.fillna("").astype(str).str.strip().str.lower()

    patterns = _build_reason_patterns(REASON_KEYWORDS)

    # Start with 'Other', then override when a reason matches (first-hit wins in the declared order)
    labels = pd.Series("Other", index=s.index)

    for reason, pat in patterns.items():
        hit = s.str.contains(pat, regex=True, na=False)
        labels = np.where((labels == "Other") & (hit), reason, labels)

    # Summary
    total = max(len(labels), 1)
    summary = (
        pd.Series(labels, name="reason")
        .value_counts(dropna=False)
        .sort_values(ascending=False)
        .rename_axis("reason")
        .reset_index(name="count")
    )
    summary["percent"] = (summary["count"] / total * 100.0).round(1)
    summary["cum_percent"] = summary["percent"].cumsum().round(1)

    # Keep only top 80% (like your complaints flow)
    summary = summary[summary["cum_percent"] <= 80.0].reset_index(drop=True)
    if summary.empty:
        # fall back to top 10 if cumulative cut-off excludes all
        summary = (
            pd.Series(labels, name="reason")
            .value_counts(dropna=False)
            .head(10)
            .rename_axis("reason")
            .reset_index(name="count")
        )
        summary["percent"] = (summary["count"] / total * 100.0).round(1)
        summary["cum_percent"] = summary["percent"].cumsum().round(1)

    return summary


# -------------------------------------------------------------------
# Streamlit render
# -------------------------------------------------------------------

def run(store: Dict, params: Dict, q: str) -> None:
    """
    Render First Pass Accuracy analysis (Jan-25 -> latest).
    """
    # Load workbook
    path = _find_workbook()
    if not path:
        st.error("Could not find a FirstPassAccuracy workbook (FirstPassAccuracy*.xlsx).")
        return

    df = _load_data(path)

    st.markdown("### First-Pass Accuracy — Jan–{}"
                .format(df["month"].max().strftime("%b %y")))

    # Pass% MoM (Jan..latest)
    by_m = _pass_mom(df)

    c1, c2 = st.columns([2, 2], gap="large")

    with c1:
        st.caption("Pass % — MoM")
        fig, ax = plt.subplots(figsize=(6.8, 3.0), dpi=110)
        ax.plot(by_m["label"], by_m["pass_pct"], marker="o", linewidth=2.5)
        for x, y in zip(by_m["label"], by_m["pass_pct"]):
            ax.text(x, y + 1, f"{y:.0f}%", ha="center", va="bottom", fontsize=9)
        # Aesthetics as requested: soft x-axis, no grid, no y-axis
        ax.grid(False)
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.spines["bottom"].set_color("#D0D0D0")
        ax.set_xlabel("")
        ax.set_ylabel("")
        plt.xticks(rotation=0)
        st.pyplot(fig, clear_figure=True)

    # Pass % by portfolio × scheme for the latest month
    latest_m = df["month"].max()
    df_latest = df[df["month"] == latest_m].copy()
    g = df_latest.groupby(["portfolio", "scheme"], dropna=False).agg(
        total=("is_pass", "size"),
        passed=("is_pass", "sum"),
    ).reset_index()
    g["pass_%"] = np.where(g["total"] > 0, g["passed"] / g["total"] * 100.0, 0.0).round(0)
    g = g.sort_values(["portfolio", "scheme"]).rename(columns={"pass_%": "pass_%"})

    with c2:
        st.caption(f"Pass % by Portfolio × Scheme — {latest_m.strftime('%b-%y')}")
        st.dataframe(
            g[["portfolio", "scheme", "pass_%"]],
            hide_index=True,
            use_container_width=True,
        )

    st.markdown("---")

    # Fail reasons (latest month only, top 80%)
    st.caption(f"Reasons for Fail — {latest_m.strftime('%b %Y')} (counts)")
    fails = df_latest[~df_latest["is_pass"]].copy()

    if fails.empty:
        st.info("No fail cases found in the latest month.")
        return

    summary = _label_reason(fails["comment"])

    # Chart
    fig2, ax2 = plt.subplots(figsize=(6.8, 3.0), dpi=110)
    ax2.bar(summary["reason"], summary["count"])
    # Aesthetics
    ax2.grid(False)
    ax2.set_yticks([])
    ax2.spines["left"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    ax2.spines["bottom"].set_color("#D0D0D0")
    plt.xticks(rotation=90)
    ax2.set_xlabel("")
    ax2.set_ylabel("")
    for i, v in enumerate(summary["count"].tolist()):
        ax2.text(i, v + max(summary["count"].max() * 0.02, 0.5), str(v), ha="center", va="bottom", fontsize=9)
    st.pyplot(fig2, clear_figure=True)

    # Table
    st.caption("Reason breakdown (top 80%) — {}".format(latest_m.strftime("%b-%y")))
    st.dataframe(summary, hide_index=True, use_container_width=True)
