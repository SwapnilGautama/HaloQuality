# -*- coding: utf-8 -*-
# questions/complaints_june_by_portfolio.py
from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt


# ---------------------------
# Small helpers
# ---------------------------

def _sec(title: str, caption: Optional[str] = None) -> None:
    st.subheader(title)
    if caption:
        st.caption(caption)


MONTH_KEY_YYYY_MM = {
    "jan": "2025-01", "feb": "2025-02", "mar": "2025-03",
    "apr": "2025-04", "may": "2025-05", "jun": "2025-06",
}

PASTEL = {
    "line": "#8cbfd4",     # soft blue
    "bar1": "#a6dcef",      # pastel sky
    "bar2": "#c9e4c5",      # pastel green
    "bar3": "#ffe3a3",      # pastel yellow
    "bar4": "#f7b5b4",      # pastel coral
    "bar5": "#c9c3e6",      # pastel lavender
    "bar6": "#d5d5d5",      # soft grey
}

def _first_col(df: pd.DataFrame, options: List[str]) -> Optional[str]:
    """Return the first column present from a list of candidate names."""
    opts = [o for o in options if o in df.columns]
    return opts[0] if opts else None

def _ensure_dt(series: pd.Series, dayfirst: bool = True) -> pd.Series:
    return pd.to_datetime(series, errors="coerce", dayfirst=dayfirst)

def _normalize_portfolio(s: pd.Series) -> pd.Series:
    # Clean whitespace, unify case, keep original visible text
    return s.astype(str).str.strip().replace({"nan": np.nan})

def _yymm_from_month_name(month_str: str, year: int = 2025) -> Optional[pd.Timestamp]:
    if not isinstance(month_str, str):
        return None
    m = month_str.strip().lower()[:3]
    if m in MONTH_KEY_YYYY_MM:
        return pd.Period(MONTH_KEY_YYYY_MM[m]).to_timestamp()
    return None


# ---------------------------
# RCA CLASSIFIERS (text → label)
# ---------------------------

def classify_rca1(text: str) -> str:
    """
    High-level buckets inspired by your slide:
    Delay, Procedure, Communication, System,
    Incorrect/Incomplete info, Other
    """
    if not isinstance(text, str) or not text.strip():
        return "Other"
    t = text.lower()

    # Delay-like
    if re.search(r"\bdelay|sla|late|backlog|chaser\b", t):
        return "Delay"

    # Procedure / process
    if re.search(r"\bprocedure|process|timescale|scheme rules|form|template|checklist\b", t):
        return "Procedure"

    # Communication
    if re.search(r"\bcommunicat|letter|email|contact|info sent|no reply\b", t):
        return "Communication"

    # System / tech
    if re.search(r"\bsystem|portal|bug|it issue|workflow|upload error\b", t):
        return "System"

    # Incorrect/Incomplete info
    if re.search(r"\bincorrect|incomplete|missing|not provided|wrong|invalid\b", t):
        return "Incorrect/Incomplete info"

    return "Other"


def classify_rca2(text: str) -> str:
    """
    More granular categories – tuned to the kinds of notes you shared.
    Falls back to 'Other'. Keep adding keywords as you see real comments.
    """
    if not isinstance(text, str) or not text.strip():
        return "Other"
    t = text.lower()

    # Map examples from the slide and typical admin notes:
    if re.search(r"\bmanual calculat|calc\b", t):
        return "Manual calculation"
    if re.search(r"\baptia standard timescale|timescale\b", t):
        return "Aptia standard Timescale"
    if re.search(r"\bpension set ?up|onboard|new joiner\b", t):
        return "Pension set up"
    if re.search(r"\bpostal|post delay|mail\b", t):
        return "Postal Delay"
    if re.search(r"\bavc\b", t):
        return "Delay – AVC"
    if re.search(r"\brequirement not checked|qc fail|check not done\b", t):
        return "Delay Requirement not checked"
    if re.search(r"\bcase not created|no case\b", t):
        return "Delay Case not created"
    if re.search(r"\b2nd review|second review\b", t):
        return "Delay 2nd Review"
    if re.search(r"\btrustee\b", t):
        return "Delay – Trustee"
    if re.search(r"\bscheme rules\b", t):
        return "Scheme Rules"
    if re.search(r"\bdrop in value|factor change\b", t):
        return "Drop in value/ factor change"
    if re.search(r"\boverpay|over payment\b", t):
        return "Overpayment"
    if re.search(r"\bpension increase\b", t):
        return "Pension Increase"
    if re.search(r"\btransfer document|transfer doc\b", t):
        return "Transfer Documentation"
    if re.search(r"\bdeath benefit|death claim\b", t):
        return "Death benefits payout"
    if re.search(r"\bcommunicat|letter|email\b", t):
        return "Communication"
    if re.search(r"\bdocument(s)? missing|missing doc|no doc\b", t):
        return "Documentation Missing"

    return "Other"


# ---------------------------
# Main render
# ---------------------------

def run(store: Dict, params: Dict, user_text: str | None = None):
    cases: pd.DataFrame = store.get("cases", pd.DataFrame()).copy()
    complaints: pd.DataFrame = store.get("complaints", pd.DataFrame()).copy()

    if cases.empty or complaints.empty:
        st.warning("No overlapping data for cases and complaints.")
        return

    # ---- Column detection
    pf_cases_col = _first_col(cases, ["Portfolio", "portfolio"])
    pf_comp_col = _first_col(complaints, ["Portfolio", "portfolio"])

    id_col = _first_col(cases, ["Case ID", "Case Id", "CaseID", "Unique Ident", "Unique identifier"])
    case_dt_col = _first_col(cases, ["Create Date", "Create Dt", "Start Date"])

    comp_dt_col = _first_col(complaints, ["Date Complaint Received - DD/MM/YY", "Complaint Date", "Date"])
    rca_text_col = _first_col(complaints, ["Brief Description - RCA done by admin", "Brief Description", "RCA Comment"])

    missing = []
    if not pf_cases_col: missing.append("Portfolio (cases)")
    if not pf_comp_col: missing.append("Portfolio (complaints)")
    if not id_col: missing.append("Case ID")
    if not case_dt_col: missing.append("Create Date (cases)")
    if not comp_dt_col and "Month" not in complaints.columns: missing.append("Complaint Date / Month")
    if not rca_text_col: missing.append("Brief Description - RCA done by admin")

    if missing:
        st.info(f"Missing columns: {missing}")
        # We'll still try with whatever exists.

    # ---- Normalize key fields
    if pf_cases_col:
        cases["portfolio"] = _normalize_portfolio(cases[pf_cases_col])
    else:
        cases["portfolio"] = np.nan

    if pf_comp_col:
        complaints["portfolio"] = _normalize_portfolio(complaints[pf_comp_col])
    else:
        complaints["portfolio"] = np.nan

    # Case dates
    if case_dt_col:
        cases["_dt"] = _ensure_dt(cases[case_dt_col])
        cases["_month"] = cases["_dt"].dt.to_period("M").astype(str)
    else:
        cases["_dt"] = pd.NaT
        cases["_month"] = np.nan

    # Complaint dates (prefer real date; else Month column like 'June')
    if comp_dt_col:
        complaints["_dt"] = _ensure_dt(complaints[comp_dt_col])
    else:
        complaints["_dt"] = pd.NaT
    if "Month" in complaints.columns:
        tmp = complaints["Month"].apply(lambda x: _yymm_from_month_name(x, 2025))
        complaints["_dt"] = complaints["_dt"].fillna(tmp)
    complaints["_month"] = complaints["_dt"].dt.to_period("M").astype(str)

    # -----------------------------
    # 1) Complaints/1,000 by portfolio (for June 2025)
    # -----------------------------
    month_key = "2025-06"  # fixed for this question
    # Unique cases per portfolio for June
    if id_col:
        cases_jun = (cases[cases["_month"] == month_key]
                     .dropna(subset=[id_col])
                     .groupby("portfolio", dropna=False)[id_col]
                     .nunique()
                     .rename("cases"))
    else:
        cases_jun = pd.Series(dtype="int64", name="cases")

    # Complaints per portfolio for June
    comp_jun = (complaints[complaints["_month"] == month_key]
                .groupby("portfolio", dropna=False)
                .size()
                .rename("complaints"))

    by_port = pd.concat([cases_jun, comp_jun], axis=1).fillna(0).astype({"cases": "int64", "complaints": "int64"})
    # Fix portfolio label for NaN
    by_port = by_port.reset_index()
    by_port["portfolio"] = by_port["portfolio"].fillna("Unknown")
    # Add Total row at top
    tot_row = pd.DataFrame([{
        "portfolio": "Total",
        "cases": int(by_port["cases"].sum()),
        "complaints": int(by_port["complaints"].sum())
    }])
    by_port = pd.concat([tot_row, by_port], ignore_index=True)
    by_port["per_1000"] = by_port.apply(
        lambda r: (r["complaints"] / r["cases"] * 1000) if r["cases"] else np.nan, axis=1
    )
    # Order by complaints desc except Total on top
    by_port = pd.concat([
        by_port.iloc[[0]],
        by_port.iloc[1:].sort_values(["complaints", "cases"], ascending=[False, False])
    ], ignore_index=True)

    # -----------------------------
    # 2) Jan–Jun ’25 MoM line (Complaints per 1,000)
    # -----------------------------
    months = pd.period_range("2025-01", "2025-06", freq="M").astype(str)
    cases_m = (cases[cases["_month"].isin(months)]
               .dropna(subset=[id_col]) if id_col else cases[cases["_month"].isin(months)])
    cases_m = cases_m.groupby("_month")[id_col].nunique() if id_col else cases_m.groupby("_month").size()
    comps_m = complaints[complaints["_month"].isin(months)].groupby("_month").size()
    trend = pd.concat([cases_m.rename("cases"), comps_m.rename("complaints")], axis=1).fillna(0)
    trend["per_1000"] = trend.apply(lambda r: (r["complaints"] / r["cases"] * 1000) if r["cases"] else 0, axis=1)
    trend = trend.reindex(months).fillna(0)

    # -----------------------------
    # 3) RCA (from Brief Description text) — June only
    # -----------------------------
    text_series = complaints.loc[complaints["_month"] == month_key, rca_text_col] if rca_text_col else pd.Series(dtype=str)
    rca1 = text_series.apply(classify_rca1).value_counts(dropna=False).rename_axis("RCA1").reset_index(name="count")
    rca1 = rca1.sort_values("count", ascending=False)

    rca2 = text_series.apply(classify_rca2).value_counts(dropna=False).rename_axis("RCA2").reset_index(name="count")
    rca2 = rca2.sort_values("count", ascending=False)
    total_jun_complaints = int(rca2["count"].sum()) if not rca2.empty else 0
    if total_jun_complaints > 0:
        rca2["percent"] = (rca2["count"] / total_jun_complaints * 100).round(1)
        rca2["cum_percent"] = rca2["percent"].cumsum().round(1)
        rca2_top = rca2.loc[rca2["cum_percent"] <= 80]
        if rca2_top.empty and not rca2.empty:
            rca2_top = rca2.head(5)  # ensure we show something
    else:
        rca2["percent"] = 0.0
        rca2["cum_percent"] = 0.0
        rca2_top = rca2

    # -----------------------------
    # LAYOUT
    # -----------------------------
    # Row 1: table (left) + MoM line (right)
    left, right = st.columns([1.05, 1.05], gap="large")

    with left:
        _sec("Complaints per 1,000 — Jun 2025 (by portfolio)",
             f"Total: cases={by_port['cases'].sum():,}, complaints={by_port['complaints'].sum():,}, per_1000={by_port.loc[0,'per_1000']:.3f if pd.notna(by_port.loc[0,'per_1000']) else '—'}")
        st.dataframe(
            by_port[["portfolio", "cases", "complaints", "per_1000"]],
            use_container_width=True,
            hide_index=True,
        )

    with right:
        _sec("Complaints per 1,000 — MoM (Jan–Jun ’25)")
        fig, ax = plt.subplots(figsize=(6.5, 3.1), dpi=150)
        x = pd.to_datetime(trend.index.to_list())
        y = trend["per_1000"].to_numpy(dtype=float)

        # soft line
        ax.plot(x, y, marker="o", linewidth=2.5, color=PASTEL["line"])
        # data labels
        for xi, yi in zip(x, y):
            ax.text(xi, yi + (max(y) * 0.02 if y.max() else 0.02), f"{yi:.2f}", ha="center", va="bottom", fontsize=8)

        # clean look
        for spine in ["top", "right", "left", "bottom"]:
            ax.spines[spine].set_visible(False)
        ax.grid(False)
        ax.set_ylabel("")
        ax.set_xlabel("")
        ax.set_yticks([])
        ax.set_xticks(x)
        ax.set_xticklabels([d.strftime("%b") for d in x])
        plt.tight_layout()
        st.pyplot(fig, clear_figure=True)

    # Row 2: RCA2 table (left) + RCA1 chart (right)
    left2, right2 = st.columns([1.05, 1.05], gap="large")

    with left2:
        _sec("RCA2 — Top 80% (June 2025)")
        if rca2_top.empty:
            st.info("No classified RCA2 reasons for June.")
        else:
            st.dataframe(
                rca2_top[["RCA2", "count", "percent", "cum_percent"]],
                use_container_width=True,
                hide_index=True,
            )

    with right2:
        _sec("RCA1 — June 2025")
        if rca1.empty:
            st.info("No classified RCA1 reasons for June.")
        else:
            # Pastel bars with labels, no gridlines/axes clutter
            colors = [PASTEL["bar1"], PASTEL["bar2"], PASTEL["bar3"], PASTEL["bar4"], PASTEL["bar5"], PASTEL["bar6"]]
            fig2, ax2 = plt.subplots(figsize=(6.5, 3.1), dpi=150)
            vals = rca1["count"].to_numpy()
            idx = np.arange(len(vals))
            col_list = (colors * ((len(vals) // len(colors)) + 1))[:len(vals)]

            bars = ax2.bar(idx, vals, color=col_list)

            # labels
            for rect, v in zip(bars, vals):
                ax2.text(rect.get_x() + rect.get_width()/2, rect.get_height() + max(vals)*0.02,
                         f"{int(v)}", ha="center", va="bottom", fontsize=8)

            # tidy up
            ax2.set_xticks(idx)
            ax2.set_xticklabels(rca1["RCA1"].tolist(), rotation=15, ha="right")
            ax2.set_yticks([])
            ax2.set_ylabel("")
            ax2.set_xlabel("")
            for spine in ["top", "right", "left", "bottom"]:
                ax2.spines[spine].set_visible(False)
            ax2.grid(False)
            plt.tight_layout()
            st.pyplot(fig2, clear_figure=True)


# Required hook for the question runner
QUESTION = {
    "slug": "complaints_june_by_portfolio",
    "label": "complaint analysis — June 2025 (by portfolio)",
    "run": run,
}
