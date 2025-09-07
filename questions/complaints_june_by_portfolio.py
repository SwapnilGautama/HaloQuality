# -*- coding: utf-8 -*-
# questions/complaints_june_by_portfolio.py
from __future__ import annotations

from typing import Optional, Dict, Tuple, Iterable
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st


# -----------------------------
# Helpers (robust & reusable)
# -----------------------------
def _first_col(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    """Return first matching column name (case-insensitive), else None."""
    lc = {c.lower(): c for c in df.columns}
    for c in candidates:
        cl = c.lower()
        if cl in lc:
            return lc[cl]
    return None


def _ensure_portfolio_series(s: pd.Series) -> pd.Series:
    return (
        s.astype(str)
        .str.strip()
        .replace({"nan": np.nan})
        .fillna("Unknown")
    )


def _month_from_cases(df: pd.DataFrame) -> pd.Series:
    """
    Cases month:
      - Prefer 'Create Date'
      - Fallbacks: 'Start Date', 'Report Date'
      -> returns Period[M]
    """
    dcol = _first_col(df, ["Create Date", "Start Date", "Report Date"])
    if dcol is None:
        # last resort: try to coerce any column containing 'date'
        date_like = [c for c in df.columns if "date" in c.lower()]
        dcol = date_like[0] if date_like else None
    if dcol is None:
        # empty series so downstream logic still works
        return pd.Series(pd.PeriodIndex([], freq="M"), index=df.index)

    dt = pd.to_datetime(df[dcol], errors="coerce", dayfirst=True, utc=False)
    return dt.dt.to_period("M")


def _month_from_complaints(df: pd.DataFrame) -> pd.Series:
    """
    Complaints month:
      - Prefer 'Date Complaint Received - DD/MM/YY'
      - Else 'Date Complaint Received'
      - Else 'Month' (assume 2025 when only month text present)
    """
    dcol = _first_col(df, ["Date Complaint Received - DD/MM/YY", "Date Complaint Received"])
    if dcol is not None:
        dt = pd.to_datetime(df[dcol], errors="coerce", dayfirst=True, utc=False)
        return dt.dt.to_period("M")

    mcol = _first_col(df, ["Month"])
    if mcol is not None:
        # Accept values like 'June', 'Jun', '2025-06', etc. Assume year 2025 if missing.
        raw = df[mcol].astype(str).str.strip()
        def _to_period(x: str) -> Optional[pd.Period]:
            if not x or x.lower() == "nan":
                return pd.NaT
            # If looks like Mon or Month name w/o year, assume 2025
            if re.fullmatch(r"[A-Za-z]{3,9}", x):
                try:
                    dt = pd.to_datetime(f"1 {x} 2025", errors="coerce", dayfirst=True)
                    return dt.to_period("M")
                except Exception:
                    return pd.NaT
            # Else try to parse freely
            dt = pd.to_datetime(x, errors="coerce", dayfirst=True)
            if pd.isna(dt):
                return pd.NaT
            return dt.to_period("M")
        return raw.map(_to_period)

    # empty
    return pd.Series(pd.PeriodIndex([], freq="M"), index=df.index)


def _reason_rca2_map(text: str) -> str:
    """Simple keyword bucketing into RCA2-like buckets, tuned to your data."""
    if not isinstance(text, str) or not text.strip():
        return "Other"
    t = text.lower()

    # Order matters (more specific first)
    if any(k in t for k in ["death", "bereav"]):
        return "Death benefits payout"
    if "pension increase" in t or "pi" in t:
        return "Pension Increase"
    if any(k in t for k in ["overpay", "over pay"]):
        return "Overpayment"
    if any(k in t for k in ["manual", "calc", "calculation"]):
        return "Manual calculation"
    if "timescale" in t or "time scale" in t:
        return "Aptia standard Timescale"
    if any(k in t for k in ["scheme rule", "scheme rules"]):
        return "Scheme Rules"
    if any(k in t for k in ["factor change", "drop in value"]):
        return "Drop in value/ factor change"
    if "postal" in t:
        return "Postal delay"
    if "avc" in t:
        return "AVC"
    if "requirement not checked" in t or "not checked" in t:
        return "Requirement not checked"
    if "case not created" in t:
        return "Case not created"
    if "2nd review" in t or "second review" in t:
        return "2nd Review"

    return "Other"


def _soft_line(ax):
    """Remove borders/grid & y-axis for minimal, pastel look."""
    for spine in ["top", "right", "left", "bottom"]:
        ax.spines[spine].set_visible(False)
    ax.grid(False)
    ax.set_ylabel("")
    ax.set_yticks([])


def _pastel_colors(n: int) -> list[str]:
    # a small pastel-ish palette
    palette = ["#8ecae6", "#bde0fe", "#cdeac0", "#ffd6a5", "#fbc4ab", "#cdb4db", "#b9fbc0"]
    if n <= len(palette):
        return palette[:n]
    # repeat if more bars than palette
    return [palette[i % len(palette)] for i in range(n)]


# -----------------------------
# Main question
# -----------------------------
def run(store, params: Optional[Dict] = None, user_text: Optional[str] = None) -> Tuple[str, pd.DataFrame]:
    """
    Render: Complaint analysis — Jun 2025 (by portfolio)
    - Portfolio table with Total row
    - MoM line (Jan–Jun 2025) Complaints per 1,000
    - June reasons (RCA2) table + bar
    - RCA2 Top 80% table
    """
    params = params or {}
    month_key = pd.Period("2025-06", freq="M")  # fixed to June '25 per spec

    # 1) Pull data
    cases = getattr(store, "cases", None) or getattr(store, "raw_cases", None) or store.get("cases")
    complaints = getattr(store, "complaints", None) or getattr(store, "raw_complaints", None) or store.get("complaints")
    if cases is None or complaints is None:
        st.error("Missing data. Need both cases and complaints data.")
        return "complaints_june_by_portfolio", pd.DataFrame()

    # Columns we need / try to find
    case_id_col = _first_col(cases, ["Case ID", "id", "Original Process Affected Case ID"])
    portfolio_cases_col = _first_col(cases, ["Portfolio", "Portfolio Name"])
    if case_id_col is None or portfolio_cases_col is None:
        st.warning(f"Missing columns in cases: '{'Case ID'}' or '{'Portfolio'}'")
        return "complaints_june_by_portfolio", pd.DataFrame()

    cases = cases.copy()
    cases["_month"] = _month_from_cases(cases)
    cases["_portfolio"] = _ensure_portfolio_series(cases[portfolio_cases_col])

    # Complaints columns
    compl_case_id_col = _first_col(complaints, ["Original Process Affected Case ID", "Case ID"])
    portfolio_compl_col = _first_col(complaints, ["Portfolio", "Portfolio Name"])
    rca_text_col = _first_col(complaints, ["Brief Description - RCA done by admin", "RCA2", "RCA 2", "RCA"])

    complaints = complaints.copy()
    complaints["_month"] = _month_from_complaints(complaints)
    if portfolio_compl_col is None:
        # if not present, try to pick from cases via join on id
        if compl_case_id_col and case_id_col:
            tmp = complaints[[compl_case_id_col]].merge(
                cases[[case_id_col, "_portfolio"]], left_on=compl_case_id_col, right_on=case_id_col, how="left"
            )
            complaints["_portfolio"] = _ensure_portfolio_series(tmp["_portfolio"])
        else:
            complaints["_portfolio"] = "Unknown"
    else:
        complaints["_portfolio"] = _ensure_portfolio_series(complaints[portfolio_compl_col])

    if rca_text_col is None:
        complaints["_rca2"] = "Other"
    else:
        complaints["_rca2"] = complaints[rca_text_col].map(_reason_rca2_map)

    # -----------------------------
    # A. Portfolio table (June only)
    # -----------------------------
    cases_jun = cases.loc[cases["_month"] == month_key]
    compl_jun = complaints.loc[complaints["_month"] == month_key]

    cases_by_pf = (
        cases_jun.groupby("_portfolio", dropna=False)[case_id_col]
        .count()
        .rename("cases")
        .to_frame()
    )
    comp_by_pf = (
        compl_jun.groupby("_portfolio", dropna=False)["_rca2"]
        .count()
        .rename("complaints")
        .to_frame()
    )

    table = cases_by_pf.join(comp_by_pf, how="outer").fillna(0)
    table["cases"] = table["cases"].astype(int)
    table["complaints"] = table["complaints"].astype(int)
    table["per_1000"] = np.where(table["cases"] > 0, table["complaints"] / (table["cases"] / 1000.0), np.nan)
    table = table.reset_index().rename(columns={"_portfolio": "portfolio"})

    # add Total row (at top)
    total_row = pd.DataFrame(
        {
            "portfolio": ["Total"],
            "cases": [int(table["cases"].sum())],
            "complaints": [int(table["complaints"].sum())],
            "per_1000": [table["complaints"].sum() / (table["cases"].sum() / 1000.0) if table["cases"].sum() > 0 else np.nan],
        }
    )
    table_display = pd.concat([total_row, table], ignore_index=True)

    st.markdown("### Complaint analysis — Jun 2025 (by portfolio)")
    st.caption(f"Total: cases={int(table['cases'].sum()):,}, complaints={int(table['complaints'].sum()):,}, per_1000={table_display.loc[0, 'per_1000']:.2f}")
    st.dataframe(table_display, use_container_width=True)

    # --------------------------------------
    # B. MoM Complaints/1000 line (Jan–Jun)
    # --------------------------------------
    months_2025 = pd.period_range("2025-01", "2025-06", freq="M")
    # cases & complaints by month (all portfolios)
    cases_m = cases[cases["_month"].isin(months_2025)].groupby("_month")[case_id_col].count()
    compl_m = complaints[complaints["_month"].isin(months_2025)].groupby("_month")["_rca2"].count()

    # Align & fill zeros
    cases_m = cases_m.reindex(months_2025, fill_value=0)
    compl_m = compl_m.reindex(months_2025, fill_value=0)
    per1000_m = np.where(cases_m > 0, compl_m / (cases_m / 1000.0), 0.0)

    xlabels = [m.strftime("%b") for m in months_2025]
    fig, ax = plt.subplots(figsize=(6.6, 3.6), dpi=150)
    ax.plot(
        range(len(months_2025)),
        per1000_m,
        marker="o",
        linewidth=2.5,
        color="#8ecae6",
        alpha=0.95,
        solid_capstyle="round",
        antialiased=True,
    )
    # data labels
    for i, y in enumerate(per1000_m):
        ax.text(i, y + (max(per1000_m) * 0.04 if max(per1000_m) > 0 else 0.1), f"{y:.1f}", ha="center", va="bottom", fontsize=9)

    _soft_line(ax)
    ax.set_xticks(range(len(months_2025)))
    ax.set_xticklabels(xlabels)
    ax.set_title("Complaints per 1,000 — Jan–Jun 2025", fontsize=11)
    st.pyplot(fig, use_container_width=True)

    # --------------------------------------
    # C. June reasons (RCA2) table + bar
    # --------------------------------------
    reasons = (
        compl_jun["_rca2"]
        .value_counts(dropna=False)
        .rename_axis("reason")
        .to_frame("count")
        .reset_index()
    )
    if reasons.empty:
        reasons["percent"] = []
    else:
        reasons["percent"] = (reasons["count"] / reasons["count"].sum()) * 100

    st.markdown("### June reasons — contribution")
    st.dataframe(reasons, use_container_width=True)

    # bar
    fig2, ax2 = plt.subplots(figsize=(6.6, 3.6), dpi=150)
    cols = _pastel_colors(len(reasons))
    bars = ax2.bar(reasons["reason"], reasons["count"], color=cols, edgecolor="none")
    for b in bars:
        h = b.get_height()
        ax2.text(b.get_x() + b.get_width() / 2, h + max(reasons["count"]) * 0.03, f"{int(h)}", ha="center", va="bottom", fontsize=9)
    _soft_line(ax2)
    ax2.set_xticklabels(reasons["reason"], rotation=20, ha="right")
    ax2.set_title("June reasons (RCA2)", fontsize=11)
    st.pyplot(fig2, use_container_width=True)

    # --------------------------------------
    # D. RCA2 Top 80% table (cumulative)
    # --------------------------------------
    if not reasons.empty:
        r2 = reasons.copy()
        r2 = r2.sort_values("count", ascending=False).reset_index(drop=True)
        r2["cum_percent"] = r2["percent"].cumsum()
        top80 = r2[r2["cum_percent"] <= 80.0]
        # If first row already >80, still keep that single row
        if top80.empty and not r2.empty:
            top80 = r2.iloc[[0]].copy()
        st.markdown("### RCA2 — Top 80%")
        st.dataframe(top80, use_container_width=True)
    else:
        st.info("No June complaints found to build RCA2 Top 80% table.")

    # Return something non-empty for app bookkeeping
    return "complaints_june_by_portfolio", table_display
