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
        # last resort: try any 'date'ish col
        date_like = [c for c in df.columns if "date" in c.lower()]
        dcol = date_like[0] if date_like else None
    if dcol is None:
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
        raw = df[mcol].astype(str).str.strip()

        def _to_period(x: str) -> Optional[pd.Period]:
            if not x or x.lower() == "nan":
                return pd.NaT
            # Pure month names -> assume 2025
            if re.fullmatch(r"[A-Za-z]{3,9}", x):
                dt = pd.to_datetime(f"1 {x} 2025", errors="coerce", dayfirst=True)
                return dt.to_period("M") if pd.notna(dt) else pd.NaT
            dt = pd.to_datetime(x, errors="coerce", dayfirst=True)
            return dt.to_period("M") if pd.notna(dt) else pd.NaT

        return raw.map(_to_period)

    return pd.Series(pd.PeriodIndex([], freq="M"), index=df.index)


def _reason_rca2_map(text: str) -> str:
    """Simple keyword bucketing into RCA2-like buckets, tuned to your data."""
    if not isinstance(text, str) or not text.strip():
        return "Other"
    t = text.lower()

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
    palette = ["#8ecae6", "#bde0fe", "#cdeac0", "#ffd6a5", "#fbc4ab", "#cdb4db", "#b9fbc0"]
    if n <= len(palette):
        return palette[:n]
    return [palette[i % len(palette)] for i in range(n)]


# -----------------------------
# Main question
# -----------------------------
def run(store, params: Optional[Dict] = None, user_text: Optional[str] = None) -> Tuple[str, pd.DataFrame]:
    """
    Render two rows side-by-side:
    Row 1: Portfolio table (June 2025) + MoM Complaints/1000 (Jan–Jun 2025)
    Row 2: RCA2 Top 80% table (June 2025) + RCA1 bar chart (June 2025)
    """
    params = params or {}
    month_key = pd.Period("2025-06", freq="M")  # fixed to Jun'25 view

    # Get data from store
    cases = getattr(store, "cases", None) or getattr(store, "raw_cases", None) or store.get("cases")
    complaints = getattr(store, "complaints", None) or getattr(store, "raw_complaints", None) or store.get("complaints")
    if cases is None or complaints is None:
        st.error("Missing data. Need both cases and complaints data.")
        return "complaints_june_by_portfolio", pd.DataFrame()

    # Identify columns
    case_id_col = _first_col(cases, ["Case ID", "id", "Original Process Affected Case ID"])
    portfolio_cases_col = _first_col(cases, ["Portfolio", "Portfolio Name"])

    compl_case_id_col = _first_col(complaints, ["Original Process Affected Case ID", "Case ID"])
    portfolio_compl_col = _first_col(complaints, ["Portfolio", "Portfolio Name"])
    rca2_text_col = _first_col(complaints, ["Brief Description - RCA done by admin", "RCA2", "RCA 2", "RCA"])
    rca1_col = _first_col(complaints, ["RCA1", "RCA 1", "Parent Case Type", "Reason", "Primary Reason", "High Level RCA"])

    warn_cols = []
    if case_id_col is None: warn_cols.append("Case ID")
    if portfolio_cases_col is None: warn_cols.append("Portfolio (cases)")
    if warn_cols:
        st.warning(f"Missing columns in cases: {warn_cols}")

    # Prepare months/portfolio
    cases = cases.copy()
    cases["_month"] = _month_from_cases(cases)
    cases["_portfolio"] = _ensure_portfolio_series(cases[portfolio_cases_col]) if portfolio_cases_col else "Unknown"

    complaints = complaints.copy()
    complaints["_month"] = _month_from_complaints(complaints)

    # Portfolio on complaints (from own col or join with cases)
    if portfolio_compl_col is None and compl_case_id_col and case_id_col:
        tmp = complaints[[compl_case_id_col]].merge(
            cases[[case_id_col, "_portfolio"]],
            left_on=compl_case_id_col,
            right_on=case_id_col,
            how="left",
        )
        complaints["_portfolio"] = _ensure_portfolio_series(tmp["_portfolio"])
    elif portfolio_compl_col is not None:
        complaints["_portfolio"] = _ensure_portfolio_series(complaints[portfolio_compl_col])
    else:
        complaints["_portfolio"] = "Unknown"

    # RCA2 derived mapping
    complaints["_rca2"] = complaints[rca2_text_col].map(_reason_rca2_map) if rca2_text_col else "Other"

    # -----------------------------
    # A. Portfolio table (June)
    # -----------------------------
    cases_jun = cases.loc[cases["_month"] == month_key]
    compl_jun = complaints.loc[complaints["_month"] == month_key]

    if case_id_col:
        cases_by_pf = cases_jun.groupby("_portfolio", dropna=False)[case_id_col].count().rename("cases").to_frame()
    else:
        cases_by_pf = pd.DataFrame({"cases": []})

    comp_by_pf = compl_jun.groupby("_portfolio", dropna=False)["_rca2"].count().rename("complaints").to_frame()

    table = cases_by_pf.join(comp_by_pf, how="outer").fillna(0)
    if not table.empty:
        table["cases"] = table["cases"].astype(int)
        table["complaints"] = table["complaints"].astype(int)
        table["per_1000"] = np.where(table["cases"] > 0, table["complaints"] / (table["cases"] / 1000.0), np.nan)
        table = table.reset_index().rename(columns={"_portfolio": "portfolio"})
    else:
        table = pd.DataFrame(columns=["portfolio", "cases", "complaints", "per_1000"])

    # add Total row (top)
    if not table.empty:
        total_row = pd.DataFrame(
            {
                "portfolio": ["Total"],
                "cases": [int(table["cases"].sum())],
                "complaints": [int(table["complaints"].sum())],
                "per_1000": [
                    table["complaints"].sum() / (table["cases"].sum() / 1000.0)
                    if table["cases"].sum() > 0 else np.nan
                ],
            }
        )
        table_display = pd.concat([total_row, table], ignore_index=True)
    else:
        table_display = table.copy()

    # -----------------------------
    # B. MoM Complaints/1,000 (Jan–Jun 2025)
    # -----------------------------
    months_2025 = pd.period_range("2025-01", "2025-06", freq="M")
    if case_id_col:
        cases_m = cases[cases["_month"].isin(months_2025)].groupby("_month")[case_id_col].count()
    else:
        cases_m = pd.Series(0, index=months_2025)
    compl_m = complaints[complaints["_month"].isin(months_2025)].groupby("_month")["_rca2"].count()

    cases_m = cases_m.reindex(months_2025, fill_value=0)
    compl_m = compl_m.reindex(months_2025, fill_value=0)
    per1000_m = np.where(cases_m > 0, compl_m / (cases_m / 1000.0), 0.0)

    # -----------------------------
    # Layout — Row 1
    # -----------------------------
    st.markdown("### Complaint analysis — Jun 2025")
    if not table_display.empty:
        st.caption(
            f"Total: cases={int(table_display.loc[0, 'cases']) if 'cases' in table_display.columns and not table_display.empty else 0:,}, "
            f"complaints={int(table_display.loc[0, 'complaints']) if 'complaints' in table_display.columns and not table_display.empty else 0:,}, "
            f"per_1000={table_display.loc[0, 'per_1000']:.2f}" if not table_display.empty and pd.notna(table_display.loc[0, 'per_1000']) else ""
        )

    c1, c2 = st.columns((1, 1), gap="large")

    with c1:
        st.subheader("Complaints per 1,000 — June 2025 (by portfolio)")
        st.dataframe(table_display, use_container_width=True)

    with c2:
        st.subheader("Complaints per 1,000 — Jan–Jun 2025")
        xlabels = [m.strftime("%b") for m in months_2025]
        fig, ax = plt.subplots(figsize=(6.4, 3.4), dpi=150)
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
        for i, y in enumerate(per1000_m):
            ax.text(i, y + (max(per1000_m) * 0.04 if max(per1000_m) > 0 else 0.1), f"{y:.1f}", ha="center", va="bottom", fontsize=9)
        _soft_line(ax)
        ax.set_xticks(range(len(months_2025)))
        ax.set_xticklabels(xlabels)
        st.pyplot(fig, use_container_width=True)

    # -----------------------------
    # C. RCA2 Top-80% (table) & RCA1 bar (chart) — June 2025
    # -----------------------------
    # RCA2 table (Top 80%)
    reasons2 = (
        compl_jun["_rca2"]
        .value_counts(dropna=False)
        .rename_axis("RCA2")
        .to_frame("count")
        .reset_index()
    )
    if not reasons2.empty:
        reasons2["percent"] = (reasons2["count"] / reasons2["count"].sum()) * 100
        r2 = reasons2.sort_values("count", ascending=False).reset_index(drop=True)
        r2["cum_percent"] = r2["percent"].cumsum()
        top80 = r2[r2["cum_percent"] <= 80.0]
        if top80.empty:
            top80 = r2.iloc[[0]].copy()
    else:
        top80 = pd.DataFrame(columns=["RCA2", "count", "percent", "cum_percent"])

    # RCA1 bar chart (June)
    if rca1_col is not None:
        rca1_series = compl_jun[rca1_col].astype(str).str.strip().replace({"nan": np.nan}).fillna("Other")
    else:
        # Fallback to RCA2 for chart if RCA1 missing
        rca1_series = compl_jun["_rca2"]

    rca1_counts = (
        rca1_series.value_counts(dropna=False)
        .rename_axis("RCA1")
        .to_frame("count")
        .reset_index()
    )

    d1, d2 = st.columns((1, 1), gap="large")
    with d1:
        st.subheader("RCA2 — Top 80% (June 2025)")
        st.dataframe(top80, use_container_width=True)

    with d2:
        st.subheader("RCA1 — June 2025")
        fig2, ax2 = plt.subplots(figsize=(6.4, 3.4), dpi=150)
        if not rca1_counts.empty:
            cols = _pastel_colors(len(rca1_counts))
            bars = ax2.bar(rca1_counts["RCA1"], rca1_counts["count"], color=cols, edgecolor="none")
            for b in bars:
                h = b.get_height()
                ax2.text(b.get_x() + b.get_width() / 2, h + (rca1_counts["count"].max() * 0.03), f"{int(h)}", ha="center", va="bottom", fontsize=9)
            _soft_line(ax2)
            ax2.set_xticklabels(rca1_counts["RCA1"], rotation=20, ha="right")
        else:
            _soft_line(ax2)
            ax2.text(0.5, 0.5, "No June RCA1 data", ha="center", va="center")
        st.pyplot(fig2, use_container_width=True)

    # Return a non-empty thing for app bookkeeping
    return "complaints_june_by_portfolio", table_display
