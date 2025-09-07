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
        .replace({"nan": np.nan, "": np.nan})
        .fillna("Unknown")
        .replace({"None": "Unknown"})   # guard against literal “None”
    )


def _month_from_cases(df: pd.DataFrame) -> pd.Series:
    """
    Cases month:
      - Prefer 'Create Date'
      - Fallbacks: 'Start Date', 'Report Date'
      -> returns Period[M]
    """
    dcol = _first_col(df, ["Create Date", "Create Date (cases)", "Start Date", "Report Date"])
    if dcol is None:
        # last resort: first column containing 'date'
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
      - Else any free-text 'Month' (assume 2025 when only month name present)
      - Else any column named like 'Complaint Date / Month'
    """
    dcol = _first_col(
        df,
        [
            "Date Complaint Received - DD/MM/YY",
            "Date Complaint Received",
            "Complaint Date",
            "Complaint Date / Month",
        ],
    )
    if dcol is not None:
        dt = pd.to_datetime(df[dcol], errors="coerce", dayfirst=True, utc=False)
        return dt.dt.to_period("M")

    mcol = _first_col(df, ["Month"])
    if mcol is not None:
        raw = df[mcol].astype(str).str.strip()

        def _to_period(x: str) -> Optional[pd.Period]:
            if not x or x.lower() == "nan":
                return pd.NaT
            # Month name only -> assume 2025
            if re.fullmatch(r"[A-Za-z]{3,9}", x):
                dt = pd.to_datetime(f"1 {x} 2025", errors="coerce", dayfirst=True)
                return dt.to_period("M") if pd.notna(dt) else pd.NaT
            # Try free parse
            dt = pd.to_datetime(x, errors="coerce", dayfirst=True)
            return dt.to_period("M") if pd.notna(dt) else pd.NaT

        return raw.map(_to_period)

    # empty
    return pd.Series(pd.PeriodIndex([], freq="M"), index=df.index)


# --- Text → RCA2 (WHAT) bucketing ---
def _reason_rca2_map(text: str) -> str:
    """Keyword bucketing into RCA2-like buckets, tuned for your data."""
    if not isinstance(text, str) or not text.strip():
        return "Other"
    t = text.lower()

    # Order matters (more specific first)
    if any(k in t for k in ["death", "bereav"]):
        return "Death benefits payout"
    if "pension increase" in t or re.search(r"\bpi\b", t):
        return "Pension Increase"
    if "overpay" in t or "over pay" in t:
        return "Overpayment"
    if any(k in t for k in ["manual", "calc", "calculation"]):
        return "Manual calculation"
    if "timescale" in t or "time scale" in t or "sla" in t:
        return "Aptia standard Timescale"
    if "scheme rule" in t or "rules" in t:
        return "Scheme Rules"
    if "factor change" in t or "drop in value" in t:
        return "Drop in value/ factor change"
    if "postal" in t:
        return "Postal delay"
    if "avc" in t:
        return "Delay – AVC"
    if "requirement not checked" in t or "not checked" in t:
        return "Delay – Requirement not checked"
    if "case not created" in t:
        return "Delay – Case not created"
    if "2nd review" in t or "second review" in t:
        return "Delay – 2nd review"
    if "communicat" in t or "letter" in t or "clarity" in t:
        return "Communication"
    if "document" in t or "missing info" in t or "incomplete" in t:
        return "Documentation missing / incomplete"
    if "system" in t or "technical" in t or "platform" in t:
        return "System"

    return "Other"


# --- RCA1 (higher level) mapper from text ---
def _reason_rca1_map(text: str) -> str:
    """
    High-level WHY buckets:
      Delay, Procedure, Communication, System, Incorrect/Incomplete information, Other
    """
    if not isinstance(text, str) or not text.strip():
        return "Other"
    t = text.lower()

    # Delay family
    if ("delay" in t) or any(k in t for k in [
        "postal delay", "2nd review", "second review", "not checked", "case not created", "avc delay"
    ]):
        return "Delay"

    # Procedure / process policy
    if any(k in t for k in ["manual", "calc", "calculation", "timescale", "time scale",
                            "scheme rule", "rules", "procedure", "policy", "factor change",
                            "pension increase", "overpay", "over pay", "death"]):
        return "Procedure"

    # Communication
    if "communicat" in t or "letter" in t or "clarity" in t or "explain" in t:
        return "Communication"

    # System / tooling
    if "system" in t or "technical" in t or "platform" in t or "it issue" in t:
        return "System"

    # Incorrect or incomplete information
    if "wrong" in t or "incorrect" in t or "incomplete" in t or "missing" in t or "documentation":
        return "Incorrect/Incomplete information"

    return "Other"


# --- RCA1 from RCA2 (fallback if text missing) ---
_RCA2_TO_RCA1 = {
    # Delay umbrella
    "Postal delay": "Delay",
    "Delay – AVC": "Delay",
    "Delay – Requirement not checked": "Delay",
    "Delay – Case not created": "Delay",
    "Delay – 2nd review": "Delay",

    # Procedure umbrella
    "Manual calculation": "Procedure",
    "Aptia standard Timescale": "Procedure",
    "Scheme Rules": "Procedure",
    "Drop in value/ factor change": "Procedure",
    "Pension Increase": "Procedure",
    "Death benefits payout": "Procedure",
    "Overpayment": "Procedure",

    # Communication
    "Communication": "Communication",

    # System
    "System": "System",

    # Incorrect / incomplete
    "Documentation missing / incomplete": "Incorrect/Incomplete information",

    # default
    "Other": "Other",
}


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
    return [palette[i % len(palette)] for i in range(n)]


# -----------------------------
# Main question
# -----------------------------
def run(store, params: Optional[Dict] = None, user_text: Optional[str] = None) -> Tuple[str, pd.DataFrame]:
    """
    Render: Complaint analysis — Jun 2025 (by portfolio)
    - Row 1: Portfolio table with Total row + MoM line (Jan–Jun'25) — (unchanged)
    - Row 2: RCA2 (WHAT) Top-80% table + RCA1 (WHY) bar chart
    """
    params = params or {}
    month_key = pd.Period("2025-06", freq="M")  # fixed to June '25 per spec

    # 1) Pull data
    cases = getattr(store, "cases", None) or getattr(store, "raw_cases", None) or store.get("cases")
    complaints = getattr(store, "complaints", None) or getattr(store, "raw_complaints", None) or store.get("complaints")
    if cases is None or complaints is None:
        st.error("Missing data. Need both cases and complaints data.")
        return "complaints_june_by_portfolio", pd.DataFrame()

    # Identify columns flexibly
    case_id_col = _first_col(cases, ["Case ID", "id", "Original Process Affected Case ID"])
    portfolio_cases_col = _first_col(cases, ["Portfolio", "Portfolio Name"])
    if case_id_col is None:
        # synthesize an id to keep groupby safe
        cases = cases.copy()
        cases["_synthetic_case_id"] = np.arange(len(cases))
        case_id_col = "_synthetic_case_id"

    cases = cases.copy()
    cases["_month"] = _month_from_cases(cases)
    cases["_portfolio"] = (
        _ensure_portfolio_series(cases[portfolio_cases_col]) if portfolio_cases_col else pd.Series("Unknown", index=cases.index)
    )

    # Complaints columns
    compl_case_id_col = _first_col(complaints, ["Original Process Affected Case ID", "Case ID", "id"])
    portfolio_compl_col = _first_col(complaints, ["Portfolio", "Portfolio Name"])
    rca_text_col = _first_col(
        complaints,
        ["Brief Description - RCA done by admin", "RCA2", "RCA 2", "RCA", "RCA1", "Complaint Reason"],
    )

    complaints = complaints.copy()
    complaints["_month"] = _month_from_complaints(complaints)
    if portfolio_compl_col is None:
        # if not present, try to pick from cases via join on id
        if compl_case_id_col and case_id_col:
            tmp = complaints[[compl_case_id_col]].merge(
                cases[[case_id_col, "_portfolio"]],
                left_on=compl_case_id_col,
                right_on=case_id_col,
                how="left",
            )
            complaints["_portfolio"] = _ensure_portfolio_series(tmp["_portfolio"])
        else:
            complaints["_portfolio"] = "Unknown"
    else:
        complaints["_portfolio"] = _ensure_portfolio_series(complaints[portfolio_compl_col])

    # Reason bucketing from free text
    if rca_text_col:
        complaints["_rca2"] = complaints[rca_text_col].map(_reason_rca2_map)
        complaints["_rca1"] = complaints[rca_text_col].map(_reason_rca1_map)
    else:
        # fallback to “Other”
        complaints["_rca2"] = "Other"
        complaints["_rca1"] = "Other"

    # If any RCA1 is missing but RCA2 exists, infer from RCA2
    mask_missing_rca1 = complaints["_rca1"].isna() | (complaints["_rca1"].astype(str).str.strip() == "")
    if mask_missing_rca1.any():
        complaints.loc[mask_missing_rca1, "_rca1"] = complaints.loc[mask_missing_rca1, "_rca2"].map(_RCA2_TO_RCA1).fillna("Other")

    # -----------------------------
    # A. Portfolio table (June only)  — Row 1 (unchanged)
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
    table = table.reset_index().rename(columns={"_portfolio": "portfolio"}).sort_values("portfolio")

    # add Total row (at top)
    total_row = pd.DataFrame(
        {
            "portfolio": ["Total"],
            "cases": [int(table["cases"].sum())],
            "complaints": [int(table["complaints"].sum())],
            "per_1000": [
                table["complaints"].sum() / (table["cases"].sum() / 1000.0)
                if table["cases"].sum() > 0
                else np.nan
            ],
        }
    )
    table_display = pd.concat([total_row, table], ignore_index=True)

    st.markdown("### Complaint analysis — Jun 2025 (by portfolio)")
    st.caption(
        f"Total: cases={int(table['cases'].sum()):,}, complaints={int(table['complaints'].sum()):,}, "
        f"per_1000={table_display.loc[0, 'per_1000']:.2f}"
    )

    # ---- Row 1: table (left) + MoM line (right)
    c1, c2 = st.columns([1.1, 1.2], gap="large")
    with c1:
        st.dataframe(table_display, use_container_width=True)

    with c2:
        # --------------------------------------
        # B. MoM Complaints/1000 line (Jan–Jun) — unchanged
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
        offset = (max(per1000_m) * 0.04) if max(per1000_m) > 0 else 0.1
        for i, y in enumerate(per1000_m):
            ax.text(i, y + offset, f"{y:.2f}", ha="center", va="bottom", fontsize=9)

        _soft_line(ax)
        ax.set_xticks(range(len(months_2025)))
        ax.set_xticklabels(xlabels)
        ax.set_title("Complaints per 1,000 — Jan–Jun 2025", fontsize=11)
        st.pyplot(fig, use_container_width=True)

    # ---- Row 2: RCA2 table (left) + RCA1 bar (right)
    st.markdown("---")
    c3, c4 = st.columns([1.0, 1.2], gap="large")

    # --------------------------------------
    # C. June reasons (RCA2) Top-80 table
    # --------------------------------------
    reasons2 = (
        compl_jun["_rca2"]
        .value_counts(dropna=False)
        .rename_axis("reason")
        .to_frame("count")
        .reset_index()
    )
    reasons2["percent"] = (reasons2["count"] / max(reasons2["count"].sum(), 1)) * 100.0

    with c3:
        st.markdown("### RCA2 — Top 80% (June 2025)")
        if reasons2.empty:
            st.info("No June complaints to show.")
        else:
            r2 = reasons2.sort_values("count", ascending=False).reset_index(drop=True).copy()
            r2["cum_percent"] = r2["percent"].cumsum()
            top80 = r2[r2["cum_percent"] <= 80.0]
            if top80.empty and not r2.empty:
                top80 = r2.iloc[[0]].copy()
            st.dataframe(top80, use_container_width=True)

    # --------------------------------------
    # D. June RCA1 bar (higher level)
    # --------------------------------------
    reasons1 = (
        compl_jun["_rca1"]
        .value_counts(dropna=False)
        .rename_axis("rca1")
        .to_frame("count")
        .reset_index()
    ).sort_values("count", ascending=False)

    with c4:
        st.markdown("### RCA1 — June 2025")
        fig2, ax2 = plt.subplots(figsize=(6.6, 3.6), dpi=150)
        cols = _pastel_colors(len(reasons1))
        bars = ax2.bar(reasons1["rca1"], reasons1["count"], color=cols, edgecolor="none")
        for b in bars:
            h = b.get_height()
            ax2.text(b.get_x() + b.get_width() / 2, h + max(reasons1["count"]) * 0.03, f"{int(h)}",
                     ha="center", va="bottom", fontsize=9)
        _soft_line(ax2)
        ax2.set_xticklabels(reasons1["rca1"], rotation=20, ha="right")
        ax2.set_title("June reasons (RCA1)", fontsize=11)
        st.pyplot(fig2, use_container_width=True)

    # Done — no third row rendered.
    return "complaints_june_by_portfolio", table_display
