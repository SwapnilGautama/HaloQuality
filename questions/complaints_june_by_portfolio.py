# -*- coding: utf-8 -*-
# questions/complaints_june_by_portfolio.py
from __future__ import annotations

from typing import Optional, Dict, Tuple, Iterable
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# ────────────────────────────────────────────────────────────────────────────────
# Column discovery helpers (robust to naming drift)
# ────────────────────────────────────────────────────────────────────────────────
def _first_col(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    lc = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lc:
            return lc[c.lower()]
    return None

def _ensure_portfolio_series(s: pd.Series) -> pd.Series:
    return (
        s.astype(str)
        .str.strip()
        .replace({"nan": np.nan, "": np.nan})
        .fillna("Unknown")
    )

# ────────────────────────────────────────────────────────────────────────────────
# Month parsing (cases)
# ────────────────────────────────────────────────────────────────────────────────
def _month_from_cases(df: pd.DataFrame) -> pd.Series:
    dcol = _first_col(df, ["Create Date", "Create Date (cases)", "Start Date", "Report Date"])
    if dcol is None:
        date_like = [c for c in df.columns if "date" in c.lower()]
        dcol = date_like[0] if date_like else None
    if dcol is None:
        return pd.Series(pd.PeriodIndex([], freq="M"), index=df.index)

    dt = pd.to_datetime(df[dcol], errors="coerce", dayfirst=True)
    return dt.dt.to_period("M")

# ────────────────────────────────────────────────────────────────────────────────
# Month parsing (complaints)
# ────────────────────────────────────────────────────────────────────────────────
def _month_from_complaints(df: pd.DataFrame) -> pd.Series:
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
        dt = pd.to_datetime(df[dcol], errors="coerce", dayfirst=True)
        return dt.dt.to_period("M")

    mcol = _first_col(df, ["Month"])
    if mcol is not None:
        raw = df[mcol].astype(str).str.strip()

        def _to_period(x: str) -> Optional[pd.Period]:
            if not x or x.lower() == "nan":
                return pd.NaT
            # Month word only -> assume 2025
            if re.fullmatch(r"[A-Za-z]{3,9}", x):
                dt = pd.to_datetime(f"1 {x} 2025", errors="coerce", dayfirst=True)
                return dt.to_period("M") if pd.notna(dt) else pd.NaT
            dt = pd.to_datetime(x, errors="coerce", dayfirst=True)
            return dt.to_period("M") if pd.notna(dt) else pd.NaT

        return raw.map(_to_period)

    return pd.Series(pd.PeriodIndex([], freq="M"), index=df.index)

# ────────────────────────────────────────────────────────────────────────────────
# RCA bucketing
# ────────────────────────────────────────────────────────────────────────────────
def _reason_rca2_map(text: str) -> str:
    """Map 'Brief Description - RCA done by admin' into RCA2 buckets."""
    if not isinstance(text, str) or not text.strip():
        return "Other"
    t = text.lower()

    # Specific before generic
    if any(k in t for k in ["death", "bereav"]):
        return "Death benefits payout"
    if "pension increase" in t or re.search(r"\bp(i|ension inc)\b", t):
        return "Pension Increase"
    if "overpay" in t or "over pay" in t:
        return "Overpayment"
    if "timescale" in t or "time scale" in t:
        return "Aptia standard Timescale"
    if "scheme rule" in t:
        return "Scheme Rules"
    if "factor change" in t or "drop in value" in t:
        return "Drop in value/ factor change"
    if "communicat" in t:
        return "Communication"
    if any(k in t for k in ["doc", "document", "form", "evidence", "incomplete", "incorrect", "missing"]):
        if "transfer" in t:
            return "Transfer Documentation"
        return "Documentation Missing"
    if any(k in t for k in ["system", "portal", "workflow", "it issue", "bug", "error"]):
        return "System Issue"

    # Delay subgrouping
    if "postal" in t or "post" in t:
        return "Delay Postal Delay"
    if "not checked" in t:
        return "Delay  Requirement not checked"
    if "case not created" in t:
        return "Delay  Case not created"
    if "2nd review" in t or "second review" in t:
        return "Delay  2nd Review"
    if "avc" in t:
        return "Delay – AVC"
    if "manual" in t or "calc" in t:
        return "Manual calculation"
    if "delay" in t:
        return "Delay (general)"

    return "Other"

def _rca1_from(rca2: str, text: Optional[str]) -> str:
    """Consolidate to RCA1: Delay, Procedure, Communication, System, Incorrect/Incomplete info, Other."""
    r = (rca2 or "").lower()
    t = (text or "").lower()

    if "delay" in r or any(k in t for k in ["delay", "manual", "calc", "postal", "not checked", "case not created", "2nd review", "avc"]):
        return "Delay"
    if "communicat" in r or "communicat" in t:
        return "Communication"
    if "system" in r or any(k in t for k in ["system", "portal", "workflow", "it issue", "bug", "error"]):
        return "System"
    if any(k in r for k in ["documentation", "transfer"]) or any(k in t for k in ["document", "form", "evidence", "incomplete", "incorrect", "missing"]):
        return "Incorrect/Incomplete information"
    if any(k in r for k in [
        "scheme rules",
        "timescale",
        "factor change",
        "pension increase",
        "overpayment",
        "death benefits payout",
        "drop in value"
    ]):
        return "Procedure"
    return "Other"

# ────────────────────────────────────────────────────────────────────────────────
# Aesthetics
# ────────────────────────────────────────────────────────────────────────────────
def _soft_line(ax):
    for s in ["top", "right", "left", "bottom"]:
        ax.spines[s].set_visible(False)
    ax.grid(False)
    ax.set_ylabel("")
    ax.set_yticks([])

def _pastel_colors(n: int) -> list[str]:
    palette = ["#8ecae6", "#bde0fe", "#cdeac0", "#ffd6a5", "#fbc4ab", "#cdb4db", "#b9fbc0"]
    return [palette[i % len(palette)] for i in range(n)]

# ────────────────────────────────────────────────────────────────────────────────
# Main question
# ────────────────────────────────────────────────────────────────────────────────
def run(store, params: Optional[Dict] = None, user_text: Optional[str] = None) -> Tuple[str, pd.DataFrame]:
    params = params or {}
    june25 = pd.Period("2025-06", freq="M")

    # Load
    cases = getattr(store, "cases", None) or getattr(store, "raw_cases", None) or store.get("cases")
    complaints = getattr(store, "complaints", None) or getattr(store, "raw_complaints", None) or store.get("complaints")
    if cases is None or complaints is None:
        st.error("Missing data. Need both cases and complaints.")
        return "complaints_june_by_portfolio", pd.DataFrame()

    # Case columns
    case_id_col = _first_col(cases, ["Case ID", "id", "Original Process Affected Case ID"])
    portfolio_cases_col = _first_col(cases, ["Portfolio", "Portfolio Name"])

    cases = cases.copy()
    if case_id_col is None:
        cases["_cid"] = np.arange(len(cases))
        case_id_col = "_cid"
    cases["_month"] = _month_from_cases(cases)
    cases["_portfolio"] = _ensure_portfolio_series(
        cases[portfolio_cases_col] if portfolio_cases_col else pd.Series("Unknown", index=cases.index)
    )

    # Complaint columns
    compl_case_id_col = _first_col(complaints, ["Original Process Affected Case ID", "Case ID", "id"])
    portfolio_compl_col = _first_col(complaints, ["Portfolio", "Portfolio Name"])
    rca_text_col = _first_col(
        complaints,
        ["Brief Description - RCA done by admin", "RCA2", "RCA 2", "RCA", "RCA1", "Complaint Reason"],
    )

    complaints = complaints.copy()
    complaints["_month"] = _month_from_complaints(complaints)

    if portfolio_compl_col is None and compl_case_id_col and case_id_col:
        # stitch portfolio from cases
        tmp = complaints[[compl_case_id_col]].merge(
            cases[[case_id_col, "_portfolio"]],
            left_on=compl_case_id_col,
            right_on=case_id_col,
            how="left",
        )
        complaints["_portfolio"] = _ensure_portfolio_series(tmp["_portfolio"])
    else:
        complaints["_portfolio"] = _ensure_portfolio_series(
            complaints[portfolio_compl_col] if portfolio_compl_col else pd.Series("Unknown", index=complaints.index)
        )

    # RCA2 + RCA1
    complaints["_rca2"] = complaints[rca_text_col].map(_reason_rca2_map) if rca_text_col else "Other"
    complaints["_rca1"] = complaints.apply(
        lambda r: _rca1_from(r["_rca2"], r.get(rca_text_col) if rca_text_col else ""), axis=1
    )

    # ── Portfolio table (June) ──────────────────────────────────────────────────
    cases_jun = cases.loc[cases["_month"] == june25]
    compl_jun = complaints.loc[complaints["_month"] == june25]

    cases_by_pf = cases_jun.groupby("_portfolio")[case_id_col].count().rename("cases").to_frame()
    comp_by_pf = compl_jun.groupby("_portfolio")["_rca2"].count().rename("complaints").to_frame()

    table = cases_by_pf.join(comp_by_pf, how="outer").fillna(0)
    table["cases"] = table["cases"].astype(int)
    table["complaints"] = table["complaints"].astype(int)
    table["per_1000"] = np.where(table["cases"] > 0, table["complaints"] / (table["cases"] / 1000.0), np.nan)
    table = table.reset_index().rename(columns={"_portfolio": "portfolio"}).sort_values("portfolio")

    total_row = pd.DataFrame(
        {
            "portfolio": ["Total"],
            "cases": [int(table["cases"].sum())],
            "complaints": [int(table["complaints"].sum())],
            "per_1000": [
                table["complaints"].sum() / (table["cases"].sum() / 1000.0) if table["cases"].sum() else np.nan
            ],
        }
    )
    table_display = pd.concat([total_row, table], ignore_index=True)

    st.markdown("### Complaint analysis — Jun 2025 (by portfolio)")
    st.caption(
        f"Total: cases={int(table['cases'].sum()):,}, complaints={int(table['complaints'].sum()):,}, "
        f"per_1000={table_display.loc[0, 'per_1000']:.3f if pd.notna(table_display.loc[0, 'per_1000']) else '–'}"
    )

    # ── Row 1: table + MoM line ────────────────────────────────────────────────
    c1, c2 = st.columns([1.1, 1.2], gap="large")
    with c1:
        st.dataframe(table_display, use_container_width=True)

    with c2:
        months_2025 = pd.period_range("2025-01", "2025-06", freq="M")
        cases_m = cases[cases["_month"].isin(months_2025)].groupby("_month")[case_id_col].count()
        compl_m = complaints[complaints["_month"].isin(months_2025)].groupby("_month")["_rca2"].count()
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
        # labels
        offset = (max(per1000_m) * 0.04) if max(per1000_m) > 0 else 0.1
        for i, y in enumerate(per1000_m):
            ax.text(i, y + offset, f"{y:.2f}", ha="center", va="bottom", fontsize=9)

        _soft_line(ax)
        ax.set_xticks(range(len(months_2025)))
        ax.set_xticklabels(xlabels)
        ax.set_title("Complaints per 1,000 — Jan–Jun 2025", fontsize=11)
        st.pyplot(fig, use_container_width=True)

    st.markdown("---")

    # ── Row 2: RCA2 table + RCA1 bar ───────────────────────────────────────────
    c3, c4 = st.columns([1.0, 1.2], gap="large")

    # RCA2 (Top 80%)
    reasons2 = (
        compl_jun["_rca2"]
        .value_counts(dropna=False)
        .rename_axis("reason")
        .to_frame("count")
        .reset_index()
    )
    reasons2["percent"] = (reasons2["count"] / max(reasons2["count"].sum(), 1)) * 100.0
    r2 = reasons2.sort_values("count", ascending=False).reset_index(drop=True)
    r2["cum_percent"] = r2["percent"].cumsum()
    top80 = r2[r2["cum_percent"] <= 80.0]
    if top80.empty and not r2.empty:
        top80 = r2.iloc[[0]].copy()

    with c3:
        st.markdown("### RCA2 — Top 80% (June 2025)")
        if top80.empty:
            st.info("No June complaints to show.")
        else:
            st.dataframe(top80, use_container_width=True)

    with c4:
        st.markdown("### RCA1 — June 2025")
        reasons1 = (
            compl_jun["_rca1"]
            .value_counts(dropna=False)
            .rename_axis("rca1")
            .to_frame("count")
            .reset_index()
        )
        # Order by count desc, keep canonical names if present
        order = ["Delay", "Procedure", "Communication", "System", "Incorrect/Incomplete information", "Other"]
        cat = pd.Categorical(reasons1["rca1"], categories=order, ordered=True)
        reasons1 = reasons1.assign(rca1=cat).sort_values(["rca1", "count"], ascending=[True, False])
        reasons1 = reasons1.dropna(subset=["rca1"])

        fig2, ax2 = plt.subplots(figsize=(6.6, 3.6), dpi=150)
        cols = _pastel_colors(len(reasons1))
        bars = ax2.bar(reasons1["rca1"].astype(str), reasons1["count"], color=cols, edgecolor="none")
        for b in bars:
            h = b.get_height()
            ax2.text(b.get_x() + b.get_width() / 2, h + (max(reasons1["count"]) * 0.03 if len(reasons1) else 0.1),
                     f"{int(h)}", ha="center", va="bottom", fontsize=9)
        _soft_line(ax2)
        ax2.set_xticklabels(reasons1["rca1"].astype(str), rotation=0, ha="center")
        ax2.set_title("June reasons (RCA1)", fontsize=11)
        st.pyplot(fig2, use_container_width=True)

    # IMPORTANT: return empty df so no extra table is rendered by the host app
    return "complaints_june_by_portfolio", pd.DataFrame()
