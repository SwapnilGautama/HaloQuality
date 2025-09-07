# -*- coding: utf-8 -*-
# questions/complaints_june_by_portfolio.py
from __future__ import annotations

from typing import Dict, Optional, Tuple, List
import re
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# Utilities
# ------------------------------------------------------------

def _get_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    cols = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in cols:
            return cols[c.lower()]
    return None

def _norm_portfolio(s: pd.Series) -> pd.Series:
    if s is None:
        return s
    return (
        s.astype(str)
         .str.strip()
         .str.replace(r"\s+", " ", regex=True)
         .str.replace("_", " ", regex=False)
         .str.replace("-", "-", regex=False)  # keep hyphen (e.g., Baes-Leatherhead)
         .str.title()
         .replace({"Baes Leatherhead": "Baes-Leatherhead"})
    )

def _to_month(dt_series: pd.Series, dayfirst=True) -> pd.Series:
    # Robust month (Period 'M') parsing
    dt = pd.to_datetime(dt_series, errors="coerce", dayfirst=dayfirst)
    return dt.dt.to_period("M")

def _month_from_text(mtext: pd.Series, year: int) -> pd.Series:
    # Accepts "Jun", "June", "JUN" → 2025-06 as Period('M')
    mtext = mtext.astype(str).str.strip().str.lower()
    mon_map = {
        "jan":1,"january":1,"feb":2,"february":2,"mar":3,"march":3,
        "apr":4,"april":4,"may":5,"jun":6,"june":6,"jul":7,"july":7,
        "aug":8,"august":8,"sep":9,"sept":9,"september":9,
        "oct":10,"october":10,"nov":11,"november":11,"dec":12,"december":12
    }
    mm = mtext.map(mon_map).astype("Int64")
    # Build periods only where we have a month number
    out = pd.PeriodIndex(year=year, month=1, freq="M")
    s = pd.Series(pd.PeriodIndex(year=year, month=1, freq="M"), index=mtext.index)
    s[:] = pd.NaT
    mask = mm.notna()
    s.loc[mask] = pd.PeriodIndex(year=year, month=mm[mask].astype(int), freq="M")
    return s

# ------------------------------------------------------------
# Text classification for RCA1 / RCA2 from "Brief Description - RCA done by admin"
# ------------------------------------------------------------

def _clean_text(s: pd.Series) -> pd.Series:
    return (
        s.fillna("")
         .astype(str)
         .str.strip()
         .str.replace(r"\s+", " ", regex=True)
         .str.lower()
    )

# RCA2 (more granular) via regex keyword mapping
RCA2_MAP: List[Tuple[str, str]] = [
    (r"\bscheme rules?\b", "Scheme Rules"),
    (r"\baptia (standard )?timescale\b", "Aptia standard Timescale"),
    (r"\bdrop in value|factor change\b", "Drop in value/ factor change"),
    (r"\bdeath benefit", "Death benefits payout"),
    (r"\boverpayment\b", "Overpayment"),
    (r"\bpension increase\b", "Pension Increase"),
    (r"\btransfer (doc|document|documentation)\b", "Transfer Documentation"),
    (r"\bmanual calculat", "Manual calculation"),
    (r"\bsecond review|2nd review\b", "Delay 2nd Review"),
    (r"\bpostal delay\b", "Delay Postal Delay"),
    (r"\bavc\b", "Delay – AVC"),
    (r"\brequirement not checked\b", "Delay Requirement not checked"),
    (r"\bcase not created\b", "Delay Case not created"),
    (r"\btrustee\b", "Delay – Trustee"),
    (r"\bverification|proof|kyc|identity\b", "Verification/Proof"),
    (r"\baddress\b", "Address/Contact"),
    (r"\bform|paperwork|document(s)? missing|missing doc", "Documentation Missing"),
    (r"\bincorrect|wrong|typo|mismatch\b", "Incorrect/Incomplete information"),
    (r"\bcommunicat|letter|email|call|phone\b", "Communication"),
    (r"\bsystem|portal|it issue|tech(nical)?|error\b", "System"),
    (r"\bdelay|overdue|late|sla|timescale|waiting|wait\b", "Delay (General)"),
]

# RCA1 buckets – short names for chart
def _rca1_from_text(text: str) -> str:
    if not text:
        return "Other"
    t = text.lower()

    if re.search(r"\bdelay|overdue|late|sla|timescale|waiting|wait\b", t):
        return "Delay"
    if re.search(r"\bprocedure|process|checklist|form|paperwork|document|transfer|verification|approval\b", t):
        return "Procedure"
    if re.search(r"\bcommunicat|letter|email|call|phone|contact|clarit\b", t):
        return "Communication"
    if re.search(r"\bsystem|portal|it|technical|error|bug|glitch\b", t):
        return "System"
    if re.search(r"\bincorrect|wrong|typo|incomplete|missing|mismatch\b", t):
        return "Incorrect/Incomplete info"
    return "Other"

def _rca2_from_text(text: str) -> str:
    if not text:
        return "Other"
    t = text.lower()
    for pat, label in RCA2_MAP:
        if re.search(pat, t):
            return label
    return "Other"

# ------------------------------------------------------------
# Main renderer
# ------------------------------------------------------------

def run(store: Dict[str, pd.DataFrame], params: Dict, user_text: str = "") -> Tuple[str, pd.DataFrame]:
    """Render:
       Row 1: Complaints/1000 table (Total on top) + MoM Jan–Jun'25 line chart
       Row 2: RCA2 Top-80% table + RCA1 June bar chart
    """

    cases = store.get("cases", pd.DataFrame()).copy()
    complaints = store.get("complaints", pd.DataFrame()).copy()

    # ---- Columns (robust aliases)
    col_case_id = _get_col(cases, ["Case ID", "Case_Id", "id"])
    col_case_dt = _get_col(cases, ["Create Date", "Start Date", "Report Date", "Date"])
    col_case_port = _get_col(cases, ["Portfolio", "portfolio"])

    col_comp_dt_full = _get_col(complaints, ["Date Complaint Received - DD/MM/YY", "Complaint Date", "Date"])
    col_comp_month_txt = _get_col(complaints, ["Month"])  # e.g., "June"
    col_comp_port = _get_col(complaints, ["Portfolio", "portfolio"])
    col_text = _get_col(complaints, ["Brief Description - RCA done by admin", "Brief Description", "Comments"])

    # Guardrails
    missing = []
    if col_case_id is None: missing.append("Case ID")
    if col_case_dt is None: missing.append("Create Date")
    if col_case_port is None: missing.append("Portfolio")
    if col_comp_port is None: missing.append("Portfolio (complaints)")
    if col_text is None: missing.append("Brief Description - RCA done by admin")

    if missing:
        st.info(f"Missing columns: {missing}")
        return ("complaints_june_by_portfolio", pd.DataFrame())

    # ---- Normalize portfolio
    cases[col_case_port] = _norm_portfolio(cases[col_case_port])
    complaints[col_comp_port] = _norm_portfolio(complaints[col_comp_port])

    # ---- Months
    cases["_month"] = _to_month(cases[col_case_dt])
    if col_comp_month_txt and complaints[col_comp_month_txt].notna().any():
        complaints["_month"] = _month_from_text(complaints[col_comp_month_txt], 2025)
    else:
        complaints["_month"] = _to_month(complaints[col_comp_dt_full], dayfirst=True)

    # Keep only valid months
    cases = cases.dropna(subset=["_month"])
    complaints = complaints.dropna(subset=["_month"])

    # --------------------------------------------------------
    # 1) Portfolio table for June 2025 (per_1000)
    # --------------------------------------------------------
    month_jun = pd.Period("2025-06", freq="M")
    cases_jun = cases.loc[cases["_month"] == month_jun]
    comp_jun = complaints.loc[complaints["_month"] == month_jun]

    by_port_cases = (cases_jun
                     .groupby(col_case_port, dropna=False)[col_case_id]
                     .nunique()
                     .rename("cases")
                     .reset_index())
    by_port_comp = (comp_jun
                    .groupby(col_comp_port, dropna=False)
                    .size()
                    .rename("complaints")
                    .reset_index())

    port_tbl = pd.merge(by_port_cases, by_port_comp,
                        left_on=col_case_port, right_on=col_comp_port, how="outer")
    port_tbl[col_case_port] = port_tbl[col_case_port].fillna(port_tbl[col_comp_port])
    port_tbl = port_tbl.drop(columns=[col_comp_port])

    port_tbl["cases"] = port_tbl["cases"].fillna(0).astype(int)
    port_tbl["complaints"] = port_tbl["complaints"].fillna(0).astype(int)
    port_tbl["per_1000"] = np.where(port_tbl["cases"] > 0,
                                    (port_tbl["complaints"] * 1000.0) / port_tbl["cases"],
                                    np.nan)

    # Total row on top
    total_row = pd.DataFrame({
        col_case_port: ["Total"],
        "cases": [int(port_tbl["cases"].sum())],
        "complaints": [int(port_tbl["complaints"].sum())]
    })
    total_row["per_1000"] = (total_row["complaints"] * 1000.0) / total_row["cases"]
    port_tbl = pd.concat([total_row, port_tbl], ignore_index=True)
    port_tbl = port_tbl[[col_case_port, "cases", "complaints", "per_1000"]].sort_values(
        by=[col_case_port], key=lambda s: s.replace({"Total": ""}), ignore_index=True
    )

    # --------------------------------------------------------
    # 2) MoM complaints/1000 (Jan–Jun 2025) – overall
    # --------------------------------------------------------
    want_months = pd.period_range("2025-01", "2025-06", freq="M")

    cases_m = (cases[cases["_month"].isin(want_months)]
               .groupby("_month")[col_case_id].nunique())
    comp_m  = (complaints[complaints["_month"].isin(want_months)]
               .groupby("_month").size())

    mom = (pd.DataFrame({"cases": cases_m, "complaints": comp_m})
             .reindex(want_months)
             .fillna(0))
    mom["per_1000"] = np.where(mom["cases"] > 0,
                               (mom["complaints"] * 1000.0) / mom["cases"],
                               0.0)

    # --------------------------------------------------------
    # 3) RCA text classification (June 2025) – RCA2 & RCA1
    # --------------------------------------------------------
    comp_jun["_text"] = _clean_text(comp_jun[col_text])

    # RCA2 granular
    comp_jun["RCA2"] = comp_jun["_text"].map(_rca2_from_text)

    rca2 = (comp_jun
            .groupby("RCA2")
            .size()
            .rename("count")
            .reset_index()
            .sort_values("count", ascending=False, ignore_index=True))

    total_comp = int(rca2["count"].sum()) if len(rca2) else 0
    if total_comp > 0:
        rca2["percent"] = (rca2["count"] / total_comp) * 100.0
        rca2["cum_percent"] = rca2["percent"].cumsum()
        # keep top 80%
        rca2_top = rca2.loc[rca2["cum_percent"] <= 80.0].copy()
        # if none passed 80 yet (e.g., first row > 80), keep the head(1)
        if rca2_top.empty and not rca2.empty:
            rca2_top = rca2.head(1).copy()
    else:
        rca2_top = rca2.copy()

    # RCA1 buckets
    comp_jun["RCA1"] = comp_jun["_text"].map(_rca1_from_text)
    rca1 = (comp_jun
            .groupby("RCA1")
            .size()
            .rename("count")
            .reset_index())

    # Put categories in a preferred order
    cat_order = ["Delay", "Procedure", "Communication", "System", "Incorrect/Incomplete info", "Other"]
    rca1["order"] = rca1["RCA1"].map({c:i for i,c in enumerate(cat_order)})
    rca1 = rca1.sort_values(["order", "RCA1"], ignore_index=True).drop(columns=["order"], errors="ignore")

    # --------------------------------------------------------
    # Render
    # --------------------------------------------------------
    st.caption(f"Total: cases={int(mom['cases'].sum())}, complaints={int(mom['complaints'].sum())}, per_1000={mom['per_1000'].mean():0.2f}")

    # ------ Row 1: table + line ------
    c1, c2 = st.columns([1.1, 0.9], gap="large")

    with c1:
        st.dataframe(
            port_tbl.rename(columns={col_case_port: "portfolio"}),
            use_container_width=True
        )

    with c2:
        fig, ax = plt.subplots(figsize=(6.2, 3.2), dpi=150)

        x = np.arange(len(want_months))
        y = mom["per_1000"].to_numpy()

        # soft pastel
        line_color = "#7FB3D5"  # light teal/blue
        ax.plot(x, y, marker="o", linewidth=2.5, color=line_color)

        # labels above points
        for i, val in enumerate(y):
            ax.text(i, val + (max(y)*0.04 if max(y) > 0 else 0.15), f"{val:0.2f}",
                    ha="center", va="bottom", fontsize=9)

        # style: no top/right/left spines, no y-axis/grid, only Jan–Jun ticks
        for spine in ["top", "right", "left", "bottom"]:
            ax.spines[spine].set_visible(False)
        ax.set_yticks([])
        ax.set_xticks(x)
        ax.set_xticklabels([p.strftime("%b") for p in want_months])
        ax.tick_params(axis="x", length=0, labelsize=10)
        ax.set_title("Complaints per 1,000 — MoM (Jan–Jun '25)", pad=6, fontsize=11)
        st.pyplot(fig, clear_figure=True)

    # ------ Row 2: RCA2 table + RCA1 bar ------
    c3, c4 = st.columns([1.1, 0.9], gap="large")

    with c3:
        st.subheader("RCA2 — Top 80% (June 2025)")
        if not rca2_top.empty:
            show = rca2_top.copy()
            show["percent"] = show["percent"].map(lambda v: f"{v:0.1f}")
            show["cum_percent"] = show["cum_percent"].map(lambda v: f"{v:0.1f}")
            st.dataframe(show, use_container_width=True)
        else:
            st.info("No RCA2 items found for June 2025.")

    with c4:
        st.subheader("RCA1 — June 2025")
        if not rca1.empty:
            fig2, ax2 = plt.subplots(figsize=(6.2, 3.2), dpi=150)

            # pastel palette
            palette = ["#AED6F1", "#A9DFBF", "#F9E79F", "#F5B7B1", "#D7BDE2", "#E5E7E9"]
            bars = ax2.bar(np.arange(len(rca1)), rca1["count"].to_numpy(),
                           color=palette[:len(rca1)], edgecolor="none")

            # no frame/grid/y-axis
            for spine in ["top", "right", "left", "bottom"]:
                ax2.spines[spine].set_visible(False)
            ax2.set_yticks([])

            # x labels
            ax2.set_xticks(np.arange(len(rca1)))
            ax2.set_xticklabels(rca1["RCA1"].tolist(), rotation=20, ha="right", fontsize=9)

            # data labels
            for rect, val in zip(bars, rca1["count"].tolist()):
                ax2.text(rect.get_x() + rect.get_width()/2, rect.get_height() + max(rca1["count"])*0.03,
                         f"{val}", ha="center", va="bottom", fontsize=9)

            st.pyplot(fig2, clear_figure=True)
        else:
            st.info("No RCA1 items found for June 2025.")

    # Return a name + the main table so the app contract stays intact
    return ("complaints_june_by_portfolio", port_tbl)
