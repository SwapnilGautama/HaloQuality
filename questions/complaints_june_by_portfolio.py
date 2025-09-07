# questions/complaints_june_by_portfolio.py
# -*- coding: utf-8 -*-

from __future__ import annotations
import re
from typing import Dict, Iterable, Tuple, List

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------
# Theming / CSS
# ---------------------------------------------------------------------
DARK_BLUE = "#0B3A77"   # headers
DARK_GREY = "#2F2F2F"   # body text
SOFT_GREY = "#DADDE3"   # axis line
PASTEL   = "#7EA6E0"    # line color (pastel blue)
BAR_COLORS = ["#7EA6E0", "#7BC6B7", "#F3B880", "#B6B8C8", "#F2A8A8", "#C9CDD7"]

st.markdown(
    f"""
    <style>
      /* titles */
      h1, h2, h3, h4 {{
        color: {DARK_BLUE} !important;
      }}
      /* body text */
      .stMarkdown, .stText, .st-emotion-cache-ue6h4q, .st-emotion-cache-1v0mbdj, .stDataFrame {{
        color: {DARK_GREY} !important;
      }}
      /* tighten section padding very slightly */
      .block-container {{ padding-top: 1.1rem; }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------
# Column normalization helpers
# ---------------------------------------------------------------------
def _first_present(df: pd.DataFrame, choices: Iterable[str]) -> str | None:
    for c in choices:
        if c in df.columns:
            return c
    return None

def _norm_columns(cases: pd.DataFrame, complaints: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, str]]:
    m = {}

    cases = cases.copy()
    complaints = complaints.copy()

    # Portfolio
    m["c_port"] = _first_present(cases, ["Portfolio", "portfolio"])
    m["q_port"] = _first_present(complaints, ["Portfolio", "portfolio"])
    if not m["c_port"] or not m["q_port"]:
        raise ValueError("Missing 'Portfolio' column in cases/complaints.")

    # Create date in cases
    m["c_date"] = _first_present(
        cases, ["Create Date (cases)", "Create Date", "Start Date", "Create Dt", "Create_Date"]
    )
    if not m["c_date"]:
        raise ValueError("Missing 'Create Date (cases)' in cases.")

    # Complaints date/month
    m["q_date"] = _first_present(
        complaints,
        [
            "Date Complaint Received - DD/MM/YY",
            "Complaint Date / Month",
            "Date",
            "Month",
            "month",
        ],
    )
    if not m["q_date"]:
        raise ValueError("Missing complaint date/month field in complaints.")

    # Normalize month keys
    cases["_month_key"] = pd.to_datetime(cases[m["c_date"]], errors="coerce").dt.to_period("M").astype(str)

    qcol = m["q_date"]
    if complaints[qcol].dtype == "O":
        # Either explicit date or a month name (assume 2025 for month-only)
        def to_mkey(x: str) -> str:
            s = str(x).strip()
            # month name?
            if re.fullmatch(r"[A-Za-z]{{3,9}}", s):
                dt = pd.to_datetime(f"01-{s}-2025", dayfirst=True, errors="coerce")
            else:
                dt = pd.to_datetime(s, errors="coerce", dayfirst=True)
            return pd.NaT if pd.isna(dt) else str(pd.Period(dt, freq="M"))
        complaints["_month_key"] = complaints[qcol].map(to_mkey)
    else:
        complaints["_month_key"] = pd.to_datetime(complaints[qcol], errors="coerce").dt.to_period("M").astype(str)

    return cases, complaints, m

# ---------------------------------------------------------------------
# RCA mappers
# ---------------------------------------------------------------------
def _rca1_bucket(text: str) -> str:
    """
    Higher-level RCA1 buckets from free-text.
    """
    t = str(text).lower()

    delay_kw = [
        "delay", "manual calc", "manual calculation", "waiting", "await", "hold", "postal", "post", "late",
        "backlog", "sla", "timescale", "not created", "second review", "set up", "setup"
    ]
    proc_kw = [
        "procedure", "scheme rule", "rules", "factor change", "drop in value", "consent", "approval", "authori"
    ]
    comm_kw = [
        "communicat", "letter", "email", "contact", "incorrect form", "missing form", "documentation", "doc"
    ]
    system_kw = [
        "system", "portal", "workflow", "interface", "upload", "it issue", "file error"
    ]
    wrong_kw = [
        "incorrect", "incomplete", "wrong", "error", "typo", "mis-"
    ]

    def any_kw(kws: List[str]) -> bool:
        return any(k in t for k in kws)

    if any_kw(delay_kw):   return "Delay"
    if any_kw(proc_kw):    return "Procedure"
    if any_kw(comm_kw):    return "Communication"
    if any_kw(system_kw):  return "System"
    if any_kw(wrong_kw):   return "Incorrect/Incomplete info"
    return "Other"

def _rca2_reason(text: str) -> str:
    """
    Deeper RCA2 from admin brief description (keyword model).
    """
    t = str(text).lower()

    # ordered rules
    rules = [
        (r"waiting|await|hold|chase|remind|tpa|third ?party|employer",    "Waiting on member/TPA"),
        (r"bank|payment|bacs|cheque|bounce|re-issue|refund",              "Bank/Payment issue"),
        (r"postal|post|mail",                                            "Postal delay"),
        (r"manual calc|manual calculation|calc error|recalc",             "Manual calculation"),
        (r"trustee|trustees?",                                           "Trustee"),
        (r"\bavc\b|additional voluntary",                                "AVC"),
        (r"data entry|keying|typed|input error",                          "Data entry error"),
        (r"death benefit|dep.? benefit|dependant",                        "Death benefits payout"),
        (r"case not created|no case|missing case",                        "Case not created"),
        (r"pension set up|set.?up|onboarding",                             "Pension set up"),
    ]
    for pat, label in rules:
        if re.search(pat, t):
            return label
    return "Other"

# ---------------------------------------------------------------------
# Core rendering
# ---------------------------------------------------------------------
TITLE = "complaint analysis — June 2025 (by portfolio)"

def run(store, params: Dict | None = None, user_text: str | None = None):
    """
    Streamlit renderer for the question. Returns a tuple (slug, df_for_app) to
    keep app expectations intact, while drawing all visuals here.
    """
    cases = store.cases
    complaints = store.complaints

    # Normalize columns
    cases, complaints, m = _norm_columns(cases, complaints)

    # June 2025 join key
    june_key = "2025-06"

    # -----------------------------
    # Table: Complaints per 1,000 by portfolio (June 2025)
    # -----------------------------
    c_jun = cases.query("_month_key == @june_key")
    q_jun = complaints.query("_month_key == @june_key")

    tbl = (
        c_jun.groupby(m["c_port"], as_index=False)
             .size()
             .rename(columns={"size": "cases"})
        .merge(
            q_jun.groupby(m["q_port"], as_index=False)
                .size()
                .rename(columns={"size": "complaints"}),
            left_on=m["c_port"], right_on=m["q_port"], how="left"
        )
    )

    # Align portfolio column and fill NaNs
    tbl["complaints"] = tbl["complaints"].fillna(0).astype(int)
    tbl["per_1000"] = (tbl["complaints"] / tbl["cases"] * 1000).replace([np.inf, -np.inf], np.nan)

    # Total row
    total_row = pd.DataFrame({
        m["c_port"]: ["Total"],
        "cases": [tbl["cases"].sum()],
        "complaints": [tbl["complaints"].sum()],
        "per_1000": [np.where(tbl["cases"].sum()==0, np.nan, tbl["complaints"].sum()/tbl["cases"].sum()*1000)]
    })
    tbl = pd.concat([total_row, tbl[[m["c_port"], "cases", "complaints", "per_1000"]]], ignore_index=True)
    tbl = tbl.rename(columns={m["c_port"]: "portfolio"}).sort_values("portfolio").reset_index(drop=True)

    # -----------------------------
    # Line: MoM Complaints per 1,000 (Jan–Jun 2025)
    # -----------------------------
    months = pd.period_range("2025-01", "2025-06", freq="M").astype(str).tolist()
    cases_m = (cases.query("_month_key in @months")
                    .groupby("_month_key").size().reindex(months, fill_value=0))
    comp_m  = (complaints.query("_month_key in @months")
                    .groupby("_month_key").size().reindex(months, fill_value=0))
    per1000_m = (comp_m / cases_m.replace(0, np.nan) * 1000).fillna(0)

    # -----------------------------
    # RCA blocks (June 2025)
    # -----------------------------
    desc_col = _first_present(complaints, [
        "Brief Description - RCA done by admin",
        "Brief Description – RCA done by admin",
        "Brief Description - RCA (admin)",
        "RCA admin brief",
    ])
    if not desc_col:
        # fallback to whatever text field exists
        desc_col = _first_present(complaints, ["Description", "Brief Description", "Comments"]) or m["q_date"]

    qj_text = complaints.query("_month_key == @june_key")[desc_col].fillna("")

    # RCA1
    rca1 = qj_text.map(_rca1_bucket).value_counts().rename_axis("RCA1").reset_index(name="count")
    # consistent order
    order = ["Delay", "Procedure", "Communication", "System", "Incorrect/Incomplete info", "Other"]
    rca1["order"] = rca1["RCA1"].apply(lambda x: order.index(x) if x in order else len(order))
    rca1 = rca1.sort_values(["order", "RCA1"]).drop(columns="order")

    # RCA2 top 80%
    rca2_all = qj_text.map(_rca2_reason).value_counts().rename_axis("RCA2").reset_index(name="count")
    rca2_all["percent"] = rca2_all["count"] / max(rca2_all["count"].sum(), 1) * 100
    rca2_all["cum_percent"] = rca2_all["percent"].cumsum()
    rca2_top = rca2_all[rca2_all["cum_percent"] <= 80].copy()
    if rca2_top.empty and not rca2_all.empty:
        rca2_top = rca2_all.head(10).copy()  # safety

    # -----------------------------------------------------------------
    # RENDER
    # -----------------------------------------------------------------
    st.markdown(f"### Complaint analysis — Jun 2025 (by portfolio)")
    c1, c2 = st.columns((1.05, 1.0), gap="large")

    with c1:
        st.markdown("#### Complaints per 1,000 — by portfolio (June 2025)")
        show = tbl.copy()
        show["per_1000"] = show["per_1000"].round(4)
        st.dataframe(
            show,
            hide_index=True,
            use_container_width=True,
        )

    with c2:
        st.markdown("#### Complaints per 1,000 — MoM (Jan–Jun ’25)")
        # Build custom minimalist chart
        fig, ax = plt.subplots(figsize=(6, 3.2), dpi=150)
        x = np.arange(len(months))
        y = per1000_m.values

        # smooth-ish line – same points but no markers beyond labelled dots
        ax.plot(x, y, color=PASTEL, linewidth=2.5, solid_capstyle="round")

        # point labels
        for xi, yi in zip(x, y):
            ax.scatter([xi], [yi], color=PASTEL, s=20, zorder=3)
            ax.text(xi, yi + (max(y) * 0.04 if max(y) else 0.08), f"{yi:.2f}", ha="center", va="bottom", fontsize=9, color=DARK_GREY)

        # axes styling: remove y, soft x line
        ax.get_yaxis().set_visible(False)
        for spine in ["left", "right", "top"]:
            ax.spines[spine].set_visible(False)
        ax.spines["bottom"].set_color(SOFT_GREY)
        ax.spines["bottom"].set_linewidth(1.5)

        ax.set_xticks(x)
        # show MMM on x axis
        labels = [pd.Period(m, freq="M").to_timestamp().strftime("%b") for m in months]
        ax.set_xticklabels(labels, fontsize=9, color=DARK_GREY)

        ax.grid(False)
        ax.set_xlabel("")  # minimalist
        ax.set_ylabel("")
        st.pyplot(fig, clear_figure=True, use_container_width=True)

    # Row 2: swap places (RCA1 chart LEFT, RCA2 table RIGHT)
    st.markdown("---")
    r1, r2 = st.columns((1.0, 1.05), gap="large")

    with r1:
        st.markdown("#### RCA1 — June 2025")
        # Bar chart with soft style
        fig2, ax2 = plt.subplots(figsize=(6, 3.2), dpi=150)
        names = rca1["RCA1"].tolist()
        vals = rca1["count"].tolist()
        xpos = np.arange(len(names))
        colors = (BAR_COLORS * ((len(names) // len(BAR_COLORS)) + 1))[:len(names)]

        ax2.bar(xpos, vals, color=colors, edgecolor="white", linewidth=0)
        # labels
        for xi, yi in zip(xpos, vals):
            ax2.text(xi, yi + (max(vals) * 0.03 if max(vals) else 0.3), f"{yi}", ha="center", va="bottom", fontsize=9, color=DARK_GREY)

        # axes cleanup
        ax2.get_yaxis().set_visible(False)
        for s in ["left", "right", "top"]:
            ax2.spines[s].set_visible(False)
        ax2.spines["bottom"].set_color(SOFT_GREY)
        ax2.spines["bottom"].set_linewidth(1.5)

        ax2.set_xticks(xpos)
        ax2.set_xticklabels(names, rotation=0, ha="center", fontsize=9, color=DARK_GREY)
        ax2.grid(False)
        ax2.set_xlabel("")
        ax2.set_ylabel("")

        st.pyplot(fig2, clear_figure=True, use_container_width=True)

    with r2:
        st.markdown("#### RCA2 — Top 80% (June 2025)")
        r2show = rca2_top[["RCA2", "count", "percent", "cum_percent"]].copy()
        r2show["percent"] = r2show["percent"].round(2)
        r2show["cum_percent"] = r2show["cum_percent"].round(2)
        st.dataframe(r2show, hide_index=True, use_container_width=True)

    # Return the main table to keep app's expectations intact.
    return ("complaints_june_by_portfolio", tbl)
