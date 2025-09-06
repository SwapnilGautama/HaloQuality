# questions/complaints_june_by_portfolio.py
from __future__ import annotations

import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from typing import Dict, Optional

# -------------------------
# Styling helpers
# -------------------------
def _pastel_palette():
    return ["#8ecae6", "#ffb703", "#219ebc", "#adb5bd", "#e5989b", "#bde0fe"]

def _section(title: str, caption: Optional[str] = None):
    st.subheader(title)
    if caption:
        st.caption(caption)

# -------------------------
# Month building (robust)
# -------------------------
_MON_MAP = {
    "jan": "01", "feb": "02", "mar": "03", "apr": "04",
    "may": "05", "jun": "06", "jul": "07", "aug": "08",
    "sep": "09", "oct": "10", "nov": "11", "dec": "12",
}

def _month_series_from_cases(df: pd.DataFrame) -> pd.Series:
    """
    Priority:
      1) _month_dt (already normalized by data_store)
      2) Create Date (text) -> to_period('M')
    """
    if "_month_dt" in df.columns:
        return pd.to_datetime(df["_month_dt"], errors="coerce").dt.to_period("M").astype(str)
    if "Create Date" in df.columns:
        return pd.to_datetime(df["Create Date"], errors="coerce").dt.to_period("M").astype(str)
    # Fallback empty
    return pd.Series(pd.NA, index=df.index, dtype="object")

def _month_series_from_complaints(df: pd.DataFrame) -> pd.Series:
    """
    Priority:
      1) _month_dt (already normalized by data_store)
      2) Date Complaint Received - DD/MM/YY
      3) Month (e.g., 'June'/'Jun') -> assume year 2025 per user rule
    """
    if "_month_dt" in df.columns:
        return pd.to_datetime(df["_month_dt"], errors="coerce").dt.to_period("M").astype(str)

    # 2) full date
    date_col = "Date Complaint Received - DD/MM/YY"
    if date_col in df.columns:
        s = pd.to_datetime(df[date_col], dayfirst=True, errors="coerce")
        return s.dt.to_period("M").astype(str)

    # 3) month name only => assume 2025
    if "Month" in df.columns:
        def to_key(x):
            if pd.isna(x):
                return pd.NA
            m3 = str(x).strip()[:3].lower()
            mm = _MON_MAP.get(m3)
            return f"2025-{mm}" if mm else pd.NA
        return df["Month"].map(to_key)

    return pd.Series(pd.NA, index=df.index, dtype="object")

# -------------------------
# Reason labelling (slide-aligned)
# -------------------------
_BUCKETS = {
    "Delay": [
        r"\bdelay\b", r"\bmanual calc", r"\bmanual calculation",
        r"\bpostal\b", r"\b2(nd)? review\b", r"\btimescale\b", r"\bslow\b", r"\blate\b"
    ],
    "Procedure": [
        r"\bscheme rules?\b", r"\brule\b", r"\bstandard timescale\b", r"\bSLA\b",
        r"\bprocess\b", r"\bprocedure\b"
    ],
    "Communication": [
        r"\bletter\b", r"\bcommunication\b", r"\bnot (informed|told|clear)\b",
        r"\bno (reply|response)\b", r"\bupdate\b"
    ],
    "System": [
        r"\bsystem\b", r"\bit issue\b", r"\bworkflow\b", r"\bplatform\b", r"\bbug\b", r"\berror\b"
    ],
    "Incorrect/Incomplete information": [
        r"\bincorrect\b", r"\bwrong\b", r"\bincomplete\b", r"\bmissing\b",
        r"\bnot provided\b", r"\bno evidence\b"
    ],
}

_SUBREASONS = {
    "Delay Manual calculation": [r"\bmanual calc(ulation)?\b"],
    "Aptia standard Timescale": [r"\bstandard timescale\b", r"\btimescale\b", r"\bSLA\b"],
    "Delay Pension set up": [r"\bpension set up\b", r"\bsetup\b"],
    "Delay Postal Delay": [r"\bpostal\b", r"\bpost\b", r"\bmail\b"],
    "Delay – AVC": [r"\bAVC\b"],
    "Delay Requirement not checked": [r"\brequirement not checked\b", r"\bnot checked\b"],
    "Delay Case not created": [r"\bcase not created\b"],
    "Delay 2nd Review": [r"\b2(nd)? review\b", r"\bsecond review\b"],
    "Delay – Trustee": [r"\btrustee\b"],
    "Scheme Rules": [r"\bscheme rules?\b"],
    "Drop in value/ factor change": [r"\bfactor change\b", r"\bdrop in value\b"],
    "Death benefits payout": [r"\bdeath benefit(s)?\b"],
    "Overpayment": [r"\boverpayment\b"],
    "Pension Increase": [r"\bpension increase\b"],
    "Transfer Documentation": [r"\btransfer doc(umentation)?\b"],
}

import re
def _first_match(text: str, pats: list[str]) -> bool:
    for pat in pats:
        if re.search(pat, text, flags=re.I):
            return True
    return False

def _label_reasons(df: pd.DataFrame, text_col: str) -> pd.DataFrame:
    out = df.copy()
    t = out[text_col].fillna("").astype(str)

    buckets = []
    for s in t:
        assigned = None
        for bucket, pats in _BUCKETS.items():
            if _first_match(s, pats):
                assigned = bucket
                break
        buckets.append(assigned or "Other")
    out["reason_bucket"] = buckets

    details = []
    for s in t:
        chosen = None
        for label, pats in _SUBREASONS.items():
            if _first_match(s, pats):
                chosen = label
                break
        details.append(chosen)
    out["reason_detail"] = details
    return out

def _summarize_reasons(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if df.empty:
        return (pd.DataFrame(columns=["reason_bucket","count","pct"]),
                pd.DataFrame(columns=["reason_detail","count","pct"]))
    bucket = (df["reason_bucket"].value_counts(dropna=False)
              .rename_axis("reason_bucket").reset_index(name="count"))
    bucket["pct"] = (bucket["count"] / bucket["count"].sum() * 100).round(1)

    details = (df.dropna(subset=["reason_detail"])
               .groupby("reason_detail", dropna=False)
               .size().reset_index(name="count")
               .sort_values("count", ascending=False))
    if not details.empty:
        details["pct"] = (details["count"] / details["count"].sum() * 100).round(1)
    else:
        details = pd.DataFrame(columns=["reason_detail","count","pct"])

    return bucket, details

# -------------------------
# Main question
# -------------------------
def run(store, params: Dict, user_text: str):
    """
    One-stop 'Complaint analysis — June 2025 (by portfolio)' question:

      1) Table: cases, complaints, per_1000 by Portfolio for June'25
      2) MoM Trend: complaints per 1,000 (fills missing months = 0, pastel line)
      3) Reasons deep-dive for June'25 from 'Brief Description – RCA done by admin'

    Optional filter: params["portfolio"]
    """

    # ---- Read and normalize months
    cases = store.cases.copy()
    cmpl  = store.complaints.copy()

    cases["_month"] = _month_series_from_cases(cases)
    cmpl["_month"]  = _month_series_from_complaints(cmpl)

    # Drop NA months defensively
    cases = cases.dropna(subset=["_month"])
    cmpl  = cmpl.dropna(subset=["_month"])

    # Filter by portfolio if provided
    portfolio = params.get("portfolio")
    if portfolio:
        cases = cases[cases["Portfolio"].str.lower().eq(str(portfolio).lower())]
        cmpl  = cmpl[ cmpl["Portfolio"].str.lower().eq(str(portfolio).lower())]

    # Choose June 2025 (or allow router to pass explicit key)
    month_key = params.get("month_key", "2025-06")

    # ---- 1) June 2025 by Portfolio (table)
    c_june = cases[cases["_month"].eq(month_key)].copy()
    q_june = cmpl[ cmpl["_month"].eq(month_key)].copy()

    # Group by Portfolio
    c_by_p = c_june.groupby("Portfolio").size().rename("cases")
    q_by_p = q_june.groupby("Portfolio").size().rename("complaints")

    by_port = pd.concat([c_by_p, q_by_p], axis=1).fillna(0).astype(int).reset_index()
    if not by_port.empty:
        by_port["per_1000"] = np.where(
            by_port["cases"] > 0,
            (by_port["complaints"] / by_port["cases"] * 1000).round(2),
            None
        )
        by_port = by_port.sort_values(["complaints","cases"], ascending=[False, False], ignore_index=True)
    else:
        by_port = pd.DataFrame(columns=["Portfolio","cases","complaints","per_1000"])

    total_cases = by_port["cases"].sum() if not by_port.empty else 0
    total_comp  = by_port["complaints"].sum() if not by_port.empty else 0
    overall = (total_comp / total_cases * 1000) if total_cases else 0

    title = f"Complaint analysis — Jun 2025 (by portfolio)" if not portfolio else f"Complaint analysis — Jun 2025 (portfolio: {portfolio})"
    st.markdown(f"### {title}")
    st.caption(f"Total: cases={total_cases:,}, complaints={total_comp:,}, per_1000={overall:.2f}")

    if by_port.empty:
        st.info("No rows returned for the current filters.")
    else:
        st.dataframe(
            by_port.rename(columns={"Portfolio": "portfolio"}),
            use_container_width=True
        )

    # ---- 2) Complaints per 1,000 (MoM) trend
    _section("Complaints per 1,000 (MoM)", "Missing months are filled with 0. Line uses soft pastel colors.")
    # Index range: last 13 months across either frame
    all_m = pd.Index(cases["_month"].unique()).union(pd.Index(cmpl["_month"].unique()))
    if len(all_m) > 0:
        last = pd.Period(sorted(all_m)[-1], "M")
        idx  = pd.period_range(last-12, last, freq="M").astype(str)
        c_m  = cases.groupby("_month").size().reindex(idx, fill_value=0)
        q_m  = cmpl.groupby("_month").size().reindex(idx, fill_value=0)
        trend = pd.DataFrame({"month": idx, "cases": c_m.values, "complaints": q_m.values})
        trend["per_1000"] = (trend["complaints"] / trend["cases"].replace(0, np.nan) * 1000).fillna(0).round(2)

        fig, ax = plt.subplots(figsize=(8.5, 3.6))
        pal = _pastel_palette()
        ax.plot(trend["month"], trend["per_1000"], marker="o", linewidth=2.5, color=pal[0])
        ax.grid(True, alpha=0.25)
        ax.set_ylabel("per 1,000")
        ax.set_xlabel("Month")
        # fewer x ticks if lots of months
        step = max(1, len(trend)//12)
        ax.set_xticks(range(0, len(trend), step))
        for s in ["top","right"]:
            ax.spines[s].set_visible(False)
        st.pyplot(fig, use_container_width=True)

        st.dataframe(trend, use_container_width=True)
    else:
        st.info("No month values found to build a trend.")

    # ---- 3) Reasons deep-dive (June 2025)
    _section("June 2025 — Reasons deep-dive")
    text_col = "Brief Description - RCA done by admin"
    if text_col not in cmpl.columns:
        st.warning(f"Cannot produce reasons: column not found in complaints → '{text_col}'")
        return

    q_june_for_reasons = q_june.copy()
    if q_june_for_reasons.empty:
        st.info("No June 2025 complaints found for current filters.")
        return

    labelled = _label_reasons(q_june_for_reasons, text_col)
    bucket, detail = _summarize_reasons(labelled)

    st.markdown("**By bucket**")
    st.dataframe(bucket, use_container_width=True)

    st.markdown("**By detailed reason**")
    st.dataframe(detail, use_container_width=True)
