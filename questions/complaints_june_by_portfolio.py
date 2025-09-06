# questions/complaints_june_by_portfolio.py
from __future__ import annotations

import re
from typing import Dict, Optional, Tuple
import pandas as pd
import numpy as np
import streamlit as st

# -------------------------
# small helpers
# -------------------------
def _section(title: str, caption: Optional[str] = None) -> None:
    st.subheader(title)
    if caption:
        st.caption(caption)

MON_MAP = {
    "jan": "01", "feb": "02", "mar": "03", "apr": "04",
    "may": "05", "jun": "06", "jul": "07", "aug": "08",
    "sep": "09", "oct": "10", "nov": "11", "dec": "12",
}

def _month_from_cases(df: pd.DataFrame) -> pd.Series:
    # prefer normalized month if present
    if "_month_dt" in df.columns:
        return pd.to_datetime(df["_month_dt"], errors="coerce").dt.to_period("M").astype(str)
    # fall back to Create Date
    if "Create Date" in df.columns:
        return pd.to_datetime(df["Create Date"], errors="coerce").dt.to_period("M").astype(str)
    return pd.Series(pd.NA, index=df.index, dtype="object")

def _month_from_complaints(df: pd.DataFrame) -> pd.Series:
    if "_month_dt" in df.columns:
        return pd.to_datetime(df["_month_dt"], errors="coerce").dt.to_period("M").astype(str)

    if "Date Complaint Received - DD/MM/YY" in df.columns:
        s = pd.to_datetime(df["Date Complaint Received - DD/MM/YY"], dayfirst=True, errors="coerce")
        return s.dt.to_period("M").astype(str)

    # Month only -> assume 2025
    if "Month" in df.columns:
        def to_key(x):
            if pd.isna(x):
                return pd.NA
            m3 = str(x).strip()[:3].lower()
            mm = MON_MAP.get(m3)
            return f"2025-{mm}" if mm else pd.NA
        return df["Month"].map(to_key)

    return pd.Series(pd.NA, index=df.index, dtype="object")

# -------------------------
# reason labelling (simple rules)
# -------------------------
BUCKETS = {
    "Delay": [
        r"\bdelay\b", r"\bmanual calc(ulation)?\b", r"\bpostal\b",
        r"\b2(nd)? review\b", r"\btimescale\b", r"\bslow\b", r"\blate\b"
    ],
    "Procedure": [r"\bscheme rules?\b", r"\bstandard timescale\b", r"\bSLA\b", r"\bprocedure\b"],
    "Communication": [r"\bletter\b", r"\bcommunication\b", r"\bnot (informed|told|clear)\b", r"\bno (reply|response)\b"],
    "System": [r"\bsystem\b", r"\bworkflow\b", r"\bplatform\b", r"\bbug\b", r"\berror\b"],
    "Incorrect/Incomplete information": [r"\bincorrect\b", r"\bwrong\b", r"\bincomplete\b", r"\bmissing\b", r"\bno evidence\b"]
}

DETAILS = {
    "Delay Manual calculation": [r"\bmanual calc(ulation)?\b"],
    "Aptia standard Timescale": [r"\bstandard timescale\b", r"\btimescale\b", r"\bSLA\b"],
    "Delay Pension set up": [r"\bpension set up\b", r"\bsetup\b"],
    "Delay Postal Delay": [r"\bpostal\b", r"\bpost\b", r"\bmail\b"],
    "Delay - AVC": [r"\bAVC\b"],
    "Delay Requirement not checked": [r"\brequirement not checked\b", r"\bnot checked\b"],
    "Delay Case not created": [r"\bcase not created\b"],
    "Delay 2nd Review": [r"\b2(nd)? review\b", r"\bsecond review\b"],
    "Delay - Trustee": [r"\btrustee\b"],
    "Scheme Rules": [r"\bscheme rules?\b"],
    "Drop in value/ factor change": [r"\bfactor change\b", r"\bdrop in value\b"],
    "Death benefits payout": [r"\bdeath benefit(s)?\b"],
    "Overpayment": [r"\boverpayment\b"],
    "Pension Increase": [r"\bpension increase\b"],
    "Transfer Documentation": [r"\btransfer doc(umentation)?\b"],
}

def _has_any(text: str, pats: list[str]) -> bool:
    s = "" if pd.isna(text) else str(text)
    for p in pats:
        if re.search(p, s, flags=re.I):
            return True
    return False

def _label_reasons(df: pd.DataFrame, text_col: str) -> pd.DataFrame:
    out = df.copy()
    t = out[text_col].fillna("").astype(str)

    buckets = []
    for s in t:
        lab = "Other"
        for bucket, pats in BUCKETS.items():
            if _has_any(s, pats):
                lab = bucket
                break
        buckets.append(lab)
    out["reason_bucket"] = buckets

    details = []
    for s in t:
        lab = None
        for det, pats in DETAILS.items():
            if _has_any(s, pats):
                lab = det
                break
        details.append(lab)
    out["reason_detail"] = details
    return out

def _summarize_reasons(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if df.empty:
        return (pd.DataFrame(columns=["reason_bucket", "count", "pct"]),
                pd.DataFrame(columns=["reason_detail", "count", "pct"]))
    bucket = df["reason_bucket"].value_counts(dropna=False).rename_axis("reason_bucket").reset_index(name="count")
    bucket["pct"] = (bucket["count"] / bucket["count"].sum() * 100).round(1)

    detail = df.dropna(subset=["reason_detail"]).groupby("reason_detail", dropna=False).size().reset_index(name="count")
    detail = detail.sort_values("count", ascending=False)
    if not detail.empty:
        detail["pct"] = (detail["count"] / detail["count"].sum() * 100).round(1)
    return bucket, detail

# -------------------------
# main entry point
# -------------------------
def run(store, params: Dict, user_text: str):
    """
    Complaint analysis (single question):
      - June 2025 portfolio table (cases, complaints, per_1000)
      - MoM complaints-per-1000 line (past 13 months, missing months filled with 0)
      - June 2025 reasons deep-dive from 'Brief Description - RCA done by admin'
    """

    # 1) get data + month keys
    cases = store.cases.copy()
    cmpl  = store.complaints.copy()

    cases["_month"] = _month_from_cases(cases)
    cmpl["_month"]  = _month_from_complaints(cmpl)

    cases = cases.dropna(subset=["_month"])
    cmpl  = cmpl.dropna(subset=["_month"])

    month_key = params.get("month_key", "2025-06")
    portfolio = params.get("portfolio")

    if portfolio:
        mask_cases = cases["Portfolio"].str.lower().eq(str(portfolio).lower())
        mask_cmpl  = cmpl["Portfolio"].str.lower().eq(str(portfolio).lower())
        cases = cases[mask_cases]
        cmpl  = cmpl[mask_cmpl]

    # 2) June table by portfolio
    c_june = cases[cases["_month"].eq(month_key)]
    q_june = cmpl[ cmpl["_month"].eq(month_key)]

    c_by_p = c_june.groupby("Portfolio").size().rename("cases")
    q_by_p = q_june.groupby("Portfolio").size().rename("complaints")
    by_port = pd.concat([c_by_p, q_by_p], axis=1).fillna(0).astype(int).reset_index()

    if not by_port.empty:
        by_port["per_1000"] = np.where(
            by_port["cases"] > 0,
            (by_port["complaints"] / by_port["cases"] * 1000).round(2),
            None
        )
        by_port = by_port.sort_values(["complaints", "cases"], ascending=[False, False], ignore_index=True)

    total_cases = int(by_port["cases"].sum()) if not by_port.empty else 0
    total_comp  = int(by_port["complaints"].sum()) if not by_port.empty else 0
    overall = (total_comp / total_cases * 1000) if total_cases else 0.0

    title = "Complaint analysis — Jun 2025 (by portfolio)" if not portfolio else f"Complaint analysis — Jun 2025 (portfolio: {portfolio})"
    st.markdown(f"### {title}")
    st.caption(f"Total: cases={total_cases:,}, complaints={total_comp:,}, per_1000={overall:.2f}")

    if by_port.empty:
        st.info("No rows returned for the current filters.")
    else:
        st.dataframe(by_port.rename(columns={"Portfolio": "portfolio"}), use_container_width=True)

    # 3) MoM per_1000 trend (past 13 months)
    _section("Complaints per 1,000 (MoM)", "Missing months are filled with 0.")
    all_months = pd.Index(cases["_month"].unique()).union(pd.Index(cmpl["_month"].unique()))
    if len(all_months) > 0:
        last = pd.Period(sorted(all_months)[-1], "M")
        idx = pd.period_range(last - 12, last, freq="M").astype(str)

        c_m = cases.groupby("_month").size().reindex(idx, fill_value=0)
        q_m = cmpl.groupby("_month").size().reindex(idx, fill_value=0)
        trend = pd.DataFrame({"month": idx, "cases": c_m.values, "complaints": q_m.values})
        trend["per_1000"] = (trend["complaints"] / trend["cases"].replace(0, np.nan) * 1000).fillna(0).round(2)

        # Try Altair (soft colors, smooth line); fallback to st.line_chart if Altair not available
        try:
            import altair as alt
            chart = (
                alt.Chart(trend)
                .mark_line(interpolate="monotone", point=True, strokeWidth=3)
                .encode(
                    x=alt.X("month:N", title="Month"),
                    y=alt.Y("per_1000:Q", title="per 1,000"),
                    color=alt.value("#7BAFD4")  # soft pastel-ish blue
                )
            )
            st.altair_chart(chart, use_container_width=True)
        except Exception:
            st.line_chart(trend.set_index("month")["per_1000"])

        st.dataframe(trend, use_container_width=True)
    else:
        st.info("No month values found to build a trend.")

    # 4) Reasons deep-dive for June
    _section("June 2025 — Reasons deep-dive")
    text_col = "Brief Description - RCA done by admin"
    if text_col not in cmpl.columns:
        st.warning(f"Cannot produce reasons: complaints column not found: '{text_col}'")
        return

    if q_june.empty:
        st.info("No June 2025 complaints found for current filters.")
        return

    labelled = _label_reasons(q_june, text_col)
    bucket, detail = _summarize_reasons(labelled)

    st.markdown("**By bucket**")
    st.dataframe(bucket, use_container_width=True)

    st.markdown("**By detailed reason**")
    st.dataframe(detail, use_container_width=True)
