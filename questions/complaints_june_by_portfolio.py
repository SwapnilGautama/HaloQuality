# questions/complaints_june_by_portfolio.py
from __future__ import annotations

import re
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

# Streamlit (if available; safe import so it won't break tests)
try:
    import streamlit as st
except Exception:  # pragma: no cover
    st = None

# Matplotlib (plots render only when Streamlit is available)
try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None


# =========================
#        Helpers
# =========================

def _find_col(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    """Return first existing column (case/space-insensitive) from candidate names."""
    if df is None or df.empty:
        return None
    norm = {c.lower().strip(): c for c in df.columns}
    tight = {k.replace(" ", ""): v for k, v in norm.items()}
    for cand in candidates:
        key = cand.lower().strip()
        if key in norm:
            return norm[key]
        key_tight = key.replace(" ", "")
        if key_tight in tight:
            return tight[key_tight]
    return None


def _month_key_from_datetime(series: pd.Series) -> pd.Series:
    """Convert any date-like series to 'YYYY-MM' (coercing errors)."""
    s = pd.to_datetime(series, dayfirst=True, errors="coerce")
    return s.dt.to_period("M").astype(str)


def _month_key_from_month_name(series: pd.Series, year: int) -> pd.Series:
    """Convert month name (e.g. 'June') to 'YYYY-MM' using the supplied year."""
    s = series.astype(str).str.strip()
    dt = pd.to_datetime("1 " + s + f" {year}", errors="coerce", dayfirst=True)
    return dt.dt.to_period("M").astype(str)


def _parse_month_from_params_or_text(params: Dict[str, Any],
                                     user_text: Optional[str]) -> Tuple[str, int]:
    """
    Choose target month key and assumed year (for 'Month' name fallbacks).
    Priority:
      (1) params['month'] as 'YYYY-MM' or 'Month YYYY'
      (2) user_text with 'Jun 2025' / 'June 2025' / 'June'
      (3) default to '2025-06'
    """
    # (1) params
    if params and isinstance(params.get("month"), str):
        m = params["month"].strip()
        if re.match(r"^\d{4}-\d{2}$", m):
            return m, int(m[:4])
        m2 = re.match(r"^([A-Za-z]{3,})\s+(\d{4})$", m)
        if m2:
            year = int(m2.group(2))
            mk = pd.to_datetime(f"1 {m2.group(1)} {year}", errors="coerce").to_period("M").astype(str)
            return mk, year

    # (2) user text
    if user_text:
        mt = re.search(r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\b(?:\s+(\d{4}))?",
                       user_text, re.I)
        if mt:
            mon = mt.group(1)
            year = int(mt.group(2)) if mt.group(2) else 2025
            mk = pd.to_datetime(f"1 {mon} {year}", errors="coerce").to_period("M").astype(str)
            return mk, year

    # (3) default
    return "2025-06", 2025


def _clean_portfolio(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.replace(r"\s+", " ", regex=True).str.title()


# ----- Light NLP for reasons -----

_reason_order = [
    "Delay",
    "Procedure",
    "Communication",
    "System",
    "Incorrect/Incomplete information",
    "Other",
]

def _normalize_text(x: Any) -> str:
    t = "" if pd.isna(x) else str(x)
    t = t.lower()
    t = re.sub(r"[^a-z0-9\s/+-]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

def _categorize_reason(t: str) -> Tuple[str, Optional[str]]:
    # Coarse category
    if re.search(r"\b(delay|late|overdue|sla|timescale|backlog|manual calc|manual calculation|set ?up|postal delay)\b", t):
        cat = "Delay"
    elif re.search(r"\b(procedure|process|form|paperwork|documentation|document)\b", t):
        cat = "Procedure"
    elif re.search(r"\b(communicat|call|email|letter|reply|response|clarity)\b", t):
        cat = "Communication"
    elif re.search(r"\b(system|portal|it issue|down|bug|glitch)\b", t):
        cat = "System"
    elif re.search(r"\b(incorrect|incomplete|wrong|missing|error)\b", t):
        cat = "Incorrect/Incomplete information"
    else:
        cat = "Other"

    # Sub-reasons for June deep-dive
    sr = None
    if re.search(r"\b(timescale|sla|standard time)\b", t):
        sr = "Aptia standard Timescale"
    elif "scheme rule" in t or re.search(r"\bscheme rules?\b", t):
        sr = "Scheme Rules"
    elif "drop in value" in t or "factor change" in t:
        sr = "Drop in value/ factor change"
    elif "death" in t:
        sr = "Death benefits payout"
    elif "overpayment" in t:
        sr = "Overpayment"
    elif "pension increase" in t or re.search(r"\bpi\b", t):
        sr = "Pension Increase"
    elif "transfer" in t and re.search(r"\b(doc|document|paper)\b", t):
        sr = "Transfer Documentation"
    elif "manual calc" in t or "manual calculation" in t:
        sr = "Manual calculation"
    elif "postal delay" in t:
        sr = "Postal Delay"
    elif "case not created" in t or "case not" in t:
        sr = "Case not created"
    elif "requirement not checked" in t or "not checked" in t:
        sr = "Requirement not checked"
    return cat, sr


# =========================
#          Main
# =========================

def run(store: Dict[str, Any], params: Dict[str, Any], user_text: Optional[str] = None):
    """
    Complaint analysis (single month) with the whole slide in one screen:

      1) Portfolio table (cases, complaints, complaints/1,000)
      2) MoM complaints/1,000 trend (12-month window ending at the target month)
      3) June reasons (sub-reason) table with % (rule-based labels)
      4) Reason Trend (Apr–Jun ’25) grouped bar (coarse categories)

    Inputs expected in `store`:
      - store["cases"]       : DataFrame with Portfolio and a date column (e.g. 'Create Date')
      - store["complaints"]  : DataFrame with Portfolio and either
                               'Date Complaint Received - DD/MM/YY' or 'Month' (assumes year)

    Returns:
      ((title:str, subtitle:str), DataFrame[portfolio, cases, complaints, per_1000])
    """
    cases: pd.DataFrame = store.get("cases", pd.DataFrame()).copy()
    complaints: pd.DataFrame = store.get("complaints", pd.DataFrame()).copy()

    if (cases is None or cases.empty) and (complaints is None or complaints.empty):
        return ("No data loaded.", ""), pd.DataFrame()

    # Target month and assumed year for "Month" fallback in complaints
    target_month_key, assumed_year = _parse_month_from_params_or_text(params, user_text)

    # ---------- CASES ----------
    port_c_cases = _find_col(cases, ["Portfolio", "portfolio"])
    if not port_c_cases:
        return ("Missing 'Portfolio' in cases.", ""), pd.DataFrame()

    date_c_cases = _find_col(cases, ["Create Date", "Create Dt", "CreateDate", "Start Date", "Start Dt", "StartDate"])
    if not date_c_cases:
        return ("Missing a usable date column in cases (e.g., 'Create Date').", ""), pd.DataFrame()

    cases["_month_key"] = _month_key_from_datetime(cases[date_c_cases])
    cases["_portfolio"] = _clean_portfolio(cases[port_c_cases])

    cases_target = cases.loc[cases["_month_key"] == target_month_key].copy()
    cases_by_port = cases_target.groupby("_portfolio", dropna=False).size().reset_index(name="cases")

    # Keep monthly period for trend
    cases["_month"] = pd.PeriodIndex(cases["_month_key"], freq="M")

    # ---------- COMPLAINTS ----------
    port_c_comp = _find_col(complaints, ["Portfolio", "portfolio"])
    if not port_c_comp:
        return ("Missing 'Portfolio' in complaints.", ""), pd.DataFrame()

    comp_date_col = _find_col(complaints, ["Date Complaint Received - DD/MM/YY"])
    if comp_date_col:
        complaints["_month_key"] = _month_key_from_datetime(complaints[comp_date_col])
    else:
        month_name_col = _find_col(complaints, ["Month", "Report Month", "Complaint Month"])
        if not month_name_col:
            return ("Missing date in complaints. Provide 'Date Complaint Received - DD/MM/YY' or 'Month'.", ""), pd.DataFrame()
        complaints["_month_key"] = _month_key_from_month_name(complaints[month_name_col], assumed_year)

    complaints["_portfolio"] = _clean_portfolio(complaints[port_c_comp])
    comp_target = complaints.loc[complaints["_month_key"] == target_month_key].copy()
    comps_by_port = comp_target.groupby("_portfolio", dropna=False).size().reset_index(name="complaints")

    # ---------- Join & table ----------
    out = pd.merge(cases_by_port, comps_by_port, how="outer", on="_portfolio").fillna(0)
    out["cases"] = out["cases"].astype("int64", errors="ignore")
    out["complaints"] = out["complaints"].astype("int64", errors="ignore")
    out["per_1000"] = (out["complaints"] / out["cases"].where(out["cases"] != 0, pd.NA)) * 1000
    out["per_1000"] = out["per_1000"].round(2)
    out = out.rename(columns={"_portfolio": "portfolio"})
    out = out[["portfolio", "cases", "complaints", "per_1000"]].sort_values(
        ["per_1000", "portfolio"], ascending=[False, True], na_position="last"
    ).reset_index(drop=True)

    tot_cases = int(out["cases"].sum()) if not out.empty else 0
    tot_comps = int(out["complaints"].sum()) if not out.empty else 0
    tot_per_1000 = round((tot_comps / tot_cases) * 1000, 2) if tot_cases else 0.0

    # =========================
    #       Streamlit UI
    # =========================
    if st is not None:
        st.subheader(f"Complaint analysis — {pd.Period(target_month_key).strftime('%b %Y')} (by portfolio)")
        st.caption(f"Total: cases={tot_cases:,}, complaints={tot_comps:,}, per_1000={tot_per_1000}")

        # -------- Row 1: portfolio table | MoM line
        c1, c2 = st.columns((1.1, 1))
        with c1:
            st.dataframe(out, use_container_width=True)
        with c2:
            if plt is not None:
                # 12-month window ending at the target month
                end = pd.Period(target_month_key, freq="M")
                start = end - 12
                rng = pd.period_range(start, end, freq="M")

                complaints["_month"] = pd.PeriodIndex(complaints["_month_key"], freq="M")
                m_cases = cases.groupby("_month").size().reindex(rng, fill_value=0)
                m_comps = complaints.groupby("_month").size().reindex(rng, fill_value=0)
                per1000 = (m_comps / m_cases.replace(0, np.nan) * 1000).fillna(0).round(2)

                fig, ax = plt.subplots(figsize=(7.8, 3.8))
                x = rng.astype(str)
                ax.plot(x, per1000.values, marker="o", linewidth=2.0, alpha=0.9)
                ax.set_title("Complaints per 1,000 — MoM Trend")
                ax.set_ylabel("Complaints per 1,000")
                ax.set_xlabel("Month")
                ax.set_xticks(range(len(x)))
                ax.set_xticklabels(x, rotation=35, ha="right")
                ax.grid(alpha=0.15)
                st.pyplot(fig, clear_figure=True)

        # -------- Prep text for reasons labelling
        text_cols = [
            _find_col(complaints, ["Brief Description - RCA done by admin"]),
            _find_col(complaints, ["RCA 1", "RCA1", "Root Cause 1"]),
        ]
        text_cols = [c for c in text_cols if c is not None]

        comp_for_reason = complaints.copy()
        if text_cols:
            comp_for_reason["__text"] = (
                comp_for_reason[text_cols[0]].astype(str).fillna("")
                + " "
                + (comp_for_reason[text_cols[1]].astype(str).fillna("") if len(text_cols) > 1 else "")
            ).apply(_normalize_text)
        else:
            comp_for_reason["__text"] = ""

        cats, subs = [], []
        for t in comp_for_reason["__text"]:
            c, s = _categorize_reason(t)
            cats.append(c)
            subs.append(s)
        comp_for_reason["__cat"] = cats
        comp_for_reason["__sub"] = subs
        comp_for_reason["_month"] = pd.PeriodIndex(comp_for_reason["_month_key"], freq="M")

        # -------- Row 2: June sub-reasons table | Reason trend (Apr–Jun'25) bars
        c3, c4 = st.columns((1.05, 1))
        with c3:
            june_mask = comp_for_reason["_month"] == pd.Period("2025-06", "M")
            june_sub = (
                comp_for_reason.loc[june_mask, "__sub"]
                .value_counts(dropna=True)
                .rename_axis("reason")
                .reset_index(name="count")
            )
            if not june_sub.empty:
                june_sub["percent"] = (june_sub["count"] / june_sub["count"].sum() * 100).round(1)
                st.subheader("June reasons — contribution")
                st.dataframe(june_sub, use_container_width=True)
            else:
                st.subheader("June reasons — contribution")
                st.info("No June sub-reasons could be derived from the available comment fields.")
        with c4:
            if plt is not None:
                months_amj = [pd.Period("2025-04", "M"), pd.Period("2025-05", "M"), pd.Period("2025-06", "M")]
                counts = {}
                for p in months_amj:
                    tmp = comp_for_reason.loc[comp_for_reason["_month"] == p, "__cat"].value_counts()
                    counts[str(p)] = tmp.reindex(_reason_order, fill_value=0)

                labels = _reason_order
                x = np.arange(len(labels))
                width = 0.27
                fig2, ax2 = plt.subplots(figsize=(7.8, 3.8))
                ax2.bar(x - width, counts[str(months_amj[0])].values, width, label="Apr '25")
                ax2.bar(x,         counts[str(months_amj[1])].values, width, label="May '25")
                ax2.bar(x + width, counts[str(months_amj[2])].values, width, label="Jun '25")
                ax2.set_xticks(x, labels)
                for label in ax2.get_xticklabels():
                    label.set_rotation(20)
                    label.set_ha("right")
                ax2.set_title("Reason Trend (Apr–Jun '25)")
                ax2.set_ylabel("Count")
                ax2.grid(axis="y", alpha=0.15)
                ax2.legend()
                st.pyplot(fig2, clear_figure=True)

    # Final return (backwards-compatible)
    title = f"Complaint analysis — {pd.Period(target_month_key).strftime('%b %Y')} (by portfolio)"
    subtitle = f"Total: cases={tot_cases:,}, complaints={tot_comps:,}, per_1000={tot_per_1000}"
    return (title, subtitle), out
