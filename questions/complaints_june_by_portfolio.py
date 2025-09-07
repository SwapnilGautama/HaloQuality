# questions/complaints_june_by_portfolio.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import re
from typing import Dict, Optional, Tuple, List

import pandas as pd
import numpy as np
import streamlit as st
import altair as alt


# -----------------------------
# Small helpers
# -----------------------------

def _section(title: str, caption: Optional[str] = None) -> None:
    st.subheader(title)
    if caption:
        st.caption(caption)


def _assumed_year(store) -> int:
    """
    Read the assumed year for complaints if 'Month' (e.g. 'June') is used.
    The app typically sets this in load_store(..., assume_year_for_complaints=2025).
    """
    default_year = 2025
    try:
        return int(getattr(store, "meta", {}).get("assume_year_for_complaints", default_year))
    except Exception:
        return default_year


def _clean_portfolio_series(s: pd.Series) -> pd.Series:
    if s is None:
        return s
    s = s.astype(str).str.strip()
    # collapse multiple spaces, title-case typical names
    s = s.str.replace(r"\s+", " ", regex=True)
    return s


def _find_cases_date_column(df_cases: pd.DataFrame) -> str:
    """
    We standardise on 'Create Date'. If it doesn't exist, try a few common fallbacks.
    """
    candidates = [
        "Create Date",
        "Start Date",
        "Created Date",
        "Report Date",
        "Created",
        "Date",
    ]
    for c in candidates:
        if c in df_cases.columns:
            return c
    # last fallback: any datetime-like column
    for c in df_cases.columns:
        if pd.api.types.is_datetime64_any_dtype(df_cases[c]):
            return c
    # if nothing, return empty and let the caller error
    return ""


def _month_period(dt_like: pd.Series) -> pd.Series:
    """
    Convert a datetime-like series to 'YYYY-MM' string month keys.
    """
    return pd.to_datetime(dt_like, errors="coerce").dt.to_period("M").astype(str)


def _month_from_cases(df_cases: pd.DataFrame) -> pd.Series:
    col = _find_cases_date_column(df_cases)
    if not col:
        raise ValueError("Could not find a date column in cases (expected 'Create Date').")
    return _month_period(df_cases[col])


def _month_from_complaints(df_comp: pd.DataFrame, assume_year: int) -> pd.Series:
    """
    Complaints: accept either
      - 'Date Complaint Received - DD/MM/YY' (parse), or
      - 'Month' with names like 'June' (assume the given year), or
      - any other date-like column that's present.
    """
    if "Date Complaint Received - DD/MM/YY" in df_comp.columns:
        return _month_period(df_comp["Date Complaint Received - DD/MM/YY"])
    if "Month" in df_comp.columns:
        # build YYYY-MM from Month-name + assumed year
        month_to_num = {
            "jan": "01", "feb": "02", "mar": "03", "apr": "04", "may": "05", "jun": "06",
            "jul": "07", "aug": "08", "sep": "09", "oct": "10", "nov": "11", "dec": "12"
        }
        m = (
            df_comp["Month"]
            .astype(str)
            .str.strip()
            .str[:3]
            .str.lower()
            .map(month_to_num)
        )
        return pd.Series([f"{assume_year}-{mm}" if pd.notna(mm) else np.nan for mm in m], index=df_comp.index)
    # try any datetime-like column
    for c in df_comp.columns:
        if pd.api.types.is_datetime64_any_dtype(df_comp[c]):
            return _month_period(df_comp[c])
    raise ValueError(
        "Could not find a usable date in complaints. "
        "Expected 'Date Complaint Received - DD/MM/YY' or 'Month'."
    )


def _tag_reason(text: str) -> str:
    """
    Simple keyword-based mapper from free-text 'Brief Description - RCA done by admin'
    to reason categories. Extend as needed.
    """
    if not text or not isinstance(text, str):
        return "Other / Unclear"
    t = text.lower()

    rules: List[Tuple[str, str]] = [
        (r"scheme\s*rule", "Scheme Rules"),
        (r"timescale|sla|standard\s*timescale", "Aptia standard Timescale"),
        (r"death\s*benefit", "Death benefits payout"),
        (r"overpay", "Overpayment"),
        (r"pension\s*increase|pi\b", "Pension Increase"),
        (r"manual\s*calc|manual\s*calculation", "Manual calculation"),
        (r"drop\s*in\s*value|factor\s*change", "Drop in value/ factor change"),
        (r"case\s*not\s*created", "Case not created"),
        (r"2nd\s*review|second\s*review", "2nd Review"),
        (r"requirement\s*not\s*checked|doc\s*not\s*checked", "Requirement not checked"),
        (r"incorrect|incomplete|wrong\s*info", "Incorrect/Incomplete information"),
        (r"procedure|process", "Procedure"),
        (r"communication|letter", "Communication"),
        (r"system", "System"),
        (r"delay", "Delay"),
    ]
    for pat, label in rules:
        if re.search(pat, t):
            return label
    return "Other / Unclear"


def _reason_df_for_june(df_comp_june: pd.DataFrame) -> pd.DataFrame:
    col = None
    for c in df_comp_june.columns:
        if "brief" in c.lower() and "admin" in c.lower():
            col = c
            break
    if col is None:
        # try a few common columns that might hold a description / comment
        for c in ["Comments", "Comment", "Description", "Details"]:
            if c in df_comp_june.columns:
                col = c
                break
    if col is None:
        # If we can't find any text column, bail out with empty
        return pd.DataFrame(columns=["reason", "count", "percent"])

    reasons = df_comp_june[col].fillna("").map(_tag_reason)
    out = (
        reasons.value_counts(dropna=False)
        .rename_axis("reason")
        .reset_index(name="count")
        .sort_values(["count", "reason"], ascending=[False, True])
    )
    total = out["count"].sum()
    out["percent"] = (out["count"] * 100 / total).round(1) if total > 0 else 0
    return out


def _monthly_overall_per_1000(df_cases: pd.DataFrame, df_comp: pd.DataFrame) -> pd.DataFrame:
    """
    For each month, compute complaints per 1,000 across all portfolios:
        per_1000 = (sum complaints) / (sum cases) * 1000
    Fill any missing months in the observed range with 0.
    """
    cases_m = (
        df_cases.groupby("_month", dropna=False)[["Case ID"]]
        .count()
        .rename(columns={"Case ID": "cases"})
        .reset_index()
    )
    comp_m = (
        df_comp.groupby("_month", dropna=False)[["__comp_row__"]]
        .count()
        .rename(columns={"__comp_row__": "complaints"})
        .reset_index()
    )
    m = pd.merge(cases_m, comp_m, on="_month", how="outer").fillna(0.0)
    m["cases"] = m["cases"].astype(int)
    m["complaints"] = m["complaints"].astype(int)
    m = m.sort_values("_month")

    # fill missing months in the observed span
    if not m.empty:
        idx = pd.period_range(m["_month"].min(), m["_month"].max(), freq="M").astype(str)
        m = m.set_index("_month").reindex(idx, fill_value=0).rename_axis("_month").reset_index()

    m["per_1000"] = (m["complaints"] * 1000 / m["cases"]).replace([np.inf, -np.inf], 0).fillna(0).round(2)
    return m


# -----------------------------
# Core run
# -----------------------------

def run(store, params: Dict, user_text: str) -> None:
    """
    Complaint analysis — June 2025 (by portfolio)
    Visual layout:
      Row 1:  portfolio table  |  monthly per-1000 line chart
      Row 2:  reasons table    |  reasons bar chart
    """
    assume_year = _assumed_year(store)

    # --- Get dataframes from store (dict-like)
    try:
        df_cases = store["cases"].copy()
        df_comp = store["complaints"].copy()
    except Exception:
        st.error("Could not access 'cases' and 'complaints' in the data store.")
        return

    # --- Basic presence checks
    missing_cases_cols = [c for c in ["Case ID", "Portfolio"] if c not in df_cases.columns]
    if missing_cases_cols:
        st.warning(f"Missing columns in cases: {missing_cases_cols}")
        return
    if "Portfolio" not in df_comp.columns:
        st.warning("Missing column 'Portfolio' in complaints.")
        return

    # --- Normalise months and portfolio text
    try:
        df_cases["_month"] = _month_from_cases(df_cases)
    except Exception as e:
        st.error(f"Failed to parse month from cases. {e}")
        return

    try:
        df_comp["_month"] = _month_from_complaints(df_comp, assume_year)
    except Exception as e:
        st.error(f"Failed to parse month from complaints. {e}")
        return

    df_cases["Portfolio"] = _clean_portfolio_series(df_cases["Portfolio"])
    df_comp["Portfolio"] = _clean_portfolio_series(df_comp["Portfolio"])

    # helper column to count complaint rows
    df_comp["__comp_row__"] = 1

    # --- Target month: June 2025
    target_month = "2025-06"
    df_cases_june = df_cases.loc[df_cases["_month"] == target_month].copy()
    df_comp_june = df_comp.loc[df_comp["_month"] == target_month].copy()

    # --- Portfolio table
    cases_by_port = (
        df_cases_june.groupby("Portfolio", dropna=False)[["Case ID"]]
        .count()
        .rename(columns={"Case ID": "cases"})
        .reset_index()
    )
    comp_by_port = (
        df_comp_june.groupby("Portfolio", dropna=False)[["__comp_row__"]]
        .count()
        .rename(columns={"__comp_row__": "complaints"})
        .reset_index()
    )

    port_tbl = (
        pd.merge(cases_by_port, comp_by_port, on="Portfolio", how="outer")
        .fillna(0.0)
        .replace([np.inf, -np.inf], 0)
    )
    port_tbl["cases"] = port_tbl["cases"].astype(int)
    port_tbl["complaints"] = port_tbl["complaints"].astype(int)
    port_tbl["per_1000"] = (port_tbl["complaints"] * 1000 / port_tbl["cases"]).replace([np.inf, -np.inf], 0).fillna(0)
    port_tbl["per_1000"] = port_tbl["per_1000"].round(2)
    port_tbl = port_tbl.sort_values(["per_1000", "Portfolio"], ascending=[False, True]).reset_index(drop=True)
    port_tbl = port_tbl.rename(columns={"Portfolio": "portfolio"})

    # --- Monthly per-1000 overall (last ~13 months)
    monthly = _monthly_overall_per_1000(
        df_cases[["_month", "Case ID"]],
        df_comp[["_month", "__comp_row__"]],
    )

    # Prepare a readable month label (e.g., 'Jun 2025')
    if not monthly.empty:
        monthly["_label"] = pd.PeriodIndex(monthly["_month"], freq="M").strftime("%b %Y")
        # keep last 13 months to mirror slide feel (Apr'24–Jun'25 style)
        monthly = monthly.tail(13)

    # --- June reasons from comments
    reasons_tbl = _reason_df_for_june(df_comp_june)

    # -----------------------------
    # Rendering
    # -----------------------------
    total_cases = int(port_tbl["cases"].sum())
    total_complaints = int(port_tbl["complaints"].sum())
    overall_per_1000 = (total_complaints * 1000 / total_cases) if total_cases else 0
    overall_per_1000 = round(overall_per_1000, 2)

    st.markdown("### Complaint analysis — Jun 2025 (by portfolio)")
    st.caption(f"Total: cases={total_cases:,}, complaints={total_complaints:,}, per_1000={overall_per_1000}")

    # ---- Row 1: portfolio table | monthly line
    c1, c2 = st.columns([1.05, 1.00])

    with c1:
        st.dataframe(
            port_tbl,
            use_container_width=True,
            hide_index=True,
        )

    with c2:
        if monthly.empty:
            st.info("No monthly data available to plot.")
        else:
            # Pastel line, smooth, points
            line_chart = (
                alt.Chart(monthly)
                .mark_line(point=True)
                .encode(
                    x=alt.X("_label:N", title="Month"),
                    y=alt.Y("per_1000:Q", title="Complaints per 1,000"),
                    tooltip=[
                        alt.Tooltip("_label:N", title="Month"),
                        alt.Tooltip("per_1000:Q", title="Per 1,000")
                    ],
                )
                .properties(height=300)
                .configure_mark(strokeWidth=3, color="#5DA5DA")  # soft blue
            )
            # gentle theme adjustments
            line_chart = (
                line_chart
                .configure_axis(labelColor="#666", titleColor="#666")
                .configure_view(stroke="#eee")
                .configure_point(size=60, filled=True, color="#AEC7E8")  # pastel marker
            )
            st.altair_chart(line_chart, use_container_width=True)

    # ---- Row 2: reasons table | bar chart
    st.markdown("### June reasons — contribution")
    c3, c4 = st.columns([1.00, 1.00])

    with c3:
        st.dataframe(
            reasons_tbl,
            use_container_width=True,
            hide_index=True,
        )

    with c4:
        if reasons_tbl.empty:
            st.info("No reasons found in June complaints.")
        else:
            top_reasons = reasons_tbl.head(10).copy()
            bar = (
                alt.Chart(top_reasons)
                .mark_bar()
                .encode(
                    x=alt.X("percent:Q", title="Percent"),
                    y=alt.Y("reason:N", sort="-x", title="Reason"),
                    tooltip=[
                        alt.Tooltip("reason:N", title="Reason"),
                        alt.Tooltip("count:Q", title="Count"),
                        alt.Tooltip("percent:Q", title="%")
                    ],
                    color=alt.value("#FFBB78"),  # pastel orange
                )
                .properties(height=320)
            )
            bar = (
                bar
                .configure_axis(labelColor="#666", titleColor="#666")
                .configure_view(stroke="#eee")
            )
            st.altair_chart(bar, use_container_width=True)

    # Footer notes for traceability
    with st.expander("Parsed filters", expanded=False):
        st.json({
            "month": target_month,
            "assume_year_for_complaints": assume_year,
            "cases_rows_jun": int(len(df_cases_june)),
            "complaints_rows_jun": int(len(df_comp_june)),
        })
