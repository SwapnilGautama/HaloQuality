# questions/complaints_june_by_portfolio.py
from __future__ import annotations

import re
import pandas as pd
import streamlit as st

ASSUME_YEAR = 2025
TARGET_PERIOD = pd.Period(f"{ASSUME_YEAR}-06", freq="M")


def _norm(s: str) -> str:
    """Normalize a column name: lowercase and strip non-alphanumerics."""
    return re.sub(r"[^a-z0-9]+", "", s.lower() if isinstance(s, str) else str(s).lower())


def _find_col(df: pd.DataFrame, want_keys: list[str]) -> str | None:
    """Find a column by normalized key matches (any of the want_keys is contained)."""
    norm_map = {col: _norm(col) for col in df.columns}
    for col, n in norm_map.items():
        for k in want_keys:
            if k in n:
                return col
    return None


def _coerce_portfolio(df: pd.DataFrame) -> pd.DataFrame:
    if "portfolio" in df.columns:
        col = "portfolio"
    else:
        col = _find_col(df, ["portfolio"])
    if col:
        df = df.rename(columns={col: "portfolio"})
        df["portfolio"] = df["portfolio"].astype(str).str.strip().str.title()
    return df


def _ensure_month_cases(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure _month from any 'create/report/start' date-ish column; dayfirst=True."""
    if "_month" in df.columns:
        return df
    # Try likely columns first
    candidates = [
        "Create Date", "Create Dt", "Report Date", "Start Date",
        "Create Dt.", "Create_Date", "CreateDate", "Created",
        "Date",
    ]
    col = next((c for c in candidates if c in df.columns), None)
    if not col:
        # Fuzzy search for any create/report/start/date-like column
        col = _find_col(df, ["createdate", "createdt", "created", "reportdate", "startdate", "date"])
    if col:
        df["_month"] = pd.to_datetime(df[col], errors="coerce", dayfirst=True).dt.to_period("M")
    else:
        df["_month"] = pd.NaT
    return df


def _ensure_month_complaints(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure _month from 'Date Complaint Received - DD/MM/YY' (fuzzy) or from 'Month' + ASSUME_YEAR."""
    if "_month" in df.columns:
        return df

    # 1) Try a precise complaint date column (fuzzy matching)
    col_date = _find_col(df, ["datecomplaintreceived", "complaintreceiveddate", "datereceived", "complaintdate"])
    if col_date:
        df["_month"] = pd.to_datetime(df[col_date], errors="coerce", dayfirst=True).dt.to_period("M")
        # If that produced nothing, fall back to Month text
        if df["_month"].isna().all():
            col_date = None

    if not col_date:
        # 2) Month text like 'June'
        col_month = _find_col(df, ["month"])
        if col_month:
            # normalize to 'June 2025'
            df["_month"] = pd.to_datetime(
                df[col_month].astype(str).str.strip() + f" {ASSUME_YEAR}", errors="coerce"
            ).dt.to_period("M")
        else:
            df["_month"] = pd.NaT
    return df


def run(store, params=None, user_text=None):
    """
    Complaint analysis for June 2025:
      - Join ONLY by Portfolio + Month (no Process)
      - Cases month: Create/Report/Start date (or prebuilt _month)
      - Complaints month: 'Date Complaint Received - DD/MM/YY' (fuzzy) or Month + ASSUME_YEAR
      - Output: portfolio | cases | complaints | per_1000  (+ overall number)
    """
    params = params or {}
    cases = store.get("cases")
    complaints = store.get("complaints")

    if cases is None or cases.empty:
        st.info("No cases data loaded.")
        return
    if complaints is None or complaints.empty:
        st.info("No complaints data loaded.")
        return

    # Normalize
    cases = _coerce_portfolio(cases.copy())
    complaints = _coerce_portfolio(complaints.copy())
    cases = _ensure_month_cases(cases)
    complaints = _ensure_month_complaints(complaints)

    # Focus on June 2025
    cases_m = cases[cases["_month"] == TARGET_PERIOD].copy()
    comp_m = complaints[complaints["_month"] == TARGET_PERIOD].copy()

    # Optional portfolio filter (e.g., "for London")
    pf = (params or {}).get("portfolio")
    if pf:
        pf = str(pf).strip().title()
        if "portfolio" in cases_m.columns:
            cases_m = cases_m[cases_m["portfolio"] == pf]
        if "portfolio" in comp_m.columns:
            comp_m = comp_m[comp_m["portfolio"] == pf]

    # Aggregate by portfolio only
    have_portfolio = "portfolio" in cases_m.columns and "portfolio" in comp_m.columns
    if not have_portfolio:
        st.warning("Missing 'Portfolio' in cases or complaints after normalization.")
        with st.expander("Parsed filters", expanded=True):
            st.write(
                {
                    "month": str(TARGET_PERIOD),
                    "portfolio": pf or "All",
                    "cases_rows_jun": int(cases_m.shape[0]),
                    "complaints_rows_jun": int(comp_m.shape[0]),
                }
            )
        return

    cases_by = cases_m.groupby("portfolio", dropna=False).size().rename("cases")
    comp_by = comp_m.groupby("portfolio", dropna=False).size().rename("complaints")

    # Outer combine (show whichever side exists)
    out = pd.concat([cases_by, comp_by], axis=1).fillna(0)
    if out.empty:
        st.info("No rows returned for the current filters.")
        with st.expander("Parsed filters", expanded=True):
            st.write(
                {
                    "month": str(TARGET_PERIOD),
                    "portfolio": pf or "All",
                    "cases_rows_jun": int(cases_m.shape[0]),
                    "complaints_rows_jun": int(comp_m.shape[0]),
                }
            )
        return

    out = out.astype({"cases": int, "complaints": int}).reset_index()
    out["per_1000"] = (
        (out["complaints"] / out["cases"].where(out["cases"] != 0, pd.NA)) * 1000
    ).round(1)
    out = out.sort_values("portfolio", na_position="last")

    # Overall
    total = pd.DataFrame(
        {
            "portfolio": ["All"],
            "cases": [int(cases_m.shape[0])],
            "complaints": [int(comp_m.shape[0])],
        }
    )
    total["per_1000"] = (
        (total["complaints"] / total["cases"].where(total["cases"] != 0, pd.NA)) * 1000
    ).round(1)

    st.subheader("Complaints per 1,000 cases — June 2025")
    with st.expander("Parsed filters", expanded=False):
        st.write(
            {
                "month": str(TARGET_PERIOD),
                "portfolio": pf or "All",
                "cases_rows_jun": int(cases_m.shape[0]),
                "complaints_rows_jun": int(comp_m.shape[0]),
            }
        )

    st.dataframe(out[["portfolio", "cases", "complaints", "per_1000"]], use_container_width=True)
    st.dataframe(total[["portfolio", "cases", "complaints", "per_1000"]], use_container_width=True)
