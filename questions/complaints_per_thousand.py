# questions/complaints_per_thousand.py
from __future__ import annotations
import pandas as pd
from typing import Dict, Any, Tuple


def _to_month_from_cases(df: pd.DataFrame) -> pd.Series:
    """
    Produce a month-start Timestamp for cases.
    Priority:
      1) existing '_month_dt' if present
      2) 'Create Date' (day-first) -> month start
    """
    if "_month_dt" in df.columns:
        return pd.to_datetime(df["_month_dt"], errors="coerce").dt.to_period("M").dt.to_timestamp()
    if "Create Date" in df.columns:
        return pd.to_datetime(df["Create Date"], errors="coerce", dayfirst=True).dt.to_period("M").dt.to_timestamp()
    # graceful fallback to any obvious date-like column
    for c in df.columns:
        if "date" in c.lower():
            s = pd.to_datetime(df[c], errors="coerce", dayfirst=True)
            if s.notna().any():
                return s.dt.to_period("M").dt.to_timestamp()
    return pd.to_datetime(pd.Series([pd.NaT] * len(df)))


def _to_month_from_complaints(df: pd.DataFrame, assume_year: int = 2025) -> pd.Series:
    """
    Produce a month-start Timestamp for complaints.
    Priority:
      1) existing '_month_dt' if present
      2) 'Date Complaint Received - DD/MM/YY' (day-first) -> month start
      3) 'Month' (e.g., 'June') + assume_year -> month start
    """
    if "_month_dt" in df.columns:
        return pd.to_datetime(df["_month_dt"], errors="coerce").dt.to_period("M").dt.to_timestamp()

    if "Date Complaint Received - DD/MM/YY" in df.columns:
        s = pd.to_datetime(df["Date Complaint Received - DD/MM/YY"], errors="coerce", dayfirst=True)
        return s.dt.to_period("M").dt.to_timestamp()

    if "Month" in df.columns:
        # handle 'Jun', 'June', case-insensitive; assume provided year
        mon = pd.to_datetime(
            df["Month"].astype(str).str.strip() + f" {assume_year}",
            errors="coerce",
            format="%B %Y"  # full month name first (June)
        )
        # If full month name fails (e.g., 'Jun'), try abbreviated
        bad = mon.isna()
        if bad.any():
            mon2 = pd.to_datetime(
                df.loc[bad, "Month"].astype(str).str.strip() + f" {assume_year}",
                errors="coerce",
                format="%b %Y"  # abbreviated month (Jun)
            )
            mon.loc[bad] = mon2
        return mon.dt.to_period("M").dt.to_timestamp()

    # graceful fallback (unlikely)
    for c in df.columns:
        if "date" in c.lower():
            s = pd.to_datetime(df[c], errors="coerce", dayfirst=True)
            if s.notna().any():
                return s.dt.to_period("M").dt.to_timestamp()
    return pd.to_datetime(pd.Series([pd.NaT] * len(df)))


def _window_from_params(params: Dict[str, Any], fallback_last_n: int = 3) -> Tuple[pd.Timestamp, pd.Timestamp, str]:
    """Compute [start_month, end_month] and a nice title from parsed params."""
    if "start_month" in params and "end_month" in params:
        sm = pd.to_datetime(params["start_month"])  # normalized YYYY-MM-01
        em = pd.to_datetime(params["end_month"])
        title = f"{sm.strftime('%b %Y')} to {em.strftime('%b %Y')}"
        return sm, em, title
    if "last_n_months" in params:
        n = int(params["last_n_months"])
        end = pd.Timestamp.today().to_period("M").to_timestamp()
        start = (end.to_period("M") - n + 1).to_timestamp()
        return start, end, f"last {n} months"

    end = pd.Timestamp.today().to_period("M").to_timestamp()
    start = (end.to_period("M") - fallback_last_n + 1).to_timestamp()
    return start, end, "last 3 months"


def run(store: Dict[str, pd.DataFrame], params: Dict[str, Any] | None = None, user_text: str = ""):
    """
    Complaints per 1,000 cases, joined ONLY on:
      - portfolio (case-insensitive)
      - month (cases: Create Date; complaints: Month (assume 2025) or Date Complaint Received)
    No process dimension.
    """
    params = params or {}
    cases = store.get("cases", pd.DataFrame()).copy()
    cmpl  = store.get("complaints", pd.DataFrame()).copy()

    notes: list[str] = []

    # Portfolio filter (case-insensitive). Default: 'all'
    portfolio = (params.get("portfolio") or "all").strip().lower()

    # Build month columns
    cases["_m"] = _to_month_from_cases(cases)
    cmpl["_m"]  = _to_month_from_complaints(cmpl, assume_year=2025)

    # Basic column checks
    need_cases = {"_m", "Portfolio"}
    need_cmpl  = {"_m", "Portfolio"}
    miss_cases = sorted(list(need_cases - set(cases.columns)))
    miss_cmpl  = sorted(list(need_cmpl  - set(cmpl.columns)))
    if miss_cases or miss_cmpl:
        if miss_cases:
            notes.append(f"Missing columns in cases: {miss_cases}")
        if miss_cmpl:
            notes.append(f"Missing columns in complaints: {miss_cmpl}")
        return ("Complaints per 1,000 cases", pd.DataFrame(), notes)

    # Normalize portfolio
    cases["portfolio_norm"] = cases["Portfolio"].astype(str).str.strip().str.lower()
    cmpl["portfolio_norm"]  = cmpl["Portfolio"].astype(str).str.strip().str.lower()

    # Filter by portfolio if provided
    if portfolio != "all":
        cases = cases[cases["portfolio_norm"] == portfolio]
        cmpl  = cmpl[cmpl["portfolio_norm"] == portfolio]

    # Keep valid months only
    cases = cases[cases["_m"].notna()]
    cmpl  = cmpl[cmpl["_m"].notna()]

    # Date window
    start_m, end_m, range_title = _window_from_params(params)
    cases = cases[(cases["_m"] >= start_m) & (cases["_m"] <= end_m)]
    cmpl  = cmpl[(cmpl["_m"] >= start_m) & (cmpl["_m"] <= end_m)]

    # Diagnostics
    notes.append(
        f"Filtered rows → cases: {len(cases)} | complaints: {len(cmpl)} "
        f"| months: {start_m.strftime('%b %Y')} → {end_m.strftime('%b %Y')} "
        f"| portfolio: {('All' if portfolio=='all' else portfolio.title())}"
    )

    # Aggregate by (portfolio, month)
    den = (cases.groupby(["portfolio_norm", "_m"], dropna=False)
                .size()
                .rename("cases")
                .reset_index())
    num = (cmpl.groupby(["portfolio_norm", "_m"], dropna=False)
               .size()
               .rename("complaints")
               .reset_index())

    out = pd.merge(num, den, on=["portfolio_norm", "_m"], how="outer")
    out["cases"] = out["cases"].fillna(0).astype("Int64")
    out["complaints"] = out["complaints"].fillna(0).astype("Int64")
    out["per_1000"] = (out["complaints"] / out["cases"].replace({0: pd.NA})) * 1000

    # If nothing overlaps, be explicit
    if out[["complaints", "cases"]].sum(numeric_only=True).sum() == 0:
        notes.append("No overlapping data for (portfolio, month) after filtering.")
        title = f"Complaints per 1,000 cases — {('All' if portfolio=='all' else portfolio.title())} — {range_title}"
        return title, pd.DataFrame(), notes

    # Prettify
    out = out.rename(columns={"_m": "month", "portfolio_norm": "portfolio"})
    out = out.sort_values(["month", "portfolio"])
    out["month"] = out["month"].dt.strftime("%b %Y")

    title = f"Complaints per 1,000 cases — {('All' if portfolio=='all' else portfolio.title())} — {range_title}"
    return title, out[["month", "portfolio", "cases", "complaints", "per_1000"]], notes
