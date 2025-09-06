# questions/complaints_june_by_portfolio.py
from __future__ import annotations
import re
from typing import Any, Dict, Optional, Tuple

import pandas as pd


# ------------ helpers ------------

def _find_col(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    """Return the first existing column (case/space-insensitive) from candidates."""
    norm = {c.lower().strip(): c for c in df.columns}
    for cand in candidates:
        key = cand.lower().strip()
        if key in norm:
            return norm[key]
        # try relaxed: remove spaces
        ks = {k.replace(" ", ""): v for k, v in norm.items()}
        if key.replace(" ", "") in ks:
            return ks[key.replace(" ", "")]
    return None


def _month_key_from_datetime(series: pd.Series) -> pd.Series:
    """Convert datetimes to YYYY-MM strings."""
    s = pd.to_datetime(series, errors="coerce", dayfirst=True)
    return s.dt.to_period("M").astype(str)


def _month_key_from_month_name(series: pd.Series, year: int) -> pd.Series:
    """
    Convert month name (e.g., 'June') to YYYY-MM using a fixed year.
    """
    s = series.astype(str).str.strip()
    # Coerce month name to a first-of-month date with provided year
    # Using '1 {name} {year}' is robust for most month names.
    dt = pd.to_datetime("1 " + s + f" {year}", errors="coerce", dayfirst=True)
    return dt.dt.to_period("M").astype(str)


def _parse_month_from_params_or_text(params: Dict[str, Any], user_text: Optional[str]) -> Tuple[str, int]:
    """
    Decide the target month key and year.

    Priority:
      1) params['month'] in 'YYYY-MM' or 'Mon YYYY'
      2) user_text: 'June 2025' or 'June'
      3) default '2025-06'
    """
    # 1) explicit param
    if params and isinstance(params.get("month"), str):
        m = params["month"].strip()
        # Accept 'YYYY-MM'
        m1 = re.match(r"^\d{4}-\d{2}$", m)
        if m1:
            year = int(m[:4])
            return m, year
        # Accept 'Mon YYYY' or 'Month YYYY'
        m2 = re.match(r"^([A-Za-z]{3,})\s+(\d{4})$", m)
        if m2:
            year = int(m2.group(2))
            month_key = pd.to_datetime(f"1 {m2.group(1)} {year}", errors="coerce").to_period("M").astype(str)
            return month_key, year

    # 2) try user text
    if user_text:
        mt = re.search(r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\b(?:\s+(\d{4}))?", user_text, re.I)
        if mt:
            mon = mt.group(1)
            year = int(mt.group(2)) if mt.group(2) else 2025
            month_key = pd.to_datetime(f"1 {mon} {year}", errors="coerce").to_period("M").astype(str)
            return month_key, year

    # 3) default: June 2025
    return "2025-06", 2025


def _clean_portfolio(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.replace(r"\s+", " ", regex=True).str.title()


# ------------ main entry ------------

def run(store: Dict[str, Any], params: Dict[str, Any], user_text: Optional[str] = None):
    """
    Complaint analysis for a single month, by portfolio.
    - Join keys: (month_key, Portfolio)
    - Cases month_key: derived from 'Create Date' (or fallback date columns)
    - Complaints month_key:
        1) from 'Date Complaint Received - DD/MM/YY' if present
        2) else from 'Month' + assumed year (defaults to 2025 or taken from params/user_text)
    """
    cases: pd.DataFrame = store.get("cases", pd.DataFrame()).copy()
    complaints: pd.DataFrame = store.get("complaints", pd.DataFrame()).copy()

    if cases.empty and complaints.empty:
        return "No data loaded.", pd.DataFrame()

    # Decide the target month/year first
    target_month_key, assumed_year = _parse_month_from_params_or_text(params, user_text)

    # ---------- Prep CASES ----------
    # Portfolio column
    port_col_cases = _find_col(cases, ["Portfolio", "portfolio"])
    if not port_col_cases:
        return "Missing 'Portfolio' in cases.", pd.DataFrame()

    # Date candidate columns (prioritize 'Create Date')
    date_col_cases = _find_col(
        cases,
        ["Create Date", "Create Dt", "CreateDate", "Start Date", "Start Dt", "StartDate"]
    )
    if not date_col_cases:
        return "Missing a usable date column in cases (e.g., 'Create Date').", pd.DataFrame()

    cases["_month_key"] = _month_key_from_datetime(cases[date_col_cases])
    cases["_portfolio"] = _clean_portfolio(cases[port_col_cases])

    cases_jun = cases.loc[cases["_month_key"] == target_month_key].copy()
    cases_by_port = (
        cases_jun.groupby("_portfolio", dropna=False)
        .size()
        .reset_index(name="cases")
    )

    # ---------- Prep COMPLAINTS ----------
    port_col_comp = _find_col(complaints, ["Portfolio", "portfolio"])
    if not port_col_comp:
        return "Missing 'Portfolio' in complaints.", pd.DataFrame()

    # Preferred date column first
    comp_date_col = _find_col(complaints, ["Date Complaint Received - DD/MM/YY"])
    if comp_date_col:
        complaints["_month_key"] = _month_key_from_datetime(complaints[comp_date_col])
    else:
        # Fall back to a Month name + assumed_year
        month_name_col = _find_col(complaints, ["Month", "Report Month", "Complaint Month"])
        if not month_name_col:
            return (
                "Missing date in complaints. Provide 'Date Complaint Received - DD/MM/YY' "
                "or 'Month' column.",
                pd.DataFrame()
            )
        complaints["_month_key"] = _month_key_from_month_name(complaints[month_name_col], assumed_year)

    complaints["_portfolio"] = _clean_portfolio(complaints[port_col_comp])
    comp_jun = complaints.loc[complaints["_month_key"] == target_month_key].copy()

    comps_by_port = (
        comp_jun.groupby("_portfolio", dropna=False)
        .size()
        .reset_index(name="complaints")
    )

    # ---------- Join on (month_key, portfolio) → in practice month_key already filtered ----------
    out = pd.merge(
        cases_by_port,
        comps_by_port,
        how="outer",
        left_on="_portfolio",
        right_on="_portfolio",
    ).fillna(0)

    # per 1,000 (guard against div by zero)
    out["cases"] = out["cases"].astype("int64", errors="ignore")
    out["complaints"] = out["complaints"].astype("int64", errors="ignore")
    out["per_1000"] = (out["complaints"] / out["cases"].where(out["cases"] != 0, pd.NA)) * 1000
    out["per_1000"] = out["per_1000"].round(2)

    # Totals
    tot_cases = out["cases"].sum()
    tot_comps = out["complaints"].sum()
    tot_per_1000 = round((tot_comps / tot_cases) * 1000, 2) if tot_cases else 0.0

    # Final presentation
    out = out.rename(columns={"_portfolio": "portfolio"})
    out = out[["portfolio", "cases", "complaints", "per_1000"]].sort_values(
        ["per_1000", "portfolio"], ascending=[False, True], na_position="last"
    )

    title = f"Complaint analysis — {pd.Period(target_month_key).strftime('%b %Y')} (by portfolio)"
    subtitle = f"Total: cases={int(tot_cases):,}, complaints={int(tot_comps):,}, per_1000={tot_per_1000}"

    return (title, subtitle), out
