# questions/complaints_per_thousand.py
from __future__ import annotations

import pandas as pd
from typing import Dict, Any, Optional
from datetime import datetime
from dateutil.relativedelta import relativedelta


# -----------------------------
# Helpers
# -----------------------------
def _first_present(d: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    if d is None:
        return None
    lower = {c.lower(): c for c in d.columns}
    for name in candidates:
        if name.lower() in lower:
            return lower[name.lower()]
    return None


def _detect_date_col(df: pd.DataFrame) -> Optional[str]:
    """Generic date detector for cases (complaints have a fixed date first)."""
    if df is None or df.empty:
        return None

    known = [
        "Case Created Date", "Created Date", "Creation Date",
        "Case Opened Date", "Opened Date",
        "Received Date", "Start Date", "Date",
        "Case Creation Date", "First Touch Date", "Case Start Date",
        "Effective Date", "Service Date", "Logged Date",
        "FPA Date", "FPA Received Date"
    ]
    col = _first_present(df, known)
    if col:
        return col

    # keyword based
    for c in df.columns:
        name = c.lower()
        if any(k in name for k in ["date", "created", "opened", "received", "start", "logged", "effective"]):
            return c

    # heuristic: choose col with best datetime parse ratio (>50%)
    best, best_ratio = None, 0.0
    for c in df.columns:
        try:
            dt = pd.to_datetime(df[c], errors="coerce", dayfirst=True)
            ratio = float(dt.notna().mean())
            if ratio > 0.5 and ratio > best_ratio:
                best, best_ratio = c, ratio
        except Exception:
            pass
    return best


def _monthify_series(s: pd.Series, *, dayfirst: bool = True) -> pd.Series:
    dt = pd.to_datetime(s, errors="coerce", dayfirst=dayfirst)
    return dt.dt.to_period("M").dt.to_timestamp()


def _normalize_process_strings(s: pd.Series) -> pd.Series:
    return s.fillna("").astype(str).str.strip()


def _month_bounds_from_params(params: Dict[str, Any], cases: pd.DataFrame, date_col: str):
    start = params.get("start_month")
    end = params.get("end_month")
    rel = params.get("relative_months")

    data_max = _monthify_series(cases[date_col]).max()
    if pd.isna(data_max):
        data_max = pd.Timestamp(datetime.utcnow()).to_period("M").to_timestamp()

    if start and end:
        s = pd.to_datetime(start, errors="coerce")
        e = pd.to_datetime(end, errors="coerce")
        if not (pd.isna(s) or pd.isna(e)):
            return s.to_period("M").to_timestamp(), e.to_period("M").to_timestamp()

    if isinstance(rel, (int, float)) and rel > 0:
        end = data_max
        start = (end - relativedelta(months=int(rel) - 1)).to_period("M").to_timestamp()
        return start, end

    # default last 3 months
    end = data_max
    start = (end - relativedelta(months=2)).to_period("M").to_timestamp()
    return start, end


# -----------------------------
# Main
# -----------------------------
def run(store: Dict[str, Any], params: Dict[str, Any] = None, user_text: str = "") -> Dict[str, Any]:
    """
    Expects in `store`:
      - cases (DataFrame)
      - complaints (DataFrame)

    Complaints date is taken from 'Date Complaint Received - DD/MM/YY' (day-first),
    with a fallback to generic detection if that exact column is missing.
    """
    params = params or {}

    cases = store.get("cases")
    complaints = store.get("complaints")
    if cases is None or cases.empty or complaints is None:
        return {"message": "No data available."}

    # ----- CASES: detect date & other columns
    cases_date_col = _detect_date_col(cases)
    if not cases_date_col:
        return {"message": "Could not find a date column in cases."}

    cases_proc_col = _first_present(cases, ["Process Name", "Process", "Parent Case Type"])
    if not cases_proc_col:
        return {"message": "Could not find a process column in cases (e.g. 'Process Name')."}

    portfolio_col = _first_present(cases, ["Portfolio", "portfolio"])
    portfolio_val = params.get("portfolio")

    # monthify & filter cases
    cases = cases.copy()
    cases["_month"] = _monthify_series(cases[cases_date_col], dayfirst=True)

    start_m, end_m = _month_bounds_from_params(params, cases, cases_date_col)
    mask = (cases["_month"] >= start_m) & (cases["_month"] <= end_m)
    if portfolio_col and portfolio_val:
        mask = mask & (
            cases[portfolio_col].astype(str).str.strip().str.lower()
            == str(portfolio_val).strip().lower()
        )
    cases = cases.loc[mask]

    # denominator: cases by month x process
    cases_proc = (
        cases.assign(_proc=_normalize_process_strings(cases[cases_proc_col]))
        .dropna(subset=["_month"])
        .groupby(["_month", "_proc"], dropna=False)
        .size()
        .rename("cases")
        .reset_index()
    )

    # ----- COMPLAINTS: fixed date column first
    complaints = complaints.copy()
    # Prefer the exact column name (case-insensitive)
    preferred_complaints_date = _first_present(
        complaints, ["Date Complaint Received - DD/MM/YY"]
    )
    if preferred_complaints_date:
        comp_month = _monthify_series(complaints[preferred_complaints_date], dayfirst=True)
    else:
        # Fallback if older file has a different name
        detected = _detect_date_col(complaints)
        comp_month = _monthify_series(complaints[detected], dayfirst=True) if detected else pd.NaT

    complaints["_comp_month"] = comp_month

    comp_case_id_col = _first_present(
        complaints,
        ["Original Process Affected Case ID", "Original Case ID", "Case ID", "Linked Case ID", "Case Ref", "Case Reference"]
    )
    comp_proc_col = _first_present(complaints, ["Parent Case Type", "Process Name", "Process"])

    complaints["_comp_proc"] = _normalize_process_strings(
        complaints[comp_proc_col] if comp_proc_col else pd.Series([""] * len(complaints), index=complaints.index)
    )

    # Try to join complaints to cases on Case ID -> Case ID
    join_case_id = _first_present(cases, ["Case ID", "Case Ref", "Case Reference", "Id"])

    if comp_case_id_col and join_case_id:
        lookup_cols = [join_case_id, "_month", cases_proc_col]
        if portfolio_col:
            lookup_cols.append(portfolio_col)
        cases_lookup = cases[lookup_cols].drop_duplicates(subset=[join_case_id])

        merged = complaints.merge(
            cases_lookup,
            left_on=comp_case_id_col,
            right_on=join_case_id,
            how="left",
            suffixes=("", "_case"),
        )

        # month preference: complaints' own month, then fall back to the case month
        merged["_final_month"] = merged["_comp_month"]
        fallback_month_mask = merged["_final_month"].isna()
        merged.loc[fallback_month_mask, "_final_month"] = merged["_month"]

        # process preference: process from cases, fall back to complaints process
        merged["_final_proc"] = _normalize_process_strings(merged[cases_proc_col])
        needs_fallback = merged["_final_proc"].eq("").fillna(True)
        merged.loc[needs_fallback, "_final_proc"] = merged["_comp_proc"]

        if portfolio_col and portfolio_val:
            merged = merged[
                merged[portfolio_col].astype(str).str.strip().str.lower()
                == str(portfolio_val).strip().lower()
            ]
    else:
        # No join path → rely on complaints' own process & month
        merged = complaints.rename(columns={"_comp_month": "_final_month", "_comp_proc": "_final_proc"})

    complaints_agg = (
        merged.dropna(subset=["_final_month"])
        .assign(_final_proc=_normalize_process_strings(merged["_final_proc"]))
        .groupby(["_final_month", "_final_proc"], dropna=False)
        .size()
        .rename("complaints")
        .reset_index()
        .rename(columns={"_final_month": "_month", "_final_proc": "_proc"})
    )

    # Combine
    out = cases_proc.merge(complaints_agg, on=["_month", "_proc"], how="outer")
    out["cases"] = out["cases"].fillna(0).astype(int)
    out["complaints"] = out["complaints"].fillna(0).astype(int)
    out["per_1000"] = out.apply(
        lambda r: (r["complaints"] / r["cases"] * 1000) if r["cases"] > 0 else 0.0, axis=1
    )

    out = (
        out.assign(month=lambda d: d["_month"].dt.strftime("%b %y"))
        .rename(columns={"_proc": "process"})
        .loc[:, ["month", "process", "cases", "complaints", "per_1000"]]
        .sort_values(["month", "process"])
        .reset_index(drop=True)
    )

    title_bits = []
    if portfolio_val:
        title_bits.append(f"Portfolio: {portfolio_val}")
    title_bits.append(f"Range: {start_m.strftime('%b %y')} → {end_m.strftime('%b %y')}")
    return {"table_title": " | ".join(title_bits), "dataframe": out}
