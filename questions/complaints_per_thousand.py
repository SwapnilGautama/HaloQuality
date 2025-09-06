# questions/complaints_per_thousand.py
from __future__ import annotations

import pandas as pd
from typing import Dict, Any, Optional
from datetime import datetime
from dateutil.relativedelta import relativedelta


def _first_present(d: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    """Return the first column name from `candidates` that exists in the dataframe (case-insensitive)."""
    lower = {c.lower(): c for c in d.columns}
    for name in candidates:
        if name.lower() in lower:
            return lower[name.lower()]
    return None


def _monthify(s: pd.Series) -> pd.Series:
    """Coerce to datetime and normalize to first-of-month."""
    dt = pd.to_datetime(s, errors="coerce")
    return dt.dt.to_period("M").dt.to_timestamp()


def _coerce_str(x):
    return None if pd.isna(x) else str(x)


def _normalize_process_strings(s: pd.Series) -> pd.Series:
    return s.fillna("").astype(str).str.strip()


def _month_bounds_from_params(params: Dict[str, Any], cases: pd.DataFrame, date_col: str):
    """
    Resolve (start_month, end_month) from params.
    Accepts:
      - start_month / end_month (YYYY-MM or YYYY-MM-DD)
      - relative_months (int) -> last N months ending at max month in cases
    """
    start = params.get("start_month")
    end = params.get("end_month")
    rel = params.get("relative_months")

    # derive dataset max month for sensible defaults
    data_max = _monthify(cases[date_col]).max()
    if pd.isna(data_max):
        data_max = pd.Timestamp(datetime.utcnow()).to_period("M").to_timestamp()

    if start and end:
        s = pd.to_datetime(start, errors="coerce")
        e = pd.to_datetime(end, errors="coerce")
        if pd.isna(s) or pd.isna(e):
            # if parsing failed, fall back to relative or last 3
            s, e = None, None
        else:
            return (
                s.to_period("M").to_timestamp(),
                e.to_period("M").to_timestamp()
            )

    if isinstance(rel, (int, float)) and rel > 0:
        end = data_max
        start = (end - relativedelta(months=int(rel) - 1)).to_period("M").to_timestamp()
        return start, end

    # default: last 3 months
    end = data_max
    start = (end - relativedelta(months=2)).to_period("M").to_timestamp()
    return start, end


def run(store: Dict[str, Any], params: Dict[str, Any] = None, user_text: str = "") -> Dict[str, Any]:
    """
    Output: dict with {"dataframe": DataFrame, "table_title": str}
    Columns returned: month, process, cases, complaints, per_1000
    """
    params = params or {}

    # --- Pull data
    cases = store.get("cases")
    complaints = store.get("complaints")
    if cases is None or complaints is None or cases.empty:
        return {"message": "No data available."}

    # --- Identify key columns in cases
    cases_date_col = _first_present(
        cases,
        [
            "Case Created Date",
            "Created Date",
            "Creation Date",
            "Case Opened Date",
            "Opened Date",
            "Received Date",
            "Start Date",
            "Date",
        ],
    )
    if not cases_date_col:
        return {"message": "Could not find a date column in cases."}

    cases_process_col = _first_present(cases, ["Process Name", "Process", "Parent Case Type"])
    if not cases_process_col:
        return {"message": "Could not find a process column in cases (e.g. 'Process Name')."}

    portfolio_col = _first_present(cases, ["Portfolio", "portfolio"])
    # portfolio filter value (e.g., 'London')
    portfolio_value = params.get("portfolio")

    # --- Identify key columns in complaints
    comp_case_id_col = _first_present(
        complaints,
        ["Original Process Affected Case ID", "Original Case ID", "Case ID", "Linked Case ID"],
    )
    comp_process_col = _first_present(
        complaints,
        ["Parent Case Type", "Process Name", "Process"],
    )
    comp_date_col = _first_present(
        complaints,
        [
            "Complaint Opened Date",
            "Received Date",
            "Created Date",
            "Date",
            "Complaint Date",
        ],
    )

    # --- Normalize months
    cases = cases.copy()
    cases["_month"] = _monthify(cases[cases_date_col])

    # Month window
    start_m, end_m = _month_bounds_from_params(params, cases, cases_date_col)
    mask = (cases["_month"] >= start_m) & (cases["_month"] <= end_m)
    if portfolio_col and portfolio_value:
        cases = cases.loc[mask & (cases[portfolio_col].astype(str).str.strip().str.lower()
                                  == str(portfolio_value).strip().lower())]
    else:
        cases = cases.loc[mask]

    # We need a per-month-per-process denominator from cases
    cases_proc = (
        cases.assign(_proc=_normalize_process_strings(cases[cases_process_col]))
        .dropna(subset=["_month", "_proc"])
        .groupby(["_month", "_proc"], dropna=False)
        .size()
        .rename("cases")
        .reset_index()
    )

    # --- Complaints mapping to process + month
    comps = complaints.copy()
    # Month for complaints (use complaints date if available; fall back to case month via join later)
    if comp_date_col:
        comps["_comp_month"] = _monthify(comps[comp_date_col])
    else:
        comps["_comp_month"] = pd.NaT

    comps["_comp_proc"] = ""
    if comp_process_col:
        comps["_comp_proc"] = _normalize_process_strings(comps[comp_process_col])

    # Join complaints → cases by Case ID if we can (to inherit portfolio/process/month)
    if comp_case_id_col and "Case ID" in cases.columns:
        # Prepare a small cases lookup with needed columns
        lookup_cols = ["Case ID", "_month", cases_process_col]
        if portfolio_col:
            lookup_cols.append(portfolio_col)

        cases_lookup = cases[lookup_cols].drop_duplicates("Case ID")

        merged = comps.merge(
            cases_lookup,
            left_on=comp_case_id_col,
            right_on="Case ID",
            how="left",
            suffixes=("", "_case"),
        )

        # Decide final complaint month: prefer complaints own month, else case month
        merged["_final_month"] = merged["_comp_month"]
        merged.loc[merged["_final_month"].isna(), "_final_month"] = merged["_month"]

        # Decide final complaint process: prefer linked case's process, else Parent Case Type
        merged["_final_proc"] = merged[cases_process_col]
        fallback_mask = merged["_final_proc"].isna() | (merged["_final_proc"].astype(str).str.strip() == "")
        merged.loc[fallback_mask, "_final_proc"] = merged["_comp_proc"]

        # Portfolio filter (if any)
        if portfolio_col and portfolio_value:
            merged = merged[
                (merged[portfolio_col].astype(str).str.strip().str.lower()
                 == str(portfolio_value).strip().lower())
            ]
    else:
        # No join possible → rely on own month/process
        merged = comps.rename(columns={"_comp_month": "_final_month", "_comp_proc": "_final_proc"})

    # Complaints aggregation
    complaints_agg = (
        merged.dropna(subset=["_final_month"])
        .assign(_final_proc=_normalize_process_strings(merged["_final_proc"]))
        .groupby(["_final_month", "_final_proc"], dropna=False)
        .size()
        .rename("complaints")
        .reset_index()
        .rename(columns={"_final_month": "_month", "_final_proc": "_proc"})
    )

    # --- Combine denominators and numerators
    out = cases_proc.merge(complaints_agg, on=["_month", "_proc"], how="outer")
    out["cases"] = out["cases"].fillna(0).astype(int)
    out["complaints"] = out["complaints"].fillna(0).astype(int)
    out["per_1000"] = out.apply(lambda r: (r["complaints"] / r["cases"] * 1000) if r["cases"] > 0 else 0.0, axis=1)

    # Format month and clean output
    out = (
        out.assign(month=lambda d: d["_month"].dt.strftime("%b %y"))
        .rename(columns={"_proc": "process"})
        .loc[:, ["month", "process", "cases", "complaints", "per_1000"]]
        .sort_values(["month", "process"])
        .reset_index(drop=True)
    )

    title_bits = []
    if portfolio_value:
        title_bits.append(f"Portfolio: {portfolio_value}")
    title_bits.append(f"Range: {start_m.strftime('%b %y')} → {end_m.strftime('%b %y')}")
    return {
        "table_title": " | ".join(title_bits),
        "dataframe": out,
    }
