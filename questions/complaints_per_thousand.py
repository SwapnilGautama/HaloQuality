# questions/complaints_per_thousand.py
from __future__ import annotations
import pandas as pd

from helpers import pick_col, ensure_datetime, portfolio_selector

TITLE = "Complaints per 1,000 cases"

def _month_bounds(start_month: str | None, end_month: str | None):
    """
    Build inclusive month range [start, end].
    Params may be 'YYYY-MM' or full dates; we normalize to month start/end.
    """
    if start_month and end_month:
        start = pd.to_datetime(start_month, errors="coerce")
        end = pd.to_datetime(end_month, errors="coerce")
    else:
        # default: last 3 months including current
        end = pd.Timestamp.today().normalize() + pd.offsets.MonthEnd(0)
        start = (end - pd.offsets.MonthBegin(3)) + pd.offsets.MonthBegin(1)  # 3 months window

    start = (start.normalize() if pd.notna(start) else pd.Timestamp.today()).replace(day=1)
    end = (end.normalize() if pd.notna(end) else pd.Timestamp.today()) + pd.offsets.MonthEnd(0)
    return start, end

def run(store, params=None, user_text: str | None = None):
    """
    Required store keys:
      - cases: dataframe with at least {Case ID, Portfolio, Process, date column}
      - complaints: dataframe with at least {Original Process Affected Case ID, Parent Case Type, Date Complaint Received - DD/MM/YY}
    """
    params = params or {}
    portfolio_val = (params.get("portfolio") or params.get("site") or "").strip()

    # Date window
    start, end = _month_bounds(params.get("start_month"), params.get("end_month"))

    # --- Load data
    cases: pd.DataFrame = store["cases"]
    complaints: pd.DataFrame = store["complaints"]

    # --- Column selection: cases
    case_id_col = pick_col(cases, ["Case ID", "CaseID", "Case_Id", "Original Case ID"], regex=r"\bcase\b.*\bid\b")
    portfolio_col = pick_col(cases, ["Portfolio"])
    case_proc_col = pick_col(cases, ["Process", "Process Name"], regex=r"\bprocess\b")
    case_date_col = pick_col(
        cases,
        ["Create Date", "Report Date", "Start Date", "Report_Date"],
        regex=r"(create|report|start)[ _-]*date",
    )

    if not case_id_col or not portfolio_col or not case_proc_col or not case_date_col:
        return TITLE, pd.DataFrame(), f"Missing columns in cases. Found: id={case_id_col}, portfolio={portfolio_col}, process={case_proc_col}, date={case_date_col}"

    cases = cases.copy()
    cases[case_date_col] = ensure_datetime(cases[case_date_col])
    cases["month"] = cases[case_date_col].dt.to_period("M").dt.to_timestamp()

    # filter by month
    mask = (cases[case_date_col] >= start) & (cases[case_date_col] <= end)
    if portfolio_val:
        mask &= portfolio_selector(cases[portfolio_col], portfolio_val)

    cases_f = cases.loc[mask, [case_id_col, portfolio_col, case_proc_col, "month"]].dropna(subset=["month"])

    if cases_f.empty:
        return TITLE, pd.DataFrame(), f"No cases after applying filters (portfolio='{portfolio_val}' months={start:%b %Y}–{end:%b %Y})."

    # aggregate cases by month+process
    cases_by = (
        cases_f.groupby(["month", case_proc_col], dropna=False, as_index=False)
               .size()
               .rename(columns={"size": "cases", case_proc_col: "process"})
    )

    # --- Column selection: complaints
    comp_id_col = pick_col(
        complaints,
        ["Original Process Affected Case ID", "Original Case ID", "Case ID"],
        regex=r"(original.*affected.*case.*id)|(original.*case.*id)|(^case.*id$)"
    )
    comp_proc_col = pick_col(complaints, ["Parent Case Type", "Process Name"], regex=r"\b(parent)?\s*case\s*type\b|\bprocess\b")
    comp_date_col = pick_col(
        complaints,
        ["Date Complaint Received - DD/MM/YY", "Date Complaint Received", "Date Received"],
        regex=r"(date).*?(complaint).*?(received)|(^date.*received$)"
    )

    if not comp_id_col or not comp_proc_col or not comp_date_col:
        # Show the cases table at least
        out = cases_by.copy()
        out["complaints"] = 0
        out["per_1000"] = 0.0
        return TITLE, out.sort_values(["month", "process"]), (
            f"Missing columns in complaints. Found: id={comp_id_col}, process={comp_proc_col}, date={comp_date_col}"
        )

    comp = complaints[[comp_id_col, comp_proc_col, comp_date_col]].copy()
    comp[comp_date_col] = ensure_datetime(comp[comp_date_col])
    comp = comp[(comp[comp_date_col] >= start) & (comp[comp_date_col] <= end)]
    comp = comp.rename(columns={comp_proc_col: "process", comp_id_col: case_id_col})
    comp["month"] = comp[comp_date_col].dt.to_period("M").dt.to_timestamp()

    # bring portfolio onto complaints via the case id (so we can filter to London)
    comp = comp.merge(cases[[case_id_col, portfolio_col]], on=case_id_col, how="left")
    if portfolio_val:
        comp = comp[portfolio_selector(comp[portfolio_col], portfolio_val)]

    # aggregate complaints by month+process
    complaints_by = (
        comp.groupby(["month", "process"], dropna=False, as_index=False)
            .size()
            .rename(columns={"size": "complaints"})
    )

    # combine and compute per-1000
    out = cases_by.merge(complaints_by, on=["month", "process"], how="left")
    out["complaints"] = out["complaints"].fillna(0).astype(int)
    out["per_1000"] = (out["complaints"] / out["cases"]).replace([pd.NA, pd.NaT], 0).fillna(0) * 1000

    # pretty month
    out["_month"] = out["month"].dt.strftime("%b %y")
    out = out[["_month", "process", "cases", "complaints", "per_1000"]].sort_values(["_month", "process"])

    return TITLE, out, None
