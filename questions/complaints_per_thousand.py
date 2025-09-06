# questions/complaints_per_thousand.py
from __future__ import annotations
import pandas as pd

TITLE = "Complaints per 1,000 cases"

# --- FIXED column names you confirmed ---
CASE_ID_COL               = "Case ID"
CASE_PORTFOLIO_COL        = "Portfolio"
CASE_PROCESS_COLS         = ["Process", "Process Name"]        # support either
CASE_DATE_COL             = "Create Date"

COMP_ID_COL               = "Original Process Affected Case ID"
COMP_PROCESS_COL          = "Parent Case Type"
COMP_DATE_COL             = "Date Complaint Received - DD/MM/YY"

def _get_col(df: pd.DataFrame, names: list[str]) -> str | None:
    for n in names:
        if n in df.columns:
            return n
    return None

def _month_bounds(params: dict) -> tuple[pd.Timestamp, pd.Timestamp]:
    if params.get("start_month") and params.get("end_month"):
        start = pd.to_datetime(params["start_month"])
        end   = pd.to_datetime(params["end_month"]) + pd.offsets.MonthEnd(0)
    elif params.get("relative_months"):
        # e.g., last 3 months
        end = pd.Timestamp.today().normalize() + pd.offsets.MonthEnd(0)
        start = (end - pd.offsets.MonthBegin(int(params["relative_months"])) + pd.offsets.MonthBegin(1)).normalize()
    else:
        # default: last 3 months
        end = pd.Timestamp.today().normalize() + pd.offsets.MonthEnd(0)
        start = (end - pd.offsets.MonthBegin(3) + pd.offsets.MonthBegin(1)).normalize()
    # ensure month starts
    start = start.replace(day=1)
    return start, end

def _portfolio_mask(series: pd.Series, portfolio_val: str | None) -> pd.Series:
    if not portfolio_val:
        return pd.Series(True, index=series.index)
    s = series.fillna("").astype(str)
    # exact match if possible, fallback to contains
    exact = s.str.casefold().eq(portfolio_val.casefold())
    return exact if exact.any() else s.str.contains(portfolio_val, case=False, na=False)

def _safe_to_datetime(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce", dayfirst=True, infer_datetime_format=True)

def run(store, params=None, user_text: str | None = None):
    params = params or {}

    # --- pull dataframes
    cases = store["cases"].copy()
    complaints = store["complaints"].copy()

    # --- resolve columns (hard-wired to your fields)
    case_id = CASE_ID_COL if CASE_ID_COL in cases.columns else _get_col(cases, [CASE_ID_COL])
    case_pf = CASE_PORTFOLIO_COL
    case_proc = _get_col(cases, CASE_PROCESS_COLS)
    case_dt = CASE_DATE_COL

    if not (case_id and case_pf and case_proc and case_dt):
        return (
            TITLE,
            pd.DataFrame(),
            f"Missing columns in cases. Need: '{CASE_ID_COL}', '{CASE_PORTFOLIO_COL}', "
            f"'{CASE_PROCESS_COLS[0]}' or '{CASE_PROCESS_COLS[1]}', '{CASE_DATE_COL}'. "
            f"Found: {list(cases.columns)}"
        )

    comp_id = COMP_ID_COL
    comp_proc = COMP_PROCESS_COL
    comp_dt = COMP_DATE_COL

    if not all(c in complaints.columns for c in [comp_id, comp_proc, comp_dt]):
        return (
            TITLE,
            pd.DataFrame(),
            f"Missing columns in complaints. Need: '{COMP_ID_COL}', '{COMP_PROCESS_COL}', '{COMP_DATE_COL}'. "
            f"Found: {list(complaints.columns)}"
        )

    # --- date coercion
    cases[case_dt] = _safe_to_datetime(cases[case_dt])
    complaints[comp_dt] = _safe_to_datetime(complaints[comp_dt])

    # --- month bounds
    start, end = _month_bounds(params)

    # --- filter cases by month + (optional) portfolio
    cases["month"] = cases[case_dt].dt.to_period("M").dt.to_timestamp()
    mask = (cases[case_dt] >= start) & (cases[case_dt] <= end)
    pf_val = (params.get("portfolio") or "").strip()
    if pf_val:
        mask &= _portfolio_mask(cases[case_pf], pf_val)

    cases_f = cases.loc[mask, [case_id, case_pf, case_proc, "month"]].dropna(subset=["month"])
    if cases_f.empty:
        note = f"No cases after applying the selected filters/date window."
        return (TITLE, pd.DataFrame(), note)

    # --- aggregate cases by month × process
    cases_by = (
        cases_f.groupby(["month", case_proc], as_index=False, dropna=False)
               .size().rename(columns={"size": "cases", case_proc: "process"})
    )

    # --- prepare complaints
    comp = complaints[[comp_id, comp_proc, comp_dt]].copy()
    comp["month"] = complaints[comp_dt].dt.to_period("M").dt.to_timestamp()
    comp = comp.rename(columns={comp_proc: "process", comp_id: case_id})

    # join to cases for portfolio and then filter portfolio (so complaints inherit portfolio from cases)
    comp = comp.merge(cases[[case_id, case_pf]], on=case_id, how="left")
    if pf_val:
        comp = comp[_portfolio_mask(comp[case_pf], pf_val)]
    comp = comp[(comp[comp_dt] >= start) & (comp[comp_dt] <= end)]

    # --- aggregate complaints by month × process
    complaints_by = (
        comp.groupby(["month", "process"], as_index=False, dropna=False)
            .size().rename(columns={"size": "complaints"})
    )

    # --- combine + compute per 1000
    out = cases_by.merge(complaints_by, on=["month", "process"], how="left")
    out["complaints"] = out["complaints"].fillna(0).astype(int)
    out["per_1000"] = (out["complaints"] / out["cases"]).fillna(0) * 1000.0
    out["_month"] = out["month"].dt.strftime("%b %y")

    # final tidy
    out = out[["_month", "process", "cases", "complaints", "per_1000"]].sort_values(["_month", "process"])
    return TITLE, out, None
