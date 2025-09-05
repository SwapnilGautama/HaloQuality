# core/join_cases_complaints.py
from __future__ import annotations
import pandas as pd
from typing import Tuple

MONTH_FMT = "%b-%y"  # e.g. "Jun-25"

def _to_month_str(dt: pd.Series) -> pd.Series:
    # Handles datetime, Excel serials already parsed by openpyxl, and strings
    s = pd.to_datetime(dt, errors="coerce")
    return s.dt.strftime(MONTH_FMT)

def _canon_col(df: pd.DataFrame, candidates: list[str], new_name: str) -> pd.DataFrame:
    for c in candidates:
        if c in df.columns:
            if c != new_name:
                df = df.rename(columns={c: new_name})
            return df
    return df  # if not found, caller can add it

def _num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")

def build_cases_complaints_join(
    cases_df: pd.DataFrame,
    complaints_df: pd.DataFrame,
    rca_df: pd.DataFrame | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      joined_summary: metrics at (Month, Portfolio_std, Process) with complaints/1000
      rca_share: RCA1/RCA2 distribution at same grain (if rca present in complaints_df)
    """

    # -----------------------
    # Normalize Cases columns
    # -----------------------
    c = cases_df.copy()

    # Canonical names we’ll use everywhere:
    # - Portfolio_std
    # - ProcessName
    # - Create_Date   (source)
    c = _canon_col(c, ["Portfolio_std", "Portfolio", "Portfolio Name"], "Portfolio_std")
    c = _canon_col(c, ["ProcessName", "Process Name", "Process"], "ProcessName")
    c = _canon_col(c, ["Create Date", "Create_Date", "CreateDate"], "Create_Date")

    # Month
    if "Month" not in c.columns:
        c["Month"] = _to_month_str(c["Create_Date"])

    # Unique cases = unique Case ID per join grain
    case_id_col = None
    for cand in ["Case ID", "CaseID", "Case_Id", "Case"]:
        if cand in c.columns:
            case_id_col = cand
            break
    if case_id_col is None:
        # fall back to counting rows if no explicit ID
        c["__row_id__"] = range(len(c))
        case_id_col = "__row_id__"

    # Optional numeric metric: “No of Days” => avg_days
    days_col = None
    for cand in ["No of Days", "No_of_Days", "Days"]:
        if cand in c.columns:
            days_col = cand
            c[cand] = _num(c[cand])
            break

    # Aggregate Cases at Month x Portfolio x Process
    cases_gb = (
        c.groupby(["Month", "Portfolio_std", "ProcessName"], dropna=False)
         .agg(
             Unique_Cases=(case_id_col, "nunique"),
             Avg_Days=(days_col, "mean") if days_col else ("Month", "size")  # if absent, dummy
         )
         .reset_index()
    )
    if not days_col:
        cases_gb = cases_gb.drop(columns=["Avg_Days"])

    # ----------------------------
    # Normalize Complaints columns
    # ----------------------------
    comp = complaints_df.copy()
    comp = _canon_col(comp, ["Portfolio_std", "Portfolio", "Portfolio Name"], "Portfolio_std")
    comp = _canon_col(comp, ["Parent Case Type", "Parent_Case_Type", "ProcessName"], "Parent_Case_Type")
    # month can be in a date-like field or already precomputed
    # try common candidates for "complaints date"
    month_src = None
    for cand in ["Report_Date", "Complaints_Date", "Date", "Created Date", "Create Date", "Month"]:
        if cand in comp.columns:
            month_src = cand
            break
    if month_src is None:
        comp["Month"] = pd.NaT
    else:
        if month_src == "Month":
            comp["Month"] = comp["Month"].astype(str)
        else:
            comp["Month"] = _to_month_str(comp[month_src])

    # Count complaints per Month x Portfolio x Parent_Case_Type
    comp_gb = (
        comp.groupby(["Month", "Portfolio_std", "Parent_Case_Type"], dropna=False)
            .size()
            .reset_index(name="Complaints")
    )

    # -------------------------------------------
    # Join Cases and Complaints on Month/Process/Portfolio
    # -------------------------------------------
    # Align "process" dimension name on both sides
    cases_gb = cases_gb.rename(columns={"ProcessName": "Process"})
    comp_gb = comp_gb.rename(columns={"Parent_Case_Type": "Process"})

    joined = pd.merge(
        cases_gb,
        comp_gb,
        on=["Month", "Portfolio_std", "Process"],
        how="left",
        validate="one_to_one"
    )

    # Ensure numeric
    for col in ["Unique_Cases", "Complaints", "Avg_Days"]:
        if col in joined.columns:
            joined[col] = _num(joined[col]).fillna(0)

    # Complaints per 1000 cases
    joined["Complaints_per_1000"] = 1000.0 * joined["Complaints"].div(joined["Unique_Cases"]).replace([pd.NA, pd.NaT], 0)
    joined["Complaints_per_1000"] = joined["Complaints_per_1000"].fillna(0)

    # Sort tidy
    joined = joined.sort_values(["Month", "Portfolio_std", "Process"], kind="mergesort").reset_index(drop=True)

    # -------------------------------------------
    # RCA shares (if complaints_df contains labels)
    # -------------------------------------------
    rca_share = pd.DataFrame()
    for rca_col in ["RCA1", "RCA2"]:
        if rca_col in comp.columns:
            tmp = (
                comp.groupby(["Month", "Portfolio_std", "Parent_Case_Type", rca_col], dropna=False)
                    .size()
                    .reset_index(name="Complaints")
            )
            tmp = tmp.rename(columns={"Parent_Case_Type": "Process"})
            # attach cases to compute share per 1k if needed
            tmp = pd.merge(
                tmp, cases_gb.rename(columns={"ProcessName": "Process"}),
                on=["Month", "Portfolio_std", "Process"], how="left"
            )
            tmp["Unique_Cases"] = _num(tmp["Unique_Cases"]).fillna(0)
            tmp["Complaints"] = _num(tmp["Complaints"]).fillna(0)
            tmp["Complaints_per_1000"] = 1000.0 * tmp["Complaints"].div(tmp["Unique_Cases"]).replace([pd.NA, pd.NaT], 0)
            tmp["Complaints_per_1000"] = tmp["Complaints_per_1000"].fillna(0)
            tmp["RCA_Level"] = rca_col
            tmp = tmp.rename(columns={rca_col: "RCA_Value"})
            rca_share = pd.concat([rca_share, tmp], ignore_index=True)

    return joined, rca_share
