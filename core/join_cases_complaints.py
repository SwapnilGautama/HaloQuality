# core/join_cases_complaints.py
from __future__ import annotations
import re
import numpy as np
import pandas as pd

MONTH_FMT = "%Y-%m"

def _to_month(s: pd.Series | list | np.ndarray) -> pd.Series:
    dt = pd.to_datetime(s, errors="coerce", dayfirst=True)
    return dt.dt.to_period("M").astype(str)

def _std_text(s: pd.Series, *, to_title=False) -> pd.Series:
    out = (
        s.astype(str)
         .str.strip()
         .str.replace(r"\s+", " ", regex=True)
    )
    if to_title:
        return out.str.title()
    return (out
            .str.lower()
            .str.replace("&", "and")
            .str.replace("-", " ")
            .str.replace("/", " ")
            .str.replace(r"[^a-z0-9 ]", "", regex=True))

def _pick_parent_case_type(df: pd.DataFrame) -> str:
    # try canonical first
    for c in ["Parent_Case_Type", "Parent Case Type", "Parent case type",
              "Parent case type (standardised)", "ParentCaseType"]:
        if c in df.columns:
            return c
    # fallback: any column that looks like 'parent ... case ... type'
    for c in df.columns:
        cl = c.lower()
        if "parent" in cl and "case" in cl and "type" in cl:
            return c
    raise ValueError("Complaints DataFrame has no Parent Case Type column")

def _pick_process_name(df: pd.DataFrame) -> str:
    for c in ["Process Name", "ProcessName", "Process", "Process_Name"]:
        if c in df.columns:
            return c
    # sometimes process is embedded in a “Process Group” – we still need a column
    for c in df.columns:
        if "process" in c.lower():
            return c
    raise ValueError("Cases DataFrame has no Process Name column")

def build_cases_complaints_join(
    cases: pd.DataFrame,
    complaints: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    c = cases.copy()
    comp = complaints.copy()

    # --- Month ---
    if "Month" not in c.columns:
        # cases Month should be from Create Date; try common names
        for cd in ["Create Date", "Create_Date", "CreateDate", "Create Dt", "Create_Dt"]:
            if cd in c.columns:
                c["Month"] = _to_month(c[cd])
                break
        else:
            # fallback to already-computed 'month' or 'Report_Date'
            if "month" in c.columns:
                c["Month"] = c["month"].astype(str)
            elif "Report_Date" in c.columns:
                c["Month"] = _to_month(c["Report_Date"])
            else:
                raise ValueError("Cases: cannot determine Month (need Create Date or Report_Date)")
    else:
        c["Month"] = c["Month"].astype(str)

    if "Month" not in comp.columns:
        if "month" in comp.columns:
            comp["Month"] = comp["month"].astype(str)
        else:
            # try any date-like column as last resort
            date_cols = [x for x in comp.columns if "date" in x.lower()]
            if not date_cols:
                raise ValueError("Complaints: cannot determine Month")
            comp["Month"] = _to_month(comp[date_cols[0]])
    else:
        comp["Month"] = comp["Month"].astype(str)

    # --- Portfolio (std) ---
    c_port_src = "Portfolio_std" if "Portfolio_std" in c.columns else ("Portfolio" if "Portfolio" in c.columns else None)
    comp_port_src = "Portfolio_std" if "Portfolio_std" in comp.columns else ("Portfolio" if "Portfolio" in comp.columns else None)
    if c_port_src is None or comp_port_src is None:
        raise ValueError("Missing Portfolio/Portfolio_std in cases or complaints")

    c["Portfolio_std"] = _std_text(c[c_port_src], to_title=True)
    comp["Portfolio_std"] = _std_text(comp[comp_port_src], to_title=True)

    # --- Process normalisation ---
    proc_cases_col = _pick_process_name(c)
    parent_col = _pick_parent_case_type(comp)
    c["Process_std"] = _std_text(c[proc_cases_col])
    comp["Process_std"] = _std_text(comp[parent_col])

    # --- Unique case count & avg days ---
    case_id_col = None
    for k in ["Case ID", "CaseID", "Case_Id", "CaseId"]:
        if k in c.columns:
            case_id_col = k
            break
    if case_id_col is None:
        raise ValueError("Cases: missing Case ID column")

    no_days_col = None
    for k in ["No of Days", "No_of_Days", "No. of Days", "NoOfDays"]:
        if k in c.columns:
            no_days_col = k
            break

    cases_gb = (
        c.groupby(["Month", "Portfolio_std", "Process_std"], dropna=False)
         .agg(
             Unique_Cases=(case_id_col, pd.Series.nunique),
             Avg_No_of_Days=(no_days_col, "mean") if no_days_col else (case_id_col, "size"),
         )
         .reset_index()
    )

    # --- Complaints count ---
    # ensure we’re grouping on 1-D columns
    comp = comp.loc[:, ~comp.columns.duplicated()]
    comp_gb = (
        comp.groupby(["Month", "Portfolio_std", "Process_std"], dropna=False)
            .size()
            .reset_index(name="Complaints")
    )

    # --- Join & KPIs ---
    joined = cases_gb.merge(comp_gb, on=["Month", "Portfolio_std", "Process_std"], how="left")
    joined["Complaints"] = pd.to_numeric(joined["Complaints"], errors="coerce").fillna(0)
    joined["Unique_Cases"] = pd.to_numeric(joined["Unique_Cases"], errors="coerce")

    denom = joined["Unique_Cases"].where(joined["Unique_Cases"] > 0)
    joined["Complaints_per_1000"] = (joined["Complaints"] * 1000.0) / denom
    joined["Complaints_per_1000"] = joined["Complaints_per_1000"].round(2)

    keys = ["Month", "Portfolio_std", "Process_std"]
    joined = joined.sort_values(keys).reset_index(drop=True)

    # --- RCA (if already labeled) ---
    if "RCA1" in comp.columns:
        rca = (
            comp.groupby(keys + ["RCA1"], dropna=False)
                .size()
                .reset_index(name="RCA1_Count")
                .sort_values(keys + ["RCA1"])
                .reset_index(drop=True)
        )
    else:
        rca = pd.DataFrame(columns=keys + ["RCA1", "RCA1_Count"])

    return joined, rca
