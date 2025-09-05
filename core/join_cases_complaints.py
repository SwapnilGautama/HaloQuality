# core/join_cases_complaints.py
from __future__ import annotations
import numpy as np
import pandas as pd

def _to_month(s) -> pd.Series:
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
    candidates = [
        "Parent_Case_Type", "Parent Case Type", "Parent case type",
        "Parent case type (standardised)", "ParentCaseType"
    ]
    for c in candidates:
        if c in df.columns: return c
    for c in df.columns:
        cl = c.lower()
        if "parent" in cl and "case" in cl and "type" in cl:
            return c
    raise ValueError("Complaints: cannot find Parent Case Type column")

def _pick_process_name(df: pd.DataFrame) -> str:
    for c in ["Process Name", "ProcessName", "Process", "Process_Name"]:
        if c in df.columns: return c
    for c in df.columns:
        if "process" in c.lower(): return c
    raise ValueError("Cases: cannot find Process Name column")

def _pick_case_id(df: pd.DataFrame) -> str:
    # explicit aliases first
    aliases = [
        "Case ID","Case_Id","CaseID","Case Id","case_id",
        "Case Number","Case_Number","CaseNo","Case_No",
        "CaseRef","Case Ref","Case Reference","CaseReference",
    ]
    for a in aliases:
        if a in df.columns:
            return a
    # regex-ish fallback: any column containing 'case' and one of id/ref/no/number
    for c in df.columns:
        cl = c.lower().replace(" ", "")
        if "case" in cl and any(k in cl for k in ("id","ref","no","number")):
            return c
    # last resort: a unique-id style column
    for c in df.columns:
        if "unique" in c.lower() and "id" in c.lower():
            return c
    raise ValueError("Cases: missing Case ID column")

def build_cases_complaints_join(
    cases: pd.DataFrame,
    complaints: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:

    # Drop duplicate-labelled columns that confuse groupby
    c = cases.loc[:, ~cases.columns.duplicated()].copy()
    comp = complaints.loc[:, ~complaints.columns.duplicated()].copy()

    # ---- Month on cases (Create Date preferred) ----
    if "Month" not in c.columns:
        for cd in ["Create Date","Create_Date","CreateDate","Create Dt","Create_Dt"]:
            if cd in c.columns:
                c["Month"] = _to_month(c[cd]); break
        else:
            if "month" in c.columns:
                c["Month"] = c["month"].astype(str)
            elif "Report_Date" in c.columns:
                c["Month"] = _to_month(c["Report_Date"])
            else:
                raise ValueError("Cases: cannot determine Month (need Create Date or Report_Date)")
    else:
        c["Month"] = c["Month"].astype(str)

    # ---- Month on complaints ----
    if "Month" not in comp.columns:
        if "month" in comp.columns:
            comp["Month"] = comp["month"].astype(str)
        else:
            # best-effort: use first date-ish column
            date_cols = [x for x in comp.columns if "date" in x.lower()]
            if not date_cols:
                raise ValueError("Complaints: cannot determine Month")
            comp["Month"] = _to_month(comp[date_cols[0]])
    else:
        comp["Month"] = comp["Month"].astype(str)

    # ---- Portfolio std (title case for display; same on both sides) ----
    c_port_src = "Portfolio_std" if "Portfolio_std" in c.columns else ("Portfolio" if "Portfolio" in c.columns else None)
    comp_port_src = "Portfolio_std" if "Portfolio_std" in comp.columns else ("Portfolio" if "Portfolio" in comp.columns else None)
    if c_port_src is None or comp_port_src is None:
        raise ValueError("Missing Portfolio/Portfolio_std in cases or complaints")

    c["Portfolio_std"] = _std_text(c[c_port_src], to_title=True)
    comp["Portfolio_std"] = _std_text(comp[comp_port_src], to_title=True)

    # ---- Process std ----
    proc_cases_col = _pick_process_name(c)
    parent_col = _pick_parent_case_type(comp)
    c["Process_std"] = _std_text(c[proc_cases_col])
    comp["Process_std"] = _std_text(comp[parent_col])

    # ---- Unique cases & Avg days ----
    case_id_col = _pick_case_id(c)

    no_days_col = None
    for k in ["No of Days", "No_of_Days", "No. of Days", "NoOfDays"]:
        if k in c.columns:
            no_days_col = k
            break

    cases_gb = (
        c.groupby(["Month","Portfolio_std","Process_std"], dropna=False)
         .agg(
             Unique_Cases=(case_id_col, pd.Series.nunique),
             Avg_No_of_Days=(no_days_col, "mean") if no_days_col else (case_id_col, "size"),
         )
         .reset_index()
    )

    # ---- Complaints count ----
    comp_gb = (
        comp.groupby(["Month","Portfolio_std","Process_std"], dropna=False)
            .size()
            .reset_index(name="Complaints")
    )

    # ---- Join & rate ----
    joined = cases_gb.merge(comp_gb, on=["Month","Portfolio_std","Process_std"], how="left")
    joined["Complaints"]   = pd.to_numeric(joined["Complaints"],   errors="coerce").fillna(0)
    joined["Unique_Cases"] = pd.to_numeric(joined["Unique_Cases"], errors="coerce")

    denom = joined["Unique_Cases"].where(joined["Unique_Cases"] > 0)
    joined["Complaints_per_1000"] = (joined["Complaints"] * 1000.0) / denom
    joined["Complaints_per_1000"] = joined["Complaints_per_1000"].round(2)

    keys = ["Month","Portfolio_std","Process_std"]
    joined = joined.sort_values(keys).reset_index(drop=True)

    # ---- RCA table if available ----
    if "RCA1" in comp.columns:
        rca = (
            comp.groupby(keys + ["RCA1"], dropna=False)
                .size()
                .reset_index(name="RCA1_Count")
                .sort_values(keys + ["RCA1"])
                .reset_index(drop=True)
        )
    else:
        rca = pd.DataFrame(columns=keys + ["RCA1","RCA1_Count"])

    return joined, rca
