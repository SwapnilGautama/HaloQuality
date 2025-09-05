# core/join_cases_complaints.py
from __future__ import annotations
from typing import List, Tuple
import pandas as pd

def _candidate(col: str, df: pd.DataFrame, fallback: str = "") -> str:
    return col if col in df.columns else fallback

def _ensure_keys(df: pd.DataFrame, keys: List[str]) -> pd.DataFrame:
    out = df.copy()
    for k in keys:
        if k not in out.columns:
            out[k] = ""
        out[k] = out[k].fillna("").astype(str)
    return out

def _agg_cases(cases_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate cases by Month / Portfolio_std / ProcessName with distinct Case ID.
    Assumes cases_df already has:
      - Month (from Create Date)
      - Portfolio_std
      - ProcessName
      - Case ID column among ['Case ID', 'CaseID', 'Case_Id'] (best effort)
    """
    if cases_df is None or cases_df.empty:
        return pd.DataFrame()

    c = cases_df.copy()
    c = _ensure_keys(c, ["Month", "Portfolio_std", "ProcessName"])

    # best-effort case id
    id_col = None
    for cand in ["Case ID", "CaseID", "Case_Id", "CaseId"]:
        if cand in c.columns:
            id_col = cand
            break

    if id_col is None:
        # fallback: count rows
        grp = c.groupby(["Month", "Portfolio_std", "ProcessName"], dropna=False)
        out = grp.size().reset_index(name="Unique_Cases")
    else:
        grp = c.groupby(["Month", "Portfolio_std", "ProcessName"], dropna=False)[id_col].nunique()
        out = grp.reset_index(name="Unique_Cases")

    return out

def _agg_complaints(compl_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Aggregate complaints by Month / Portfolio_std / ProcessName.
    Also returns an RCA breakdown table with RCA shares.

    Returns:
      (complaints_summary, rca_breakdown)
        - complaints_summary: keys + Complaints
        - rca_breakdown: keys + RCA1 + Complaints + Share (within same keys)
    """
    if compl_df is None or compl_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    d = compl_df.copy()
    # Ensure join keys exist
    d = _ensure_keys(d, ["Month", "Portfolio_std", "ProcessName"])

    base = (
        d.groupby(["Month", "Portfolio_std", "ProcessName"], dropna=False)
         .size().reset_index(name="Complaints")
    )

    # RCA breakdown
    if "RCA1" in d.columns:
        br = (
            d.groupby(["Month", "Portfolio_std", "ProcessName", "RCA1"], dropna=False)
             .size().reset_index(name="Complaints")
        )
        totals = br.groupby(["Month", "Portfolio_std", "ProcessName"])["Complaints"].transform("sum")
        br["Share"] = (br["Complaints"] / totals).fillna(0.0)
    else:
        br = pd.DataFrame(columns=["Month","Portfolio_std","ProcessName","RCA1","Complaints","Share"])

    return base, br

def build_cases_complaints_join(
    cases_df: pd.DataFrame,
    complaints_df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Produces:
      1) summary_df: Month x Portfolio_std x ProcessName with
           Unique_Cases, Complaints, Complaints_per_1000
      2) rca_breakdown_df: Month x Portfolio_std x ProcessName x RCA1
           Complaints (count) and Share (within the same key)
    Notes
    -----
    - Join key: Month + Portfolio_std + ProcessName
      * If complaints has only Parent_Case_Type, your loader maps it into ProcessName.
    """
    # aggregate
    cases_sum = _agg_cases(cases_df)
    comp_sum, rca = _agg_complaints(complaints_df)

    if cases_sum.empty and comp_sum.empty:
        return pd.DataFrame(), pd.DataFrame()

    keys = ["Month", "Portfolio_std", "ProcessName"]

    # outer join to keep all combos
    joined = pd.merge(cases_sum, comp_sum, how="outer", on=keys)
    for col in ["Unique_Cases","Complaints"]:
        if col not in joined.columns:
            joined[col] = 0
        joined[col] = joined[col].fillna(0)

    # Complaints per 1000 cases
    joined["Complaints_per_1000"] = (
        (joined["Complaints"] * 1000) / joined["Unique_Cases"].replace({0: pd.NA})
    ).astype(float)

    joined = joined.sort_values(keys).reset_index(drop=True)
    rca = rca.sort_values(keys + (["RCA1"] if "RCA1" in rca.columns else [])).reset_index(drop=True)
    return joined, rca
