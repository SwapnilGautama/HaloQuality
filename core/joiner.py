# core/joiner.py
from __future__ import annotations
from pathlib import Path
import pandas as pd

def _norm(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower()

def _load_mapping(mapping_path: Path) -> dict:
    """
    Optional mapping file: data/mappings/parent_case_to_process.csv
    Columns (any one of these pairs will work):
      - ParentCaseType, ProcessName
      - ParentCaseType, ProcessNameMap
    """
    if not mapping_path.exists():
        return {}
    df = pd.read_csv(mapping_path)
    cols = [c.strip() for c in df.columns]
    df.columns = cols
    proc_col = "ProcessName" if "ProcessName" in df.columns else ("ProcessNameMap" if "ProcessNameMap" in df.columns else None)
    if proc_col is None or "ParentCaseType" not in df.columns:
        return {}
    df["ParentCaseTypeKey"] = _norm(df["ParentCaseType"])
    df["ProcessKeyMap"] = _norm(df[proc_col])
    return dict(zip(df["ParentCaseTypeKey"], df["ProcessKeyMap"]))


def build_joined_metrics(
    cases_df: pd.DataFrame,
    complaints_df: pd.DataFrame,
    mapping_csv: str | Path = "data/mappings/parent_case_to_process.csv",
) -> pd.DataFrame:
    """
    Join Complaints to Cases on Month (MMM YY) + Portfolio + Process/Parent Case Type.

    - Denominator = unique Case_ID count per (Month, Portfolio, Process)
    - Complaints are grouped per (Month, Portfolio, ParentCaseType)
    - If a mapping file exists, map ParentCaseType -> ProcessName before joining
    - Returns a wide table with:
        Month, Portfolio_std, ProcessName (resolved), ParentCaseType,
        Unique_Cases, Avg_NoOfDays, Complaints_Count, Complaints_per_1000
    """
    if cases_df is None or cases_df.empty or complaints_df is None or complaints_df.empty:
        return pd.DataFrame()

    # ----------------- optional mapping -----------------
    mapping_path = Path(mapping_csv)
    mapping = _load_mapping(mapping_path)
    comp = complaints_df.copy()

    # If mapping exists, create a new mapped ProcessKey; else use raw
    if mapping:
        comp["ProcessKeyMapped"] = comp["ProcessKey"].map(mapping).fillna(comp["ProcessKey"])
    else:
        comp["ProcessKeyMapped"] = comp["ProcessKey"]

    # ----------------- aggregate complaints -----------------
    comp_grp = (
        comp
        .groupby(["Month","Portfolio_std","PortfolioKey","ParentCaseType","ProcessKeyMapped"], dropna=False, as_index=False)
        .size()
        .rename(columns={"size":"Complaints_Count"})
    )

    # ----------------- aggregate cases (denominator) -----------------
    # process-wise unique Case_ID count + avg days
    if "Case_ID" not in cases_df.columns:
        cases_df = cases_df.copy()
        cases_df["Case_ID"] = pd.NA

    cases_grp = (
        cases_df
        .groupby(["Month","Portfolio_std","PortfolioKey","ProcessName","ProcessKey"], dropna=False)
        .agg(Unique_Cases=("Case_ID", pd.Series.nunique),
             Avg_NoOfDays=("NoOfDays","mean"))
        .reset_index()
    )

    # ----------------- join -----------------
    # Join on normalized keys (Month + PortfolioKey + ProcessKey)
    merged = comp_grp.merge(
        cases_grp,
        how="outer",
        left_on=["Month","PortfolioKey","ProcessKeyMapped"],
        right_on=["Month","PortfolioKey","ProcessKey"],
        suffixes=("_compl","_case")
    )

    # Resolve display columns
    merged["ProcessKeyFinal"] = merged["ProcessKeyMapped"].fillna(merged["ProcessKey"])
    merged["ProcessName"] = merged["ProcessName"].fillna(merged["ParentCaseType"])
    merged["Portfolio_std"] = merged["Portfolio_std_compl"].fillna(merged["Portfolio_std_case"])

    # Safe fills
    merged["Unique_Cases"] = merged["Unique_Cases"].fillna(0)
    merged["Complaints_Count"] = merged["Complaints_Count"].fillna(0)
    merged["Avg_NoOfDays"] = merged["Avg_NoOfDays"].fillna(pd.NA)

    # KPI: complaints per 1000 cases
    merged["Complaints_per_1000"] = merged.apply(
        lambda r: (r["Complaints_Count"] * 1000.0 / r["Unique_Cases"]) if r["Unique_Cases"] else 0.0,
        axis=1
    )

    # Final tidy columns
    out = merged[[
        "Month",
        "Portfolio_std",
        "ProcessName",
        "ParentCaseType",
        "Unique_Cases",
        "Avg_NoOfDays",
        "Complaints_Count",
        "Complaints_per_1000"
    ]].copy()

    # Order nicely
    out = out.sort_values(["Month","Portfolio_std","ProcessName"], kind="stable").reset_index(drop=True)
    return out
