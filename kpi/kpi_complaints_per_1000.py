
import pandas as pd
import numpy as np
from typing import List, Optional

REQUIRED_COMPLAINTS_COLS = ["month", "Portfolio_std"]
REQUIRED_CASES_COLS = ["month", "Case ID", "Portfolio_std"]

def _validate(df: pd.DataFrame, required_cols: List[str], name: str):
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name} missing required columns: {missing}")

def complaints_per_1000(
    complaints_df: pd.DataFrame,
    cases_df: pd.DataFrame,
    month: str,
    group_by: List[str],
    portfolio_filter: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Compute complaints per 1,000 processed cases.
    - Numerator: count of complaints in given month (from complaints_df)
    - Denominator: unique Case ID count in given month (from cases_df)
    - Grouping: by columns in `group_by` (must exist in both dataframes where relevant)
    - Filters: optional portfolio filter (normalized names, e.g., 'London')
    Returns tidy DataFrame with columns: group_by + ['Complaints', 'Unique_Cases', 'Complaints_per_1000']
    """
    _validate(complaints_df, REQUIRED_COMPLAINTS_COLS, "complaints_df")
    _validate(cases_df, REQUIRED_CASES_COLS, "cases_df")

    # Filter by month
    c_df = complaints_df[complaints_df["month"] == month].copy()
    k_df = cases_df[cases_df["month"] == month].copy()

    if portfolio_filter is not None and len(portfolio_filter) > 0:
        if "Portfolio_std" in c_df.columns:
            c_df = c_df[c_df["Portfolio_std"].isin(portfolio_filter)]
        if "Portfolio_std" in k_df.columns:
            k_df = k_df[k_df["Portfolio_std"].isin(portfolio_filter)]

    # Ensure group_by columns exist in both DFs
    for col in group_by:
        if col not in c_df.columns:
            c_df[col] = np.nan
        if col not in k_df.columns:
            k_df[col] = np.nan

    # Numerator: complaints count
    num = (
        c_df
        .groupby(group_by, dropna=False)
        .size()
        .reset_index(name="Complaints")
    )

    # Denominator: unique Case IDs
    den = (
        k_df
        .dropna(subset=["Case ID"])
        .drop_duplicates(subset=["Case ID"] + group_by, keep="first")
        .groupby(group_by, dropna=False)
        .size()
        .reset_index(name="Unique_Cases")
    )

    # Align & compute KPI
    out = pd.merge(num, den, on=group_by, how="outer").fillna(0)
    out["Complaints"] = out["Complaints"].astype(int)
    out["Unique_Cases"] = out["Unique_Cases"].astype(int)
    out["Complaints_per_1000"] = np.where(out["Unique_Cases"] > 0, (out["Complaints"] / out["Unique_Cases"]) * 1000, np.nan)
    out = out.sort_values("Complaints_per_1000", ascending=False, na_position="last").reset_index(drop=True)

    # Round for readability (raw numbers remain numeric)
    out["Complaints_per_1000"] = out["Complaints_per_1000"].round(2)
    return out
