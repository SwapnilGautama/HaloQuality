# kpi/complaints_rca.py
from __future__ import annotations
import pandas as pd
from typing import List

def _keep(df: pd.DataFrame, cols: List[str]) -> List[str]:
    return [c for c in cols if c in df.columns]

def complaints_rca_summary(
    complaints_df: pd.DataFrame,
    group_by: List[str] | None = None,
    include_month: bool = True,
    use_rca2: bool = False
) -> pd.DataFrame:
    """
    Count of complaints by RCA1 (and optionally RCA2) along chosen dimensions.

    group_by can include:
      ['Portfolio_std','ProcessName','Parent_Case_Type','Scheme','TeamName','Location']
    """
    if complaints_df is None or complaints_df.empty:
        return pd.DataFrame()

    dims = group_by or ["Portfolio_std","ProcessName","Parent_Case_Type"]
    if include_month and "Month" in complaints_df.columns:
        dims = ["Month"] + dims
    dims = _keep(complaints_df, dims)

    label_cols = ["RCA1"]
    if use_rca2:
        label_cols.append("RCA2")

    dims = dims + _keep(complaints_df, label_cols)
    if not dims:
        return pd.DataFrame()

    res = (
        complaints_df
        .groupby(dims, dropna=False)
        .size()
        .reset_index(name="Complaints")
        .sort_values(dims, kind="stable")
        .reset_index(drop=True)
    )

    # Optionally compute within-group shares
    base = res.groupby(_keep(complaints_df, (["Month"] if include_month else []) + (group_by or [])))["Complaints"] \
              .transform("sum")
    res["Share"] = (res["Complaints"] / base).fillna(0.0)
    return res
