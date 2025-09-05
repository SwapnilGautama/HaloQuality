# kpi/fpa.py
from __future__ import annotations
import pandas as pd
from typing import List

def _ensure_columns(df: pd.DataFrame, cols: List[str]) -> List[str]:
    return [c for c in cols if c in df.columns]

def fpa_summary(
    fpa_df: pd.DataFrame,
    group_by: List[str] | None = None,
    include_month: bool = True
) -> pd.DataFrame:
    """
    Reviewed, Fails, Fail_Rate by chosen dimensions.
    group_by can include: 'Portfolio_std','ProcessName','Scheme','TeamName','TeamManager','Location'
    """
    if fpa_df is None or fpa_df.empty:
        return pd.DataFrame()

    dims = group_by or ["Portfolio_std","ProcessName"]
    if include_month and "Month" in fpa_df.columns:
        dims = ["Month"] + dims

    dims = _ensure_columns(fpa_df, dims)
    if not dims:
        return pd.DataFrame()

    g = (
        fpa_df
        .groupby(dims, dropna=False)
        .agg(
            Reviewed=("Case_ID", "count"),
            Fails=("FailFlag", "sum"),
            Passes=("FailFlag", lambda s: (~s.astype(bool)).sum())
        )
        .reset_index()
    )
    g["Fail_Rate"] = (g["Fails"] / g["Reviewed"]).fillna(0.0)
    return g.sort_values(dims, kind="stable").reset_index(drop=True)

def fpa_fail_reasons(
    fpa_df: pd.DataFrame,
    group_by: List[str] | None = None,
    include_month: bool = True
) -> pd.DataFrame:
    """
    Count of failures by FPA_PrimaryTag (or by all tags if you prefer).
    """
    if fpa_df is None or fpa_df.empty:
        return pd.DataFrame()

    if "FailFlag" not in fpa_df.columns or "FPA_PrimaryTag" not in fpa_df.columns:
        return pd.DataFrame()

    df = fpa_df[fpa_df["FailFlag"] == True].copy()
    dims = group_by or ["Portfolio_std","ProcessName"]
    if include_month and "Month" in df.columns:
        dims = ["Month"] + dims

    dims = _ensure_columns(df, dims)
    if not dims:
        return pd.DataFrame()

    if "FPA_PrimaryTag" not in df.columns:
        return pd.DataFrame()

    out = (
        df
        .groupby(dims + ["FPA_PrimaryTag"], dropna=False)
        .size()
        .reset_index(name="Failures")
    ).sort_values(dims + ["Failures"], ascending=[True]*len(dims)+[False], kind="stable")

    # Optional share:
    total = out.groupby(dims)["Failures"].transform("sum")
    out["Share"] = (out["Failures"] / total).fillna(0.0)
    return out.reset_index(drop=True)
