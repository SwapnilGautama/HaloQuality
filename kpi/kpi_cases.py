# kpi/kpi_cases.py
from __future__ import annotations
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


def _apply_date_window(
    df: pd.DataFrame,
    month_from: Optional[str] = None,  # "YYYY-MM"
    month_to: Optional[str] = None,    # "YYYY-MM"
) -> pd.DataFrame:
    """Filter by inclusive month_ym window."""
    if "month_ym" not in df.columns:
        return df
    m = df["month_ym"]
    if month_from:
        df = df[m >= month_from]
    if month_to:
        df = df[m <= month_to]
    return df


def cases_summary(
    cases_df: pd.DataFrame,
    group_by: Iterable[str] = ("Portfolio_std", "month_ym"),
    month_from: Optional[str] = None,
    month_to: Optional[str] = None,
    filters: Optional[Dict[str, Iterable[str]]] = None,
) -> pd.DataFrame:
    """
    Unique Case_ID count sliced by arbitrary dimensions, with average NumDays if available.

    Returns columns:
      - group_by ...
      - Unique_Cases
      - Avg_Days (if NumDays present)
    """
    if cases_df.empty:
        return pd.DataFrame()

    df = cases_df.copy()

    # Date window on Create_Date month (via month_ym)
    df = _apply_date_window(df, month_from, month_to)

    # Apply dimension filters (exact-match semantics)
    if filters:
        for col, vals in filters.items():
            if col in df.columns and vals:
                df = df[df[col].isin(list(vals))]

    # Unique cases at the requested grain
    gb_cols = [c for c in group_by if c in df.columns]
    if not gb_cols:
        gb_cols = ["month_ym"]

    # Deduplicate at group level (Case_ID unique)
    # We already de-duped globally in loader, but if a case spans dims we still count once per grain.
    dedup_cols = ["Case_ID"] + gb_cols
    df = df.dropna(subset=["Case_ID"]).drop_duplicates(subset=dedup_cols, keep="first")

    agg = {"Case_ID": "nunique"}
    if "NumDays" in df.columns:
        agg["NumDays"] = "mean"

    out = df.groupby(gb_cols, dropna=False).agg(agg).reset_index()
    out = out.rename(columns={"Case_ID": "Unique_Cases", "NumDays": "Avg_Days"})
    if "Avg_Days" in out.columns:
        out["Avg_Days"] = out["Avg_Days"].round(2)

    # If both month_ym and month_mmm exist, include a friendly month label
    if "month_ym" in out.columns and "month_mmm" in cases_df.columns and "month_mmm" not in out.columns:
        # merge to attach month_mmm
        mm = cases_df[["month_ym", "month_mmm"]].drop_duplicates()
        out = out.merge(mm, on="month_ym", how="left")

    return out.sort_values(gb_cols)


def cases_pivot_mom(
    cases_df: pd.DataFrame,
    dimension: str = "Portfolio_std",
    value: str = "Unique_Cases",
    month_from: Optional[str] = None,
    month_to: Optional[str] = None,
    filters: Optional[Dict[str, Iterable[str]]] = None,
) -> pd.DataFrame:
    """
    Return a Month x Dimension pivot (MoM) for quick tables by portfolio/location/etc.
    """
    base = cases_summary(
        cases_df,
        group_by=(dimension, "month_ym"),
        month_from=month_from,
        month_to=month_to,
        filters=filters,
    )
    if base.empty:
        return base

    # If value not present (e.g., asking for Avg_Days), compute both & then pick
    if value not in base.columns:
        # recompute ensuring Avg_Days is available
        value = "Unique_Cases"

    pivot = (
        base
        .pivot_table(index=dimension, columns="month_ym", values=value, aggfunc="sum", fill_value=0)
        .sort_index()
    )

    # Order columns chronologically
    pivot = pivot.reindex(sorted(pivot.columns), axis=1)

    # Optional total
    pivot["Total"] = pivot.sum(axis=1)
    return pivot


def cases_mom_delta(
    cases_df: pd.DataFrame,
    group_by: Iterable[str] = ("Portfolio_std",),
    month_from: Optional[str] = None,
    month_to: Optional[str] = None,
    filters: Optional[Dict[str, Iterable[str]]] = None,
) -> pd.DataFrame:
    """
    Month-on-Month delta table for unique cases and average days.
    """
    base = cases_summary(
        cases_df,
        group_by=[*group_by, "month_ym"],
        month_from=month_from,
        month_to=month_to,
        filters=filters,
    )
    if base.empty:
        return base

    # Calculate deltas vs previous month within each group
    sort_cols = [*group_by, "month_ym"]
    base = base.sort_values(sort_cols)
    base["Unique_Cases_Delta"] = base.groupby(list(group_by))["Unique_Cases"].diff()
    if "Avg_Days" in base.columns:
        base["Avg_Days_Delta"] = base.groupby(list(group_by))["Avg_Days"].diff()

    return base
