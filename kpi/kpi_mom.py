
# kpi/kpi_mom.py — KPI 5: Month-over-Month Overview (Complaints, /1000, NPS, Experience)
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple

from kpi.kpi_complaints_per_1000 import complaints_per_1000
from kpi.kpi_nps import nps_by_group
from kpi.kpi_experience_scores import experience_scores_by_group

def _prev_month_str(month: str) -> str:
    """Return previous month as 'YYYY-MM'."""
    p = pd.Period(month, freq="M")
    return (p - 1).strftime("%Y-%m")

def _merge_left(base: pd.DataFrame, other: pd.DataFrame, on: List[str], suffix: str) -> pd.DataFrame:
    other_ren = other.copy()
    for c in other_ren.columns:
        if c not in on:
            other_ren.rename(columns={c: f"{c}{suffix}"}, inplace=True)
    return base.merge(other_ren, on=on, how="outer")

def mom_overview(
    complaints_df: pd.DataFrame,
    cases_df: pd.DataFrame,
    survey_df: pd.DataFrame,
    month: str,
    group_by: List[str],
    include_somewhat: bool = False,
    min_responses: int = 5,
    fields_map: Optional[Dict[str, str]] = None
) -> Tuple[pd.DataFrame, str]:
    """
    Build a combined Month-over-Month view across key KPIs.
    Returns (DataFrame, prev_month_string).
    Columns include:
      - group_by ...
      - Complaints, Unique_Cases, Complaints_per_1000
      - Complaints_prev, Unique_Cases_prev, Complaints_per_1000_prev
      - Complaints_per_1000_delta (curr - prev)
      - NPS, NPS_prev, NPS_delta
      - Clarity_Agree_%, Timescale_Agree_%, Handling_Agree_%
      - *_prev, and deltas for each
      - Response counts for NPS/Experience (Total_Responses{_prev}, Responses_*)
    """
    prev_m = _prev_month_str(month)

    # --- Complaints & /1000 ---
    comp_curr = complaints_per_1000(complaints_df, cases_df, month, group_by, portfolio_filter=None)
    comp_prev = complaints_per_1000(complaints_df, cases_df, prev_m, group_by, portfolio_filter=None)

    base = comp_curr.copy()
    base = _merge_left(base, comp_prev, on=group_by, suffix="_prev")

    # deltas
    if "Complaints_per_1000" in base.columns and "Complaints_per_1000_prev" in base.columns:
        base["Complaints_per_1000_delta"] = base["Complaints_per_1000"] - base["Complaints_per_1000_prev"]
    if "Complaints" in base.columns and "Complaints_prev" in base.columns:
        base["Complaints_delta"] = base["Complaints"] - base["Complaints_prev"]
    if "Unique_Cases" in base.columns and "Unique_Cases_prev" in base.columns:
        base["Unique_Cases_delta"] = base["Unique_Cases"] - base["Unique_Cases_prev"]

    # --- NPS ---
    if survey_df is not None and not survey_df.empty:
        nps_curr = nps_by_group(survey_df, month, group_by, min_responses=min_responses)
        nps_prev = nps_by_group(survey_df, prev_m, group_by, min_responses=min_responses)

        base = _merge_left(base, nps_curr, on=group_by, suffix="")
        base = _merge_left(base, nps_prev, on=group_by, suffix="_prev")

        if "NPS" in base.columns and "NPS_prev" in base.columns:
            base["NPS_delta"] = base["NPS"] - base["NPS_prev"]

    # --- Experience Scores ---
    if survey_df is not None and not survey_df.empty:
        exp_curr = experience_scores_by_group(
            survey_df=survey_df, month=month, group_by=group_by,
            fields=fields_map, include_somewhat=include_somewhat, min_responses=min_responses
        )
        exp_prev = experience_scores_by_group(
            survey_df=survey_df, month=prev_m, group_by=group_by,
            fields=fields_map, include_somewhat=include_somewhat, min_responses=min_responses
        )

        base = _merge_left(base, exp_curr, on=group_by, suffix="")
        base = _merge_left(base, exp_prev, on=group_by, suffix="_prev")

        # deltas
        for metric in ["Clarity_Agree_%","Timescale_Agree_%","Handling_Agree_%"]:
            if metric in base.columns and f"{metric}_prev" in base.columns:
                base[f"{metric}_delta"] = base[metric] - base[f"{metric}_prev"]

    # Order columns nicely
    leading = list(group_by)
    cols = set(base.columns) - set(leading)
    # a reasonable order
    prefer = [
        "Complaints","Unique_Cases","Complaints_per_1000",
        "Complaints_prev","Unique_Cases_prev","Complaints_per_1000_prev",
        "Complaints_per_1000_delta","Complaints_delta","Unique_Cases_delta",
        "NPS","Total_Responses","NPS_prev","Total_Responses_prev","NPS_delta",
        "Clarity_Agree_%","Timescale_Agree_%","Handling_Agree_%",
        "Responses_Clarity","Responses_Timescale","Responses_Handling",
        "Clarity_Agree_%_prev","Timescale_Agree_%_prev","Handling_Agree_%_prev",
        "Responses_Clarity_prev","Responses_Timescale_prev","Responses_Handling_prev",
        "Clarity_Agree_%_delta","Timescale_Agree_%_delta","Handling_Agree_%_delta",
    ]
    ordered = leading + [c for c in prefer if c in base.columns] + sorted(list(cols - set(prefer)))
    base = base.reindex(columns=ordered)

    # Sorting: highest deterioration in complaints_per_1000 on top
    sort_col = "Complaints_per_1000_delta" if "Complaints_per_1000_delta" in base.columns else None
    if sort_col:
        base = base.sort_values(sort_col, ascending=False, na_position="last")

    return base.reset_index(drop=True), prev_m
