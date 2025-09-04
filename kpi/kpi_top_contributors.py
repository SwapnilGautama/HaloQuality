
# kpi/kpi_top_contributors.py — KPI 6: Top Contributors (level or MoM delta)
import pandas as pd
import numpy as np
from typing import List, Optional, Tuple, Dict

from kpi.kpi_complaints_per_1000 import complaints_per_1000
from kpi.kpi_nps import nps_by_group
from kpi.kpi_experience_scores import experience_scores_by_group

def _prev_month_str(month: str) -> str:
    p = pd.Period(month, freq="M")
    return (p - 1).strftime("%Y-%m")

def _ensure_cols(df: pd.DataFrame, cols: List[str]):
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan
    return df

def top_contributors(
    complaints_df: pd.DataFrame,
    cases_df: pd.DataFrame,
    survey_df: pd.DataFrame,
    month: str,
    group_by: List[str],
    focus: str = "complaints",           # 'complaints' | 'complaints_per_1000' | 'nps' | 'clarity' | 'timescale' | 'handling'
    mode: str = "level",                 # 'level' | 'delta'
    top_n: int = 10,
    include_somewhat: bool = False,
    min_responses: int = 5
) -> Tuple[pd.DataFrame, str]:
    """
    Returns top contributors for a metric, either current level or MoM change.
    For 'delta' mode, compares `month` vs previous month.
    Columns vary by focus; we try to include useful shares and counts.
    """
    focus = focus.lower().strip()
    mode = mode.lower().strip()
    prev_m = _prev_month_str(month)

    if focus not in {"complaints","complaints_per_1000","nps","clarity","timescale","handling"}:
        raise ValueError("Unsupported focus. Use one of: complaints, complaints_per_1000, nps, clarity, timescale, handling.")
    if mode not in {"level","delta"}:
        raise ValueError("Unsupported mode. Use 'level' or 'delta'.")

    # ---------------- Complaints ----------------
    if focus == "complaints":
        c_curr = complaints_df[complaints_df["month"] == month].copy()
        _ensure_cols(c_curr, group_by)
        grp_curr = c_curr.groupby(group_by, dropna=False).size().reset_index(name="Complaints")

        if mode == "level":
            total = grp_curr["Complaints"].sum()
            grp_curr["Share_%"] = (grp_curr["Complaints"] / total * 100.0) if total > 0 else np.nan
            out = grp_curr.sort_values("Complaints", ascending=False)
            return out.head(top_n).reset_index(drop=True), prev_m

        # delta mode
        c_prev = complaints_df[complaints_df["month"] == prev_m].copy()
        _ensure_cols(c_prev, group_by)
        grp_prev = c_prev.groupby(group_by, dropna=False).size().reset_index(name="Complaints_prev")

        merged = pd.merge(grp_curr, grp_prev, on=group_by, how="outer").fillna(0)
        merged["Delta_Complaints"] = merged["Complaints"] - merged["Complaints_prev"]
        total_change = merged["Delta_Complaints"].sum()
        merged["Share_of_change_%"] = (merged["Delta_Complaints"] / total_change * 100.0) if total_change != 0 else np.nan
        out = merged.sort_values("Delta_Complaints", ascending=False)
        return out.head(top_n).reset_index(drop=True), prev_m

    # ---------------- Complaints per 1000 ----------------
    if focus == "complaints_per_1000":
        curr = complaints_per_1000(complaints_df, cases_df, month, group_by, portfolio_filter=None)
        if mode == "level":
            # Provide shares by Complaints and Unique_Cases for context
            total_cases = curr["Unique_Cases"].sum()
            total_complaints = curr["Complaints"].sum()
            curr["Unique_Cases_Share_%"] = (curr["Unique_Cases"] / total_cases * 100.0) if total_cases > 0 else np.nan
            curr["Complaints_Share_%"] = (curr["Complaints"] / total_complaints * 100.0) if total_complaints > 0 else np.nan
            out = curr.sort_values("Complaints_per_1000", ascending=False)
            return out.head(top_n).reset_index(drop=True), prev_m

        # delta mode
        prev = complaints_per_1000(complaints_df, cases_df, prev_m, group_by, portfolio_filter=None)
        merged = pd.merge(curr, prev, on=group_by, how="outer", suffixes=("","_prev")).fillna(0)
        merged["Rate_Delta"] = merged["Complaints_per_1000"] - merged["Complaints_per_1000_prev"]
        # Weighted approximation of impact on overall rate (by current unique case mix)
        total_cases_curr = merged["Unique_Cases"].sum()
        merged["Unique_Cases_Share_%"] = (merged["Unique_Cases"] / total_cases_curr * 100.0) if total_cases_curr > 0 else np.nan
        merged["Weighted_Rate_Delta"] = merged["Rate_Delta"] * (merged["Unique_Cases"] / total_cases_curr) if total_cases_curr > 0 else np.nan
        out = merged.sort_values("Rate_Delta", ascending=False)
        return out.head(top_n).reset_index(drop=True), prev_m

    # ---------------- NPS ----------------
    if focus == "nps":
        curr = nps_by_group(survey_df, month, group_by, min_responses=min_responses)
        if mode == "level":
            total_resp = curr["Total_Responses"].sum() if "Total_Responses" in curr.columns else np.nan
            if "Total_Responses" in curr.columns and total_resp and total_resp > 0:
                curr["Responses_Share_%"] = (curr["Total_Responses"] / total_resp) * 100.0
            out = curr.sort_values("NPS", ascending=False)
            return out.head(top_n).reset_index(drop=True), prev_m
        prev = nps_by_group(survey_df, prev_m, group_by, min_responses=min_responses)
        merged = pd.merge(curr, prev, on=group_by, how="outer", suffixes=("","_prev")).fillna(0)
        merged["NPS_Delta"] = merged["NPS"] - merged["NPS_prev"]
        out = merged.sort_values("NPS_Delta", ascending=False)
        return out.head(top_n).reset_index(drop=True), prev_m

    # ---------------- Experience metrics ----------------
    metric_map = {
        "clarity": "Clarity_Agree_%",
        "timescale": "Timescale_Agree_%",
        "handling": "Handling_Agree_%"
    }
    metric_col = metric_map[focus]
    curr = experience_scores_by_group(survey_df, month, group_by, fields=None, include_somewhat=include_somewhat, min_responses=min_responses)
    if mode == "level":
        # Show response bases too
        resp_cols = [c for c in curr.columns if c.startswith("Responses_")]
        out = curr.sort_values(metric_col, ascending=False)
        return out.head(top_n).reset_index(drop=True), prev_m

    prev = experience_scores_by_group(survey_df, prev_m, group_by, fields=None, include_somewhat=include_somewhat, min_responses=min_responses)
    merged = pd.merge(curr, prev, on=group_by, how="outer", suffixes=("","_prev")).fillna(0)
    merged[f"{metric_col}_Delta"] = merged[metric_col] - merged[f"{metric_col}_prev"]
    out = merged.sort_values(f"{metric_col}_Delta", ascending=False)
    return out.head(top_n).reset_index(drop=True), prev_m
