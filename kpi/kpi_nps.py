# kpi/kpi_nps.py — KPI 3: NPS by group
import pandas as pd
import numpy as np
from typing import List

REQUIRED_SURVEY_COLS = ["month", "NPS"]

def _validate(df: pd.DataFrame, required_cols: List[str], name: str):
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name} missing required columns: {missing}")

def nps_by_group(
    survey_df: pd.DataFrame,
    month: str,
    group_by: List[str],
    min_responses: int = 5
) -> pd.DataFrame:
    """
    Compute NPS by group:
      - Promoters: NPS >= 9
      - Passives:  NPS in [7, 8]
      - Detractors: NPS <= 6
      - NPS = ((Promoters - Detractors) / TotalResponses) * 100
    Returns columns: group_by + ['Promoters','Passives','Detractors','Total_Responses','NPS']
    """
    if not group_by:
        raise ValueError("group_by must contain at least one column name.")
    _validate(survey_df, REQUIRED_SURVEY_COLS, "survey_df")

    s = survey_df[survey_df["month"] == month].copy()
    if s.empty:
        return pd.DataFrame(columns=group_by + ["Promoters","Passives","Detractors","Total_Responses","NPS"])

    # Ensure group_by columns exist
    for col in group_by:
        if col not in s.columns:
            s[col] = np.nan

    # Clean NPS to numeric
    s["NPS_num"] = pd.to_numeric(s["NPS"], errors="coerce")

    # Buckets
    s["Promoter"] = (s["NPS_num"] >= 9).astype(int)
    s["Passive"] = s["NPS_num"].between(7, 8, inclusive="both").astype(int)
    s["Detractor"] = (s["NPS_num"] <= 6).astype(int)
    s["Resp"] = s["NPS_num"].notna().astype(int)

    agg = (s
           .groupby(group_by, dropna=False)[["Promoter","Passive","Detractor","Resp"]]
           .sum()
           .reset_index()
           .rename(columns={
               "Promoter":"Promoters",
               "Passive":"Passives",
               "Detractor":"Detractors",
               "Resp":"Total_Responses"
           }))

    # Filter by minimum responses
    agg = agg[agg["Total_Responses"] >= int(min_responses)]

    # NPS
    agg["NPS"] = np.where(
        agg["Total_Responses"] > 0,
        ((agg["Promoters"] - agg["Detractors"]) / agg["Total_Responses"]) * 100.0,
        np.nan
    )
    agg["NPS"] = agg["NPS"].round(1)

    # Sort by NPS desc
    if "NPS" in agg.columns:
        agg = agg.sort_values("NPS", ascending=False, na_position="last")

    return agg.reset_index(drop=True)
