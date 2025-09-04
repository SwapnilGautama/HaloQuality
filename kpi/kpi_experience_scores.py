
# kpi/kpi_experience_scores.py — KPI 4: Experience Scores (Agree/Strongly Agree)
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional

REQUIRED_SURVEY_COLS = ["month"]

DEFAULT_FIELDS = {
    "Clarity": "Clear_Information",
    "Timescale": "Timescale",
    "Handling": "Handle_Issue",
}

# Canonical labels (case-insensitive compare on stripped text)
AGREE_STRINGS = {
    "strongly agree", "agree"
}
SOMEWHAT_STRINGS = {
    "somewhat agree"
}

def _validate(df: pd.DataFrame, required_cols: List[str], name: str):
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name} missing required columns: {missing}")

def _to_text(s) -> str:
    if pd.isna(s):
        return ""
    return str(s).strip().lower()

def _agree_percent(series: pd.Series, include_somewhat: bool = False) -> Tuple[float, int]:
    vals = series.dropna().astype(str).map(_to_text)
    total = vals.shape[0]
    if total == 0:
        return (np.nan, 0)
    good = vals.isin(AGREE_STRINGS).sum()
    if include_somewhat:
        good += vals.isin(SOMEWHAT_STRINGS).sum()
    pct = (good / total) * 100.0
    return (round(pct, 1), total)

def experience_scores_by_group(
    survey_df: pd.DataFrame,
    month: str,
    group_by: List[str],
    fields: Optional[Dict[str, str]] = None,
    include_somewhat: bool = False,
    min_responses: int = 5
) -> pd.DataFrame:
    """
    Compute Agree/Strongly Agree percent for Clarity, Timescale, Handling by group.
    - fields: mapping like {'Clarity': 'Clear_Information', 'Timescale': 'Timescale', 'Handling': 'Handle_Issue'}
    - include_somewhat: if True, counts 'Somewhat agree' as agree too.
    - min_responses: minimum responses per metric per group to include row.
    Returns: group_by + ['Clarity_Agree_%','Timescale_Agree_%','Handling_Agree_%',
                         'Responses_Clarity','Responses_Timescale','Responses_Handling']
    """
    if not group_by:
        raise ValueError("group_by must contain at least one column name.")
    _validate(survey_df, REQUIRED_SURVEY_COLS, "survey_df")

    f = dict(DEFAULT_FIELDS)
    if fields:
        f.update(fields)

    s = survey_df[survey_df["month"] == month].copy()
    if s.empty:
        return pd.DataFrame(columns=group_by + [
            "Clarity_Agree_%","Timescale_Agree_%","Handling_Agree_%",
            "Responses_Clarity","Responses_Timescale","Responses_Handling"
        ])

    # Ensure group columns exist
    for col in group_by:
        if col not in s.columns:
            s[col] = np.nan

    # Ensure metric columns exist (create empty if missing)
    for _, colname in f.items():
        if colname not in s.columns:
            s[colname] = np.nan

    # Aggregate by group
    groups = s.groupby(group_by, dropna=False)
    rows = []
    for keys, df in groups:
        clarity_pct, clarity_n = _agree_percent(df[f["Clarity"]], include_somewhat)
        timescale_pct, timescale_n = _agree_percent(df[f["Timescale"]], include_somewhat)
        handling_pct, handling_n = _agree_percent(df[f["Handling"]], include_somewhat)

        # Enforce min_responses: require each metric to have at least min_responses
        if (clarity_n < min_responses) and (timescale_n < min_responses) and (handling_n < min_responses):
            # If all three are below threshold, skip this group entirely
            continue

        row = {}
        # unpack group keys
        if isinstance(keys, tuple):
            for i, col in enumerate(group_by):
                row[col] = keys[i]
        else:
            row[group_by[0]] = keys

        row.update({
            "Clarity_Agree_%": clarity_pct,
            "Timescale_Agree_%": timescale_pct,
            "Handling_Agree_%": handling_pct,
            "Responses_Clarity": int(clarity_n),
            "Responses_Timescale": int(timescale_n),
            "Responses_Handling": int(handling_n),
        })
        rows.append(row)

    out = pd.DataFrame(rows)
    if not out.empty:
        # Sort by Clarity by default (desc), then Timescale, then Handling
        sort_cols = [c for c in ["Clarity_Agree_%","Timescale_Agree_%","Handling_Agree_%"] if c in out.columns]
        out = out.sort_values(sort_cols, ascending=[False]*len(sort_cols), na_position="last").reset_index(drop=True)
    return out
