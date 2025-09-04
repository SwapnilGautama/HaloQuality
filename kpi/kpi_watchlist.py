
# kpi/kpi_watchlist.py — KPI 9: Site/Portfolio Watchlist with alert rules
import pandas as pd
import numpy as np
from typing import List, Optional, Tuple, Dict

from kpi.kpi_mom import mom_overview

def _zscore(series: pd.Series) -> pd.Series:
    s = series.astype(float)
    mu = s.mean(skipna=True)
    sd = s.std(skipna=True, ddof=0)
    if sd and sd > 0:
        return (s - mu) / sd
    return pd.Series([np.nan]*len(s), index=s.index)

def watchlist_alerts(
    complaints_df: pd.DataFrame,
    cases_df: pd.DataFrame,
    survey_df: pd.DataFrame,
    month: str,
    group_by: List[str],
    include_somewhat: bool = False,
    min_responses: int = 5,
    # thresholds
    rate_level_thresh: float = 200.0,    # Complaints per 1k threshold (level)
    rate_delta_thresh: float = 20.0,     # Complaints per 1k MoM spike
    nps_drop_thresh: float = 10.0,       # NPS drop threshold (points)
    clarity_drop_thresh: float = 5.0,    # pp drop
    timescale_drop_thresh: float = 5.0,  # pp drop
    handling_drop_thresh: float = 5.0,   # pp drop
    z_thresh: float = 2.0                # z-score threshold for outliers
) -> Tuple[pd.DataFrame, str, Dict[str, float]]:
    """
    Build a watchlist with alerts for deteriorations.
    Uses KPI 5 (mom_overview) and then applies rule-based flags.
    Returns: (DataFrame, prev_month, thresholds_dict)
    """
    base, prev_m = mom_overview(
        complaints_df=complaints_df,
        cases_df=cases_df,
        survey_df=survey_df,
        month=month,
        group_by=group_by,
        include_somewhat=include_somewhat,
        min_responses=min_responses,
        fields_map=None
    )

    if base.empty:
        return base, prev_m, {
            "rate_level_thresh": rate_level_thresh,
            "rate_delta_thresh": rate_delta_thresh,
            "nps_drop_thresh": nps_drop_thresh,
            "clarity_drop_thresh": clarity_drop_thresh,
            "timescale_drop_thresh": timescale_drop_thresh,
            "handling_drop_thresh": handling_drop_thresh,
            "z_thresh": z_thresh
        }

    # Compute z-scores across groups
    if "Complaints_per_1000" in base.columns:
        base["Rate_Z"] = _zscore(base["Complaints_per_1000"])
    else:
        base["Rate_Z"] = np.nan

    if "Complaints_per_1000_delta" in base.columns:
        base["RateDelta_Z"] = _zscore(base["Complaints_per_1000_delta"])
    else:
        base["RateDelta_Z"] = np.nan

    alerts_col = []
    severity = []

    for _, row in base.iterrows():
        flags = []

        rate = row.get("Complaints_per_1000", np.nan)
        rate_prev = row.get("Complaints_per_1000_prev", np.nan)
        rate_delta = row.get("Complaints_per_1000_delta", np.nan)
        nps_delta = row.get("NPS_delta", np.nan)
        clarity_delta = row.get("Clarity_Agree_%_delta", np.nan)
        timescale_delta = row.get("Timescale_Agree_%_delta", np.nan)
        handling_delta = row.get("Handling_Agree_%_delta", np.nan)

        # Level and delta thresholds
        if pd.notna(rate) and rate >= rate_level_thresh:
            flags.append(f"High complaints/1k (≥ {rate_level_thresh})")
        if pd.notna(rate_delta) and rate_delta >= rate_delta_thresh:
            flags.append(f"Spike in complaints/1k (+{rate_delta_thresh}pp)")

        # z-score outliers
        if pd.notna(row.get("Rate_Z")) and abs(row["Rate_Z"]) >= z_thresh:
            flags.append(f"Rate outlier (|z| ≥ {z_thresh})")
        if pd.notna(row.get("RateDelta_Z")) and abs(row["RateDelta_Z"]) >= z_thresh:
            flags.append(f"Delta outlier (|z| ≥ {z_thresh})")

        # NPS drops (negative delta)
        if pd.notna(nps_delta) and nps_delta <= -abs(nps_drop_thresh):
            flags.append(f"NPS drop (≤ -{abs(nps_drop_thresh)})")

        # Experience drops
        if pd.notna(clarity_delta) and clarity_delta <= -abs(clarity_drop_thresh):
            flags.append(f"Clarity drop (≤ -{abs(clarity_drop_thresh)}pp)")
        if pd.notna(timescale_delta) and timescale_delta <= -abs(timescale_drop_thresh):
            flags.append(f"Timescale drop (≤ -{abs(timescale_drop_thresh)}pp)")
        if pd.notna(handling_delta) and handling_delta <= -abs(handling_drop_thresh):
            flags.append(f"Handling drop (≤ -{abs(handling_drop_thresh)}pp)")

        # Severity score (weighted)
        score = 0
        for f in flags:
            if "High complaints/1k" in f: score += 3
            elif "Spike in complaints/1k" in f: score += 3
            elif "Rate outlier" in f or "Delta outlier" in f: score += 2
            elif "NPS drop" in f: score += 2
            else: score += 1  # any experience drop

        # Status bucket
        if score >= 5 or any(("High complaints/1k" in f or "Spike in complaints/1k" in f) for f in flags):
            status = "Red"
        elif score >= 2:
            status = "Amber"
        else:
            status = "Green"

        alerts_col.append(", ".join(flags) if flags else "")
        severity.append((score, status))

    base["Alerts"] = alerts_col
    base["Severity_Score"] = [s for s, _ in severity]
    base["Status"] = [st for _, st in severity]

    # Order columns (keep important first)
    leading = list(group_by) + [
        "Status","Severity_Score","Alerts",
        "Complaints_per_1000","Complaints_per_1000_prev","Complaints_per_1000_delta","Rate_Z","RateDelta_Z",
        "Complaints","Complaints_prev","Complaints_delta","Unique_Cases","Unique_Cases_prev","Unique_Cases_delta",
        "NPS","NPS_prev","NPS_delta",
        "Clarity_Agree_%","Clarity_Agree_%_prev","Clarity_Agree_%_delta",
        "Timescale_Agree_%","Timescale_Agree_%_prev","Timescale_Agree_%_delta",
        "Handling_Agree_%","Handling_Agree_%_prev","Handling_Agree_%_delta"
    ]
    rest = [c for c in base.columns if c not in leading]
    out = base[leading + rest]

    # Sort by severity then rate delta
    out = out.sort_values(["Status","Severity_Score","Complaints_per_1000_delta"],
                          ascending=[True, False, False], na_position="last")
    # Make Status ordered as Red > Amber > Green
    status_order = {"Red": 0, "Amber": 1, "Green": 2}
    out["__status_order__"] = out["Status"].map(status_order).fillna(3)
    out = out.sort_values(["__status_order__","Severity_Score","Complaints_per_1000_delta"],
                          ascending=[True, False, False], na_position="last").drop(columns="__status_order__")

    thresholds = {
        "rate_level_thresh": float(rate_level_thresh),
        "rate_delta_thresh": float(rate_delta_thresh),
        "nps_drop_thresh": float(nps_drop_thresh),
        "clarity_drop_thresh": float(clarity_drop_thresh),
        "timescale_drop_thresh": float(timescale_drop_thresh),
        "handling_drop_thresh": float(handling_drop_thresh),
        "z_thresh": float(z_thresh)
    }
    return out.reset_index(drop=True), prev_m, thresholds
