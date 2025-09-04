
# kpi/kpi_sla_breach.py — KPI 10: SLA Breach Rate
import re
from typing import List, Optional, Tuple, Dict
import numpy as np
import pandas as pd

REQUIRED_BASE_COLS = ["month"]

_START_GUESS = [
    "Start Date","Case Created Date","Created Date","Opened Date","Opened On",
    "Report_Date","Received Date","Date Received","Created","Start"
]
_END_GUESS = [
    "Closed Date","Case Closed Date","Completed Date","Resolved Date","Date Closed",
    "End Date","Closed","Completion Date","Completed","Resolved"
]

def _validate(df: pd.DataFrame, cols: List[str], name: str):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name} missing required columns: {missing}")

def _guess_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    lowcols = {c.lower(): c for c in df.columns}
    for cand in candidates:
        lc = cand.lower()
        if lc in lowcols:
            return lowcols[lc]
    # fuzzy contains
    for c in df.columns:
        cl = c.lower()
        for cand in candidates:
            if cand.lower().replace(" ", "") in cl.replace(" ", ""):
                return c
    return None

def _parse_target_to_hours(target: str) -> float:
    """
    Parse targets like '48h', '24 H', '5d', '1.5d' into hours (float).
    Default: if pure number (e.g., '48'), interpret as hours.
    """
    if target is None:
        return 120.0  # default 5d
    s = str(target).strip().lower()
    m = re.match(r"^\s*(\d+(?:\.\d+)?)\s*([hd])?\s*$", s)
    if not m:
        raise ValueError("Invalid target format. Use like '48h' or '5d'.")
    val = float(m.group(1))
    unit = m.group(2) or "h"
    if unit == "h":
        return val
    else:
        return val * 24.0

def _business_days_between(start_series: pd.Series, end_series: pd.Series) -> np.ndarray:
    """
    Compute business days between start (inclusive) and end (exclusive).
    Rows with invalid/missing dates return NaN.
    """
    sdt = pd.to_datetime(start_series, errors="coerce", dayfirst=True)
    edt = pd.to_datetime(end_series, errors="coerce", dayfirst=True)
    valid = sdt.notna() & edt.notna()
    out = np.full(shape=len(start_series), fill_value=np.nan, dtype=float)
    if valid.any():
        sdates = sdt[valid].dt.date.values
        edates = edt[valid].dt.date.values
        # numpy busday_count excludes end date by default
        bdays = np.busday_count(sdates, edates).astype(float)
        out[valid.values] = bdays
    return out

def sla_breach_rate(
    df: pd.DataFrame,
    month: str,
    group_by: List[str],
    start_col: Optional[str] = None,
    end_col: Optional[str] = None,
    target: str = "5d",
    mode: str = "business_days",  # 'calendar_hours' | 'calendar_days' | 'business_days'
    min_cases: int = 5
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    Compute SLA breach rate for the selected month.
    - mode='calendar_hours' uses hour differences; compare to target (in hours).
    - mode='calendar_days' uses day differences; compare to target (converted to days).
    - mode='business_days' uses numpy.busday_count between dates.
    Returns (DataFrame, used_columns)
    Columns: group_by + ['Measured_Cases','Breached','Within_SLA','Breach_Rate_%',
                         'Avg_Duration_hours','P90_Duration_hours','Avg_Business_Days']
    """
    _validate(df, REQUIRED_BASE_COLS, "input_df")
    if not group_by:
        raise ValueError("group_by must contain at least one column.")

    d = df[df["month"] == month].copy()
    if d.empty:
        return pd.DataFrame(columns=group_by + ["Measured_Cases","Breached","Within_SLA","Breach_Rate_%",
                                                "Avg_Duration_hours","P90_Duration_hours","Avg_Business_Days"]), {}

    # Ensure group columns
    for g in group_by:
        if g not in d.columns:
            d[g] = np.nan

    # Guess columns if needed
    s_col = start_col or _guess_col(d, _START_GUESS)
    e_col = end_col or _guess_col(d, _END_GUESS)
    if not s_col or not e_col:
        raise ValueError("Could not find start/end datetime columns. Pass start_col and end_col explicitly.")

    # Parse datetimes
    s = pd.to_datetime(d[s_col], errors="coerce", dayfirst=True)
    e = pd.to_datetime(d[e_col], errors="coerce", dayfirst=True)

    # Valid rows
    valid = s.notna() & e.notna() & (e >= s)
    d = d.loc[valid].copy()
    if d.empty:
        return pd.DataFrame(columns=group_by + ["Measured_Cases","Breached","Within_SLA","Breach_Rate_%",
                                                "Avg_Duration_hours","P90_Duration_hours","Avg_Business_Days"]), {"start_col": s_col, "end_col": e_col}

    # Durations
    dur_hours = (e - s).dt.total_seconds() / 3600.0
    dur_days = dur_hours / 24.0
    bdays = _business_days_between(s, e)

    # Determine breach by mode
    target_h = _parse_target_to_hours(target)
    if mode == "calendar_hours":
        breached = dur_hours > target_h
    elif mode == "calendar_days":
        breached = dur_days > (target_h / 24.0)
    elif mode == "business_days":
        # If business days are NaN (shouldn't for valid rows), treat as NaN -> not counted
        breached = pd.Series(bdays, index=d.index) > (target_h / 24.0)
    else:
        raise ValueError("mode must be one of: calendar_hours, calendar_days, business_days")

    d["__breached__"] = breached.astype(bool)
    d["__dur_hours__"] = dur_hours.values
    d["__bdays__"] = bdays

    # Aggregate
    grp = d.groupby(group_by, dropna=False)
    out = grp.agg(
        Measured_Cases=("__breached__", "count"),
        Breached=("__breached__", "sum"),
        Avg_Duration_hours=("**dur_hours__", "mean") if "__dur_hours__" in d.columns else ("__breached__", "count")
    )
    # fix mean due to agg alias attempt (Py>=2.0 sometimes picky)
    out = grp["__dur_hours__"].mean().to_frame("Avg_Duration_hours").join(
          grp["__breached__"].agg(Measured_Cases="count", Breached="sum")
    )

    # business-day averages if available
    out["Avg_Business_Days"] = grp["__bdays__"].mean()

    # P90
    out["P90_Duration_hours"] = grp["__dur_hours__"].quantile(0.90)

    # Rates
    out["Within_SLA"] = out["Measured_Cases"] - out["Breached"]
    out["Breach_Rate_%"] = np.where(out["Measured_Cases"] > 0,
                                    (out["Breached"] / out["Measured_Cases"]) * 100.0,
                                    np.nan)
    # Filter by minimum base
    out = out[out["Measured_Cases"] >= int(min_cases)]

    # Round
    out["Avg_Duration_hours"] = out["Avg_Duration_hours"].round(1)
    out["P90_Duration_hours"] = out["P90_Duration_hours"].round(1)
    out["Avg_Business_Days"] = out["Avg_Business_Days"].round(2)
    out["Breach_Rate_%"] = out["Breach_Rate_%"].round(1)

    # Order
    out = out.reset_index()
    out = out.sort_values(["Breach_Rate_%","Avg_Duration_hours"], ascending=[False, False], na_position="last")

    used = {"start_col": s_col, "end_col": e_col, "mode": mode, "target": str(target)}
    return out.reset_index(drop=True), used
