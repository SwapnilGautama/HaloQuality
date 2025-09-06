# questions/complaints_june_by_portfolio.py
from __future__ import annotations
import pandas as pd
import numpy as np

TITLE = "Complaint analysis — June 2025 (by portfolio)"

def _month_from_report_month_2025(series: pd.Series) -> pd.Series:
    # map 'Jun', 'June', etc. to '2025-06'
    s = series.astype(str).str.strip().str[:3].str.title()
    map_ = {'Jan':'2025-01','Feb':'2025-02','Mar':'2025-03','Apr':'2025-04','May':'2025-05',
            'Jun':'2025-06','Jul':'2025-07','Aug':'2025-08','Sep':'2025-09','Oct':'2025-10',
            'Nov':'2025-11','Dec':'2025-12'}
    return s.map(map_)

def _month_from_date(series: pd.Series) -> pd.Series:
    dt = pd.to_datetime(series, errors="coerce")
    return dt.dt.to_period("M").astype(str)

def _as_table(cases_jun: pd.DataFrame, complaints_jun: pd.DataFrame) -> pd.DataFrame:
    cases_by_port = cases_jun.groupby("Portfolio", dropna=False).agg(
        total_cases=("Case ID", "count")
    ).reset_index()

    comp_by_port = complaints_jun.groupby("Portfolio", dropna=False).agg(
        total_complaints=("Complaint case ID", "count")
    ).reset_index()

    out = cases_by_port.merge(comp_by_port, on="Portfolio", how="outer").fillna(0)
    out["total_cases"] = out["total_cases"].astype(int)
    out["total_complaints"] = out["total_complaints"].astype(int)
    out["complaints_per_1000"] = np.where(
        out["total_cases"] > 0,
        out["total_complaints"] * 1000 / out["total_cases"],
        np.nan,
    )

    # overall row
    overall = pd.DataFrame([{
        "Portfolio": "All",
        "total_cases": int(cases_jun["Case ID"].count()),
        "total_complaints": int(complaints_jun["Complaint case ID"].count()),
    }])
    overall["complaints_per_1000"] = np.where(
        overall["total_cases"] > 0,
        overall["total_complaints"] * 1000 / overall["total_cases"],
        np.nan,
    )

    out = pd.concat([overall, out], ignore_index=True)
    out["complaints_per_1000"] = out["complaints_per_1000"].round(2)
    return out

def run(store, params, user_text=None):
    """
    store: dict with .get("cases"), .get("complaints") as DataFrames
    params: may contain month/portfolio but we force June 2025 per request
    """
    cases = store.get("cases")
    complaints = store.get("complaints")
    if cases is None or complaints is None:
        return "No data loaded.", pd.DataFrame()

    # Build month fields
    cases = cases.copy()
    complaints = complaints.copy()

    # Cases: month from Create Date
    cases["month"] = _month_from_date(cases["Create Date"])

    # Complaints: month from Report Month, assume 2025
    complaints["month"] = _month_from_report_month_2025(complaints["Report Month"])

    # Filter to June 2025 only
    cases_jun = cases.loc[cases["month"] == "2025-06", ["Case ID", "Portfolio", "month"]].copy()
    complaints_jun = complaints.loc[complaints["month"] == "2025-06",
                                    ["Complaint case ID", "Portfolio", "month"]].copy()

    # Join only by Portfolio + month
    table = _as_table(cases_jun, complaints_jun).sort_values(
        ["Portfolio"], ascending=True, kind="stable"
    )

    title = "Complaints per 1,000 cases — June 2025 (by portfolio)"
    subtitle = "Join keys: Cases(Create Date→month) + Complaints(Report Month→2025-06) + Portfolio"
    return (title, subtitle), table
