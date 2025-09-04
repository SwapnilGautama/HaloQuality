# app.py — Halo Quality KPI API (v0.3.0) with Reason Mix% KPI

import os
import glob
from pathlib import Path
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, Query, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np

# -------------------- Config paths --------------------
DEFAULT_DATA_DIR = os.getenv("HALO_DATA_DIR", "/mnt/data")

# Complaints & Survey can stay as single files (you can override or extend later)
DEFAULT_COMPLAINTS_XLSX = os.getenv(
    "COMPLAINTS_XLSX",
    f"{DEFAULT_DATA_DIR}/Complaints June'25.xlsx"
)
DEFAULT_SURVEY_XLSX = os.getenv(
    "SURVEY_XLSX",
    f"{DEFAULT_DATA_DIR}/Overall raw data - June 2025.xlsx"
)

# Cases (denominator) supports a DIRECTORY of monthly files
DEFAULT_CASES_DIR = os.getenv("CASES_DIR", f"{DEFAULT_DATA_DIR}/cases")
DEFAULT_CASES_XLSX = os.getenv(
    "CASES_XLSX",
    f"{DEFAULT_DATA_DIR}/June'25 data with unique identifier.xlsx"
)

# -------------------- Helpers --------------------
def _to_month_str(dt) -> Optional[str]:
    """Parse anything datetime-like to 'YYYY-MM' or return None."""
    try:
        if pd.isna(dt):
            return None
        if isinstance(dt, str):
            if not dt.strip():
                return None
            d = pd.to_datetime(dt, errors="coerce", dayfirst=True)
        else:
            d = pd.to_datetime(dt, errors="coerce")
        if pd.isna(d):
            return None
        return d.to_period("M").strftime("%Y-%m")
    except Exception:
        return None

def _std_portfolio(s: Any) -> Any:
    """Normalize portfolio labels to a consistent title-cased form."""
    if pd.isna(s):
        return s
    t = str(s).strip()
    low = t.lower()
    low = low.replace("leatherhead - baes", "baes leatherhead").replace("baes-leatherhead", "baes leatherhead")
    low = low.replace("north west", "northwest")
    return low.title()

def _collect_case_files(cases_input: Optional[str]) -> List[str]:
    """
    Resolve where to load case files from:
    1) If cases_input is a directory -> use all *.xlsx within.
    2) If cases_input is an existing file -> use it.
    3) Else, try DEFAULT_CASES_DIR (directory).
    4) Else, try DEFAULT_CASES_XLSX (single file).
    """
    candidates: List[str] = []

    if cases_input:
        p = Path(cases_input)
        if p.is_dir():
            candidates = sorted(glob.glob(str(p / "*.xlsx")))
        elif p.exists():
            candidates = [str(p)]

    if not candidates:
        p_dir = Path(DEFAULT_CASES_DIR)
        if p_dir.is_dir():
            candidates = sorted(glob.glob(str(p_dir / "*.xlsx")))

    if not candidates:
        p_file = Path(DEFAULT_CASES_XLSX)
        if p_file.exists():
            candidates = [str(p_file)]

    if not candidates:
        raise HTTPException(
            status_code=400,
            detail="No case files found. Put monthly .xlsx under data/cases/ "
                   "or set CASES_DIR/CASES_XLSX/reload(cases_path=...)."
        )
    return candidates

# -------------------- Data Store --------------------
class DataStore:
    def __init__(self):
        self.reload()

    def reload(
        self,
        complaints_path: str = DEFAULT_COMPLAINTS_XLSX,
        cases_path: Optional[str] = None,   # may be a directory OR a single file
        survey_path: str = DEFAULT_SURVEY_XLSX
    ):
        # ----- Complaints (single file for now) -----
        try:
            comp = pd.read_excel(complaints_path, sheet_name=0)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to load complaints file: {e}")
        date_col = "Date Complaint Received - DD/MM/YY"
        if date_col not in comp.columns:
            raise HTTPException(status_code=400, detail=f"Complaints file missing '{date_col}'")
        comp["month"] = comp[date_col].apply(_to_month_str)
        comp["Portfolio_std"] = comp.get("Portfolio", np.nan).apply(_std_portfolio) if "Portfolio" in comp.columns else np.nan
        self.complaints = comp

        # ----- Cases (can be MANY files) -----
        frames: List[pd.DataFrame] = []
        for fpath in _collect_case_files(cases_path):
            try:
                df = pd.read_excel(fpath, sheet_name=0)
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Failed to load cases file '{Path(fpath).name}': {e}")
            for c in ["Report_Date", "Case ID"]:
                if c not in df.columns:
                    raise HTTPException(status_code=400, detail=f"Cases file missing '{c}': {Path(fpath).name}")
            df["month"] = df["Report_Date"].apply(_to_month_str)
            df["Portfolio_std"] = df.get("Portfolio", np.nan).apply(_std_portfolio) if "Portfolio" in df.columns else np.nan
            frames.append(df)

        cases = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["Report_Date", "Case ID", "month", "Portfolio_std"])
        # Drop empties and dedupe by month+Case ID so a case is counted once per month
        cases = cases.dropna(subset=["Case ID"])
        cases["Case ID"] = cases["Case ID"].astype(str)
        cases = cases.drop_duplicates(subset=["month", "Case ID"], keep="first")
        self.cases = cases

        # ----- Survey (optional) -----
        try:
            survey = pd.read_excel(survey_path, sheet_name=0)
            survey["Portfolio_std"] = survey.get("Portfolio", np.nan).apply(_std_portfolio) if "Portfolio" in survey.columns else np.nan
            survey["month"] = survey.get("Month_received", np.nan).apply(_to_month_str) if "Month_received" in survey.columns else np.nan
            self.survey = survey
        except Exception:
            self.survey = pd.DataFrame()

store = DataStore()

app = FastAPI(title="Halo Quality KPI API", version="0.3.0")

# -------------------- Models --------------------
class ComplaintsPer1000Response(BaseModel):
    month: str
    group_by: List[str]
    rows: List[Dict[str, Any]]

class ReasonMixResponse(BaseModel):
    month: str
    group_by: List[str]
    reason_field: str
    rows: List[Dict[str, Any]]

# -------------------- Routes --------------------
@app.get("/health")
def health():
    return {
        "status": "ok",
        "complaints_rows": int(len(store.complaints)),
        "cases_rows": int(len(store.cases)),
        "cases_months": sorted([m for m in store.cases["month"].dropna().unique().tolist()]) if not store.cases.empty else []
    }

@app.post("/reload")
def reload(
    complaints_path: Optional[str] = None,
    cases_path: Optional[str] = None,   # pass a directory path here to load all monthly files
    survey_path: Optional[str] = None
):
    store.reload(
        complaints_path or DEFAULT_COMPLAINTS_XLSX,
        cases_path or None,  # None triggers directory discovery via DEFAULT_CASES_DIR/DEFAULT_CASES_XLSX
        survey_path or DEFAULT_SURVEY_XLSX
    )
    return {"status": "reloaded"}

# KPI 1: Complaints per 1000 (denominator = unique Case ID count)
from kpis.kpi_complaints_per_1000 import complaints_per_1000

@app.get("/kpi/complaints_per_1000", response_model=ComplaintsPer1000Response)
def kpi_complaints_per_1000(
    month: str = Query(..., description="YYYY-MM (e.g., 2025-06)"),
    group_by: str = Query("Portfolio_std", description="Comma-separated columns to group by, e.g. 'Portfolio_std'"),
    portfolio_in: Optional[str] = Query(None, description="Optional CSV list to filter normalized portfolios (e.g., 'London,Chichester')")
):
    group_cols = [c.strip() for c in group_by.split(",") if c.strip()]
    portfolio_filter = None
    if portfolio_in:
        portfolio_filter = [p.strip().title() for p in portfolio_in.split(",") if p.strip()]

    try:
        df = complaints_per_1000(
            complaints_df=store.complaints,
            cases_df=store.cases,
            month=month,
            group_by=group_cols,
            portfolio_filter=portfolio_filter
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {"month": month, "group_by": group_cols, "rows": df.to_dict(orient="records")}

# KPI 2: Reason Mix % (free-text aware)
from kpis.kpi_reason_mix import reason_mix_percent

@app.get("/kpi/reason_mix", response_model=ReasonMixResponse)
def kpi_reason_mix(
    month: str = Query(..., description="YYYY-MM (e.g., 2025-06)"),
    group_by: str = Query("Portfolio_std", description="Comma-separated columns to group by, e.g. 'Portfolio_std'"),
    source_cols: str = Query(
        "Complaint Reason - Why is the member complaining ? ,Current Activity Reason,Root Cause,Process Category",
        description="Comma-separated list of columns to inspect (first non-empty wins)"
    ),
    top_n: int = Query(10, ge=1, le=50, description="Top reasons per group (by percent)"),
    include_unknown: bool = Query(True, description="Include 'Unknown' bucket")
):
    group_cols = [c.strip() for c in group_by.split(",") if c.strip()]
    reason_sources = [c.strip() for c in source_cols.split(",") if c.strip()]

    try:
        df, used_field = reason_mix_percent(
            complaints_df=store.complaints,
            month=month,
            group_by=group_cols,
            source_cols=reason_sources,
            top_n=top_n,
            include_unknown=include_unknown
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {
        "month": month,
        "group_by": group_cols,
        "reason_field": used_field,
        "rows": df.to_dict(orient="records")
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
