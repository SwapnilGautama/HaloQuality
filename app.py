
# app.py — Halo Quality KPI API (v0.4.0) with KPIs 1-4
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

# Complaints & Survey single-file defaults (overridable)
DEFAULT_COMPLAINTS_XLSX = os.getenv("COMPLAINTS_XLSX", f"{DEFAULT_DATA_DIR}/Complaints June'25.xlsx")
DEFAULT_SURVEY_XLSX    = os.getenv("SURVEY_XLSX", f"{DEFAULT_DATA_DIR}/Overall raw data - June 2025.xlsx")

# Cases (denominator) supports a DIRECTORY of monthly files
DEFAULT_CASES_DIR  = os.getenv("CASES_DIR", f"{DEFAULT_DATA_DIR}/cases")
DEFAULT_CASES_XLSX = os.getenv("CASES_XLSX", f"{DEFAULT_DATA_DIR}/June'25 data with unique identifier.xlsx")

# -------------------- Helpers --------------------
def _to_month_str(dt) -> Optional[str]:
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
    if pd.isna(s):
        return s
    t = str(s).strip()
    low = t.lower()
    low = low.replace("leatherhead - baes", "baes leatherhead").replace("baes-leatherhead", "baes leatherhead")
    low = low.replace("north west", "northwest")
    return low.title()

def _collect_case_files(cases_input: Optional[str]) -> List[str]:
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
        raise HTTPException(status_code=400, detail="No case files found. Put monthly .xlsx under data/cases or set CASES_DIR/CASES_XLSX/reload(cases_path=...).")
    return candidates

# -------------------- Data Store --------------------
class DataStore:
    def __init__(self):
        self.reload()

    def reload(self, complaints_path: str = DEFAULT_COMPLAINTS_XLSX, cases_path: Optional[str] = None, survey_path: str = DEFAULT_SURVEY_XLSX):
        # Complaints
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

        # Cases (many files)
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
        cases = cases.dropna(subset=["Case ID"])
        cases["Case ID"] = cases["Case ID"].astype(str)
        cases = cases.drop_duplicates(subset=["month", "Case ID"], keep="first")
        self.cases = cases

        # Survey (optional)
        try:
            survey = pd.read_excel(survey_path, sheet_name=0)
            survey["Portfolio_std"] = survey.get("Portfolio", np.nan).apply(_std_portfolio) if "Portfolio" in survey.columns else np.nan
            survey["month"] = survey.get("Month_received", np.nan).apply(_to_month_str) if "Month_received" in survey.columns else np.nan
            self.survey = survey
        except Exception:
            self.survey = pd.DataFrame()

store = DataStore()

app = FastAPI(title="Halo Quality KPI API", version="0.4.0")

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

class NPSResponse(BaseModel):
    month: str
    group_by: List[str]
    rows: List[Dict[str, Any]]

class ExperienceScoresResponse(BaseModel):
    month: str
    group_by: List[str]
    used_fields: Dict[str, str]
    include_somewhat: bool
    rows: List[Dict[str, Any]]

# -------------------- Routes --------------------
@app.get("/health")
def health():
    return {
        "status": "ok",
        "complaints_rows": int(len(store.complaints)),
        "cases_rows": int(len(store.cases)),
        "cases_months": sorted([m for m in store.cases["month"].dropna().unique().tolist()]) if not store.cases.empty else [],
        "survey_rows": int(len(store.survey))
    }

@app.post("/reload")
def reload(complaints_path: Optional[str] = None, cases_path: Optional[str] = None, survey_path: Optional[str] = None):
    store.reload(complaints_path or DEFAULT_COMPLAINTS_XLSX, cases_path or None, survey_path or DEFAULT_SURVEY_XLSX)
    return {"status": "reloaded"}

# KPI 1: Complaints per 1000
from kpi.kpi_complaints_per_1000 import complaints_per_1000

@app.get("/kpi/complaints_per_1000", response_model=ComplaintsPer1000Response)
def kpi_complaints_per_1000(month: str = Query(..., description="YYYY-MM (e.g., 2025-06)"),
                            group_by: str = Query("Portfolio_std", description="Comma-separated columns to group by"),
                            portfolio_in: Optional[str] = Query(None, description="CSV list to filter normalized portfolios")):
    group_cols = [c.strip() for c in group_by.split(",") if c.strip()]
    portfolio_filter = [p.strip().title() for p in portfolio_in.split(",")] if portfolio_in else None
    try:
        df = complaints_per_1000(store.complaints, store.cases, month, group_cols, portfolio_filter)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"month": month, "group_by": group_cols, "rows": df.to_dict(orient="records")}

# KPI 2: Reason Mix %
from kpi.kpi_reason_mix import reason_mix_percent

@app.get("/kpi/reason_mix", response_model=ReasonMixResponse)
def kpi_reason_mix(month: str = Query(..., description="YYYY-MM (e.g., 2025-06)"),
                   group_by: str = Query("Portfolio_std", description="Comma-separated columns to group by"),
                   source_cols: str = Query("Complaint Reason - Why is the member complaining ? ,Current Activity Reason,Root Cause,Process Category,Event Type",
                                            description="CSV list of fields to inspect; first non-empty wins"),
                   top_n: int = Query(10, ge=1, le=50),
                   include_unknown: bool = Query(True)):
    group_cols = [c.strip() for c in group_by.split(",") if c.strip()]
    reason_sources = [c.strip() for c in source_cols.split(",") if c.strip()]
    try:
        df, used_field = reason_mix_percent(store.complaints, month, group_cols, reason_sources, top_n, include_unknown)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"month": month, "group_by": group_cols, "reason_field": used_field, "rows": df.to_dict(orient="records")}

# KPI 3: NPS by group
from kpi.kpi_nps import nps_by_group

@app.get("/kpi/nps", response_model=NPSResponse)
def kpi_nps(month: str = Query(..., description="YYYY-MM (e.g., 2025-06)"),
            group_by: str = Query("Portfolio_std", description="Comma-separated columns to group by"),
            min_responses: int = Query(5, ge=1, le=10000)):
    if store.survey.empty:
        raise HTTPException(status_code=400, detail="Survey file is not loaded or empty. Set SURVEY_XLSX or /reload with survey_path.")
    group_cols = [c.strip() for c in group_by.split(",") if c.strip()]
    try:
        df = nps_by_group(store.survey, month, group_cols, min_responses)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"month": month, "group_by": group_cols, "rows": df.to_dict(orient="records")}

# KPI 4: Experience Scores (Agree/Strongly Agree)
from kpi.kpi_experience_scores import experience_scores_by_group

@app.get("/kpi/experience_scores", response_model=ExperienceScoresResponse)
def kpi_experience_scores(month: str = Query(..., description="YYYY-MM (e.g., 2025-06)"),
                          group_by: str = Query("Portfolio_std", description="Comma-separated columns to group by"),
                          fields: str = Query("Clarity=Clear_Information,Timescale=Timescale,Handling=Handle_Issue",
                                              description="Mapping CSV like 'Clarity=Clear_Information,Timescale=Timescale,Handling=Handle_Issue'"),
                          include_somewhat: bool = Query(False, description="If true, counts 'Somewhat agree' as agree too"),
                          min_responses: int = Query(5, ge=1, le=10000)):
    if store.survey.empty:
        raise HTTPException(status_code=400, detail="Survey file is not loaded or empty. Set SURVEY_XLSX or /reload with survey_path.")
    group_cols = [c.strip() for c in group_by.split(",") if c.strip()]
    # parse mapping
    field_map: Dict[str, str] = {}
    if fields:
        for token in fields.split(","):
            token = token.strip()
            if token and "=" in token:
                k, v = token.split("=", 1)
                field_map[k.strip()] = v.strip()
    try:
        df = experience_scores_by_group(store.survey, month, group_cols, field_map if field_map else None, include_somewhat, min_responses)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    used_fields = {"Clarity": field_map.get("Clarity", "Clear_Information"),
                   "Timescale": field_map.get("Timescale", "Timescale"),
                   "Handling": field_map.get("Handling", "Handle_Issue")}
    return {"month": month, "group_by": group_cols, "used_fields": used_fields, "include_somewhat": include_somewhat, "rows": df.to_dict(orient="records")}



# KPI 5: Month-over-Month overview
from kpi.kpi_mom import mom_overview

class MoMResponse(BaseModel):
    month: str
    prev_month: str
    group_by: List[str]
    include_somewhat: bool
    min_responses: int
    rows: List[Dict[str, Any]]

@app.get("/kpi/mom", response_model=MoMResponse)
def kpi_mom(
    month: str = Query(..., description="YYYY-MM (e.g., 2025-06)"),
    group_by: str = Query("Portfolio_std", description="Comma-separated columns to group by"),
    include_somewhat: bool = Query(False, description="For experience metrics, include 'Somewhat agree' as agree"),
    min_responses: int = Query(5, ge=1, le=10000, description="Minimum survey responses per group")
):
    group_cols = [c.strip() for c in group_by.split(",") if c.strip()]

    df, prev_m = mom_overview(
        complaints_df=store.complaints,
        cases_df=store.cases,
        survey_df=store.survey,
        month=month,
        group_by=group_cols,
        include_somewhat=include_somewhat,
        min_responses=min_responses,
        fields_map=None  # use defaults; caller can add fields param later if needed
    )

    return {
        "month": month,
        "prev_month": prev_m,
        "group_by": group_cols,
        "include_somewhat": include_somewhat,
        "min_responses": min_responses,
        "rows": df.to_dict(orient="records")
    }
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
