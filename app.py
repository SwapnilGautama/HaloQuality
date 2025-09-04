
import os
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, Query, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np

# -------------------- Config paths --------------------
DEFAULT_DATA_DIR = os.getenv("HALO_DATA_DIR", "/mnt/data")
DEFAULT_COMPLAINTS_XLSX = os.getenv("COMPLAINTS_XLSX", f"{DEFAULT_DATA_DIR}/Complaints June'25.xlsx")
DEFAULT_CASES_XLSX      = os.getenv("CASES_XLSX", f"{DEFAULT_DATA_DIR}/June'25 data with unique identifier.xlsx")
DEFAULT_SURVEY_XLSX     = os.getenv("SURVEY_XLSX", f"{DEFAULT_DATA_DIR}/Overall raw data - June 2025.xlsx")

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

# -------------------- Data Store --------------------
class DataStore:
    def __init__(self):
        self.reload()

    def reload(self,
               complaints_path: str = DEFAULT_COMPLAINTS_XLSX,
               cases_path: str = DEFAULT_CASES_XLSX,
               survey_path: str = DEFAULT_SURVEY_XLSX):
        # Complaints
        try:
            comp = pd.read_excel(complaints_path, sheet_name=0)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to load complaints file: {e}")
        date_col = "Date Complaint Received - DD/MM/YY"
        if date_col not in comp.columns:
            raise HTTPException(status_code=400, detail=f"Complaints file missing '{date_col}'")
        comp["month"] = comp[date_col].apply(_to_month_str)
        if "Portfolio" in comp.columns:
            comp["Portfolio_std"] = comp["Portfolio"].apply(_std_portfolio)
        else:
            comp["Portfolio_std"] = np.nan
        self.complaints = comp

        # Cases
        try:
            cases = pd.read_excel(cases_path, sheet_name=0)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to load cases file: {e}")
        for c in ["Report_Date", "Case ID"]:
            if c not in cases.columns:
                raise HTTPException(status_code=400, detail=f"Cases file missing '{c}'")
        cases["month"] = cases["Report_Date"].apply(_to_month_str)
        if "Portfolio" in cases.columns:
            cases["Portfolio_std"] = cases["Portfolio"].apply(_std_portfolio)
        else:
            cases["Portfolio_std"] = np.nan
        self.cases = cases

        # Survey (optional)
        try:
            survey = pd.read_excel(survey_path, sheet_name=0)
            if "Portfolio" in survey.columns:
                survey["Portfolio_std"] = survey["Portfolio"].apply(_std_portfolio)
            survey["month"] = survey.get("Month_received", np.nan).apply(_to_month_str) if "Month_received" in survey.columns else np.nan
            self.survey = survey
        except Exception:
            self.survey = pd.DataFrame()

store = DataStore()

app = FastAPI(title="Halo Quality KPI API", version="0.1.0")

class ComplaintsPer1000Response(BaseModel):
    month: str
    group_by: List[str]
    rows: List[Dict[str, Any]]

@app.get("/health")
def health():
    return {"status": "ok", "complaints_rows": int(len(store.complaints)), "cases_rows": int(len(store.cases))}

@app.post("/reload")
def reload(complaints_path: Optional[str] = None,
           cases_path: Optional[str] = None,
           survey_path: Optional[str] = None):
    store.reload(
        complaints_path or DEFAULT_COMPLAINTS_XLSX,
        cases_path or DEFAULT_CASES_XLSX,
        survey_path or DEFAULT_SURVEY_XLSX
    )
    return {"status": "reloaded"}

from kpis.kpi_complaints_per_1000 import complaints_per_1000

@app.get("/kpi/complaints_per_1000", response_model=ComplaintsPer1000Response)
def kpi_complaints_per_1000(
    month: str,
    group_by: str = "Portfolio_std",
    portfolio_in: Optional[str] = None
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

    rows = df.to_dict(orient="records")
    return {"month": month, "group_by": group_cols, "rows": rows}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
