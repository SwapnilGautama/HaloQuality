# data_store.py
import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "data"  # adjust if your path differs

# ---------- helpers ----------
def _canon(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.strip().str.lower()
    s = s.str.replace(r"\s+", " ", regex=True)
    s = s.str.replace("–", "-", regex=False).str.replace("—", "-", regex=False)
    return s

def _monthify(dt: pd.Series) -> pd.Series:
    m = pd.to_datetime(dt, errors="coerce", dayfirst=True)
    return m.dt.to_period("M").dt.to_timestamp()

def _mmm_yy(dt: pd.Series) -> pd.Series:
    return pd.to_datetime(dt, errors="coerce").dt.strftime("%b %y")

# ---------- cases ----------
def load_cases(dir_path: Path) -> pd.DataFrame:
    frames = []
    for f in sorted(dir_path.glob("*.xlsx")):
        frames.append(pd.read_excel(f))
    if not frames:
        return pd.DataFrame(columns=["portfolio","process","month_dt","month","case_id"])

    raw = pd.concat(frames, ignore_index=True)

    # Map your exact columns
    cases = raw.rename(columns={
        "Create Date": "create_date",
        "Portfolio": "portfolio",
        "Process Name": "process",
        "Case ID": "case_id",
    }).copy()

    cases["portfolio"] = _canon(cases["portfolio"])
    cases["process"]   = _canon(cases["process"])
    cases["month_dt"]  = _monthify(cases["create_date"])
    cases["month"]     = _mmm_yy(cases["month_dt"])
    cases = cases.dropna(subset=["month_dt"])

    if "case_id" in cases.columns:
        cases = cases.drop_duplicates(subset=["case_id","month_dt"])

    return cases[["portfolio","process","month_dt","month","case_id"]]

# ---------- complaints ----------
def load_complaints(dir_path: Path, cases_for_lookup: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for f in sorted(dir_path.glob("*.xlsx")):
        frames.append(pd.read_excel(f, sheet_name=0))
    if not frames:
        return pd.DataFrame(columns=["portfolio","process","month_dt","month","complaint_id","rca1","comment","orig_case_id"])

    raw = pd.concat(frames, ignore_index=True)

    complaints = raw.rename(columns={
        "Date Complaint Received - DD/MM/YY": "complaint_date",
        "Portfolio": "portfolio",
        "Parent Case Type": "process",
        "Original Process Affected Case ID": "orig_case_id",
        "RCA1": "rca1",
        "Case Comments": "comment",
        # "Complaint ID": "complaint_id",  # if you have a unique complaint id column, map it here
    }).copy()

    # Make a synthetic complaint_id if none exists
    if "Complaint ID" in raw.columns and "complaint_id" not in complaints.columns:
        complaints["complaint_id"] = raw["Complaint ID"]
    if "complaint_id" not in complaints.columns:
        complaints["complaint_id"] = np.arange(len(complaints))

    complaints["month_dt"] = _monthify(complaints["complaint_date"])
    complaints["month"]    = _mmm_yy(complaints["month_dt"])
    complaints = complaints.dropna(subset=["month_dt"])

    # Normalize
    if "portfolio" in complaints.columns:
        complaints["portfolio"] = _canon(complaints["portfolio"])
    else:
        complaints["portfolio"] = np.nan

    complaints["process"] = _canon(complaints.get("process", "").astype(str))

    # Back-fill Portfolio/Process from Cases via Original Process Affected Case ID when missing
    # (or when Parent Case Type is blank)
    if "orig_case_id" in complaints.columns:
        look = cases_for_lookup[["case_id","portfolio","process"]].copy()
        look = look.rename(columns={"case_id": "orig_case_id",
                                    "portfolio": "portfolio_from_case",
                                    "process": "process_from_case"})
        complaints = complaints.merge(look, on="orig_case_id", how="left")

        # fill portfolio if missing
        complaints["portfolio"] = complaints["portfolio"].fillna(complaints["portfolio_from_case"])
        # fill process if Parent Case Type missing
        complaints["process"] = complaints["process"].mask(
            complaints["process"].eq("") | complaints["process"].isna(),
            complaints["process_from_case"]
        )

    complaints["rca1"] = complaints.get("rca1", np.nan)
    complaints["comment"] = complaints.get("comment", "")

    complaints["portfolio"] = _canon(complaints["portfolio"])
    complaints["process"]   = _canon(complaints["process"])

    return complaints[["portfolio","process","month_dt","month","complaint_id","rca1","comment","orig_case_id"]]

# ---------- store ----------
def load_store() -> dict:
    cases = load_cases(DATA_DIR / "cases")
    complaints = load_complaints(DATA_DIR / "complaints", cases_for_lookup=cases)

    return {
        "cases": cases,
        "complaints": complaints,
        "cases_rows": len(cases),
        "complaints_rows": len(complaints),
        "last_case_month": cases["month_dt"].max() if not cases.empty else None,
        "last_complaint_month": complaints["month_dt"].max() if not complaints.empty else None,
    }
