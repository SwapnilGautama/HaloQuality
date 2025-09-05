# core/loader_cases.py
from __future__ import annotations
from pathlib import Path
from typing import Dict, List
import re
import pandas as pd

# Map raw headers → normalized names we’ll use everywhere
CASE_ALIASES: Dict[str, str] = {
    # dates / ids
    "Create Date": "Create_Date",
    "Create_Date": "Create_Date",
    "Case ID": "Case_ID",

    # dims you asked to slice by
    "Event Type": "EventType",
    "Portfolio": "Portfolio_std",
    "Location": "Location",
    "ClientName": "ClientName",
    "Client Name": "ClientName",
    "Scheme": "Scheme",
    "Team Name": "TeamName",
    "Team Name?": "TeamName",
    "Team": "TeamName",
    "Process Name": "ProcessName",
    "Process": "ProcessName",
    "Process Group": "ProcessGroup",
    "Current Outsourcing Team": "CurrentOutsourcingTeam",
    "Onshore/Offshore": "OnshoreOffshore",
    "Completes Current Onshore/Offshore": "OnshoreOffshore",
    "Manual/RPA": "ManualRPA",
    "Manual/ RPA": "ManualRPA",
    "Critical": "Critical",
    "Pend Case": "PendCase",
    "Within SLA": "WithinSLA",
    "Within SLA Operation": "WithinSLA",
    "Consented/Non consented": "Consented",
    "Mercer Consented": "MercerConsented",
    "Vulnerable Customer": "VulnerableCustomer",

    # metrics
    "No. of Days": "NoOfDays",
}

def _clean_header(h: str) -> str:
    return re.sub(r"\s+", " ", str(h)).strip()

def _apply_aliases(df: pd.DataFrame, aliases: Dict[str, str]) -> pd.DataFrame:
    remap = {}
    for c in df.columns:
        key = _clean_header(c)
        if key in aliases:
            remap[c] = aliases[key]
        else:
            # safe fallback: normalize to snake-ish
            remap[c] = re.sub(r"\s+", "_", key)
    return df.rename(columns=remap)

def _to_datetime(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce", dayfirst=True, infer_datetime_format=True)

def _strip(df: pd.DataFrame, cols: List[str]) -> None:
    for c in cols:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()

def load_cases(cases_dir: str | Path = "data/cases") -> pd.DataFrame:
    """
    Load & normalize all case files under data/cases.
    Ensures:
      - Create_Date parsed, month_ym/month_mmm derived
      - Case_ID present & de-duplicated
      - Dimensions standardized (EventType, Portfolio_std, Location, ClientName, Scheme,
        TeamName, ProcessName, ProcessGroup, CurrentOutsourcingTeam, OnshoreOffshore,
        ManualRPA, Critical, PendCase, WithinSLA, Consented, MercerConsented, VulnerableCustomer)
      - NoOfDays kept as numeric for KPI like avg days
    """
    cases_dir = Path(cases_dir)
    if not cases_dir.exists():
        return pd.DataFrame()

    frames: List[pd.DataFrame] = []
    for f in sorted(cases_dir.glob("**/*")):
        if f.suffix.lower() not in (".xlsx", ".xls", ".csv"):
            continue
        try:
            if f.suffix.lower() == ".csv":
                df = pd.read_csv(f)
            else:
                df = pd.read_excel(f, dtype=None)
        except Exception:
            df = pd.read_excel(f, engine="openpyxl", dtype=None)

        if df.empty:
            continue

        df = _apply_aliases(df, CASE_ALIASES)

        # dates
        if "Create_Date" in df.columns:
            df["Create_Date"] = _to_datetime(df["Create_Date"])
            df["month_ym"] = df["Create_Date"].dt.strftime("%Y-%m")
            df["month_mmm"] = df["Create_Date"].dt.strftime("%b %y")

        # normalize text dims
        _strip(df, [
            "EventType","Portfolio_std","Location","ClientName","Scheme","TeamName",
            "ProcessName","ProcessGroup","CurrentOutsourcingTeam","OnshoreOffshore",
            "ManualRPA","Critical","PendCase","WithinSLA","Consented",
            "MercerConsented","VulnerableCustomer"
        ])

        # numeric NoOfDays
        if "NoOfDays" in df.columns:
            df["NoOfDays"] = pd.to_numeric(df["NoOfDays"], errors="coerce")

        frames.append(df)

    if not frames:
        return pd.DataFrame()

    data = pd.concat(frames, ignore_index=True)

    # keep one row per Case_ID (latest Create_Date wins)
    if "Case_ID" in data.columns:
        data = (
            data.sort_values(["Case_ID", "Create_Date"], na_position="last")
                .drop_duplicates(subset=["Case_ID"], keep="last")
        )

    # Columns we expect to use across KPIs/joins
    keep = [
        "Case_ID","Create_Date","month_ym","month_mmm","NoOfDays",
        "EventType","Portfolio_std","Location","ClientName","Scheme","TeamName",
        "ProcessName","ProcessGroup","CurrentOutsourcingTeam","OnshoreOffshore",
        "ManualRPA","Critical","PendCase","WithinSLA","Consented",
        "MercerConsented","VulnerableCustomer"
    ]
    keep = [c for c in keep if c in data.columns]
    return data[keep].copy()
