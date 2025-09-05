# ingestion/loader.py
from __future__ import annotations
import re
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# ---------- Column alias map (source header -> canonical) ----------
# Add/extend here instead of touching code elsewhere.
CASE_ALIASES: Dict[str, str] = {
    # identity
    "Case ID": "Case_ID",
    "Case_ID": "Case_ID",
    "Report_Date": "Report_Date",
    "Create Date": "Create_Date",
    "Create_Date": "Create_Date",

    # dimensions
    "Event Type": "EventType",
    "Portfolio": "Portfolio_std",
    "Location": "Location_std",
    "ClientName": "ClientName",
    "Client Name": "ClientName",
    "Scheme": "Scheme",
    "Team Name": "TeamName",
    "TeamName": "TeamName",
    "Process Name": "ProcessName",
    "Process_Name": "ProcessName",
    "Process Group": "ProcessGroup",
    "Current Outsourcing Team": "OutsourcingTeam",
    "Onshore/Offshore": "Shore",
    "Manual/RPA": "Automation",
    "Critical": "Critical",
    "Pend Case": "PendCase",
    "Within SLA": "WithinSLA",
    "Consented/Non consented": "Consented",
    "Mercer Consented": "MercerConsented",
    "Vulnerable Customer": "VulnerableCustomer",

    # numerics
    "No. of Days": "NumDays",
    "Number of Days": "NumDays",
}

# Friendly labels for dims (used by NL parser)
DIM_CANONICAL = {
    "event type": "EventType",
    "event": "EventType",
    "portfolio": "Portfolio_std",
    "location": "Location_std",
    "client": "ClientName",
    "clientname": "ClientName",
    "scheme": "Scheme",
    "team": "TeamName",
    "teamname": "TeamName",
    "process": "ProcessName",
    "process name": "ProcessName",
    "process group": "ProcessGroup",
    "outsourcing": "OutsourcingTeam",
    "outsourcing team": "OutsourcingTeam",
    "shore": "Shore",
    "onshore/offshore": "Shore",
    "automation": "Automation",
    "manual/rpa": "Automation",
    "critical": "Critical",
    "pend": "PendCase",
    "pend case": "PendCase",
    "within sla": "WithinSLA",
    "consented": "Consented",
    "mercer consented": "MercerConsented",
    "vulnerable": "VulnerableCustomer",
    "vulnerable customer": "VulnerableCustomer",
}

YES_NO_MAP = {
    "y": "Yes", "yes": "Yes", "true": "Yes", "1": "Yes",
    "n": "No",  "no": "No",  "false": "No", "0": "No",
}

def _clean_header(h: str) -> str:
    """Normalize column header for alias lookup: strip, collapse space, preserve case for alias map."""
    return re.sub(r"\s+", " ", str(h)).strip()

def _apply_aliases(df: pd.DataFrame, aliases: Dict[str, str]) -> pd.DataFrame:
    remapped = {}
    for c in df.columns:
        key = _clean_header(c)
        if key in aliases:
            remapped[c] = aliases[key]
        else:
            # fallback: make a safe-ish name (spaces->_, strip)
            remapped[c] = re.sub(r"\s+", "_", key)
    return df.rename(columns=remapped)

def _to_datetime(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce", dayfirst=True, infer_datetime_format=True)

def _norm_yes_no(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().map(YES_NO_MAP).fillna(series)

def _strip_text(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip()

def load_cases(cases_dir: str | Path = "data/cases") -> pd.DataFrame:
    """
    Load and normalize ALL Excel files from data/cases into a canonical tidy DataFrame.

    Canonical columns we return (when available):
      - Case_ID (string)
      - Create_Date (datetime64[ns])
      - Report_Date (datetime64[ns], optional)
      - month_ym (YYYY-MM string from Create_Date)
      - month_mmm (MMM YY string from Create_Date)
      - EventType, Portfolio_std, Location_std, ClientName, Scheme, TeamName
      - ProcessName, ProcessGroup, OutsourcingTeam
      - Shore (Onshore/Offshore), Automation (Manual/RPA)
      - Critical, PendCase, WithinSLA, Consented, MercerConsented, VulnerableCustomer
      - NumDays (numeric)
    """
    cases_dir = Path(cases_dir)
    if not cases_dir.exists():
        return pd.DataFrame()

    frames: List[pd.DataFrame] = []
    for f in sorted(cases_dir.glob("**/*")):
        if f.suffix.lower() not in (".xlsx", ".xls"):
            continue
        try:
            df = pd.read_excel(f, dtype=None)
        except Exception:
            # Some sheets require engine='openpyxl'
            df = pd.read_excel(f, engine="openpyxl", dtype=None)
        if df.empty:
            continue

        df = _apply_aliases(df, CASE_ALIASES)

        # Date fields
        if "Create_Date" in df.columns:
            df["Create_Date"] = _to_datetime(df["Create_Date"])
        if "Report_Date" in df.columns:
            df["Report_Date"] = _to_datetime(df["Report_Date"])

        # Month keys from Create_Date (denominator grain)
        if "Create_Date" in df.columns:
            df["month_ym"] = df["Create_Date"].dt.strftime("%Y-%m")
            df["month_mmm"] = df["Create_Date"].dt.strftime("%b %y")

        # Normalize key categorical text
        for col in [
            "EventType", "Portfolio_std", "Location_std", "ClientName", "Scheme", "TeamName",
            "ProcessName", "ProcessGroup", "OutsourcingTeam", "Shore", "Automation",
            "PendCase",
        ]:
            if col in df.columns:
                df[col] = _strip_text(df[col])

        # Normalize boolean flags to Yes/No strings
        for col in ["Critical", "WithinSLA", "Consented", "MercerConsented", "VulnerableCustomer"]:
            if col in df.columns:
                df[col] = _norm_yes_no(df[col])

        # Numerics
        if "NumDays" in df.columns:
            df["NumDays"] = pd.to_numeric(df["NumDays"], errors="coerce")

        # Ensure Case_ID exists and is string
        if "Case_ID" in df.columns:
            df["Case_ID"] = _strip_text(df["Case_ID"].astype(str))

        frames.append(df)

    if not frames:
        return pd.DataFrame()

    data = pd.concat(frames, ignore_index=True)

    # Deduplicate (same case may appear in multiple pulls/files).
    # Keep the earliest Create_Date record per Case_ID.
    if "Case_ID" in data.columns and "Create_Date" in data.columns:
        data = data.sort_values(["Case_ID", "Create_Date"], ascending=[True, True])
        data = data.drop_duplicates(subset=["Case_ID"], keep="first")

    # Final tidy sort
    if "Create_Date" in data.columns:
        data = data.sort_values("Create_Date")

    return data


def available_case_dims(df: pd.DataFrame) -> List[str]:
    """Return canonical dimension columns present in df."""
    dims = [
        "EventType", "Portfolio_std", "Location_std", "ClientName", "Scheme", "TeamName",
        "ProcessName", "ProcessGroup", "OutsourcingTeam", "Shore", "Automation",
        "Critical", "PendCase", "WithinSLA", "Consented", "MercerConsented", "VulnerableCustomer",
    ]
    return [c for c in dims if c in df.columns]
