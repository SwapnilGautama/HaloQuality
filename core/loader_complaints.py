# core/loader_complaints.py
from __future__ import annotations
from pathlib import Path
from typing import Dict, List
import re
import pandas as pd

COMPL_ALIASES: Dict[str, str] = {
    "Report_Date": "Report_Date",
    "Report Date": "Report_Date",
    "Parent Case Type": "ParentCaseType",
    "Portfolio": "Portfolio_std",
    "Scheme": "Scheme",
    "Receipt Method": "ReceiptMethod",
    "Parent Team": "ParentTeam",
    "Aptia Error": "AptiaError",
    "Control": "Control",
    "Brief Description - RCA done by admin": "RCA_Brief",
    "Why": "RCA_Why",
}

def _clean_header(h: str) -> str:
    return re.sub(r"\s+", " ", str(h)).strip()

def _apply_aliases(df: pd.DataFrame, aliases: Dict[str, str]) -> pd.DataFrame:
    remap = {}
    for c in df.columns:
        k = _clean_header(c)
        if k in aliases:
            remap[c] = aliases[k]
        else:
            remap[c] = re.sub(r"\s+", "_", k)
    return df.rename(columns=remap)

def _to_datetime(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce", dayfirst=True, infer_datetime_format=True)

def _strip(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip()

def load_complaints(complaints_dir: str | Path = "data/complaints") -> pd.DataFrame:
    complaints_dir = Path(complaints_dir)
    if not complaints_dir.exists():
        return pd.DataFrame()

    frames: List[pd.DataFrame] = []
    for f in sorted(complaints_dir.glob("**/*")):
        if f.suffix.lower() not in (".xlsx", ".xls"):
            continue
        try:
            df = pd.read_excel(f, dtype=None)
        except Exception:
            df = pd.read_excel(f, engine="openpyxl", dtype=None)
        if df.empty:
            continue

        df = _apply_aliases(df, COMPL_ALIASES)

        if "Report_Date" in df.columns:
            df["Report_Date"] = _to_datetime(df["Report_Date"])
            df["report_month_ym"] = df["Report_Date"].dt.strftime("%Y-%m")
            df["report_month_mmm"] = df["Report_Date"].dt.strftime("%b %y")

        for c in ["ParentCaseType","Portfolio_std","Scheme","ReceiptMethod","ParentTeam","AptiaError","Control"]:
            if c in df.columns:
                df[c] = _strip(df[c])

        for c in ["RCA_Brief","RCA_Why"]:
            if c in df.columns:
                df[c] = df[c].fillna("").astype(str)
        df["RCA_Text"] = (df.get("RCA_Brief","") + " " + df.get("RCA_Why","")).str.strip()

        frames.append(df)

    if not frames:
        return pd.DataFrame()

    data = pd.concat(frames, ignore_index=True)
    keep = [
        "ParentCaseType","Report_Date","report_month_ym","report_month_mmm",
        "Portfolio_std","Scheme","ReceiptMethod","ParentTeam","AptiaError","Control",
        "RCA_Brief","RCA_Why","RCA_Text"
    ]
    keep = [c for c in keep if c in data.columns]
    return data[keep].copy()
