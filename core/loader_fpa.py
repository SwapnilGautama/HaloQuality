# core/loader_fpa.py
from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Optional
import re
import pandas as pd

# -------- Column aliases (robust to header variants) -----------------
FPA_ALIASES: Dict[str, str] = {
    # dates / ids
    "Report Date": "Review_Date",
    "Report_Date": "Review_Date",
    "Date": "Review_Date",
    "Review Date": "Review_Date",
    "Create Date": "Review_Date",  # fallback if that's what you get
    "Case ID": "Case_ID",

    # dims
    "Portfolio": "Portfolio_std",
    "Process Name": "ProcessName",
    "Process": "ProcessName",
    "Scheme": "Scheme",
    "Team Name": "TeamName",
    "Team": "TeamName",
    "Team Manager": "TeamManager",
    "Location": "Location",

    # result + comments
    "Review Result": "ReviewResult",
    "Review_Result": "ReviewResult",
    "Result": "ReviewResult",
    "Case Comment": "CaseComment",
    "Comments": "CaseComment",
    "Comment": "CaseComment",
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
            # normalize to code-safe header
            remap[c] = re.sub(r"\s+", "_", k)
    return df.rename(columns=remap)

def _to_datetime(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce", dayfirst=True, infer_datetime_format=True)

def _strip_cols(df: pd.DataFrame, cols: List[str]) -> None:
    for c in cols:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()

def _norm_key(s: pd.Series | str) -> pd.Series | str:
    if isinstance(s, pd.Series):
        return s.astype(str).str.strip().str.lower()
    return str(s).strip().lower()

def _is_fail(series: pd.Series) -> pd.Series:
    """
    Heuristic for 'fail' values in ReviewResult.
    Accepts values like: 'fail', 'failed', 'no', 'n', 'x', '0' etc.
    """
    s = series.astype(str).str.strip().str.lower()
    return s.isin({"fail", "failed", "no", "n", "x", "0", "fail - major", "fail - minor"})

def load_fpa(fpa_dir: str | Path = "data/first_pass_accuracy") -> pd.DataFrame:
    """
    Load & normalize all FPA files. Ensures:
      - Review_Date parsed
      - Month (MMM YY) derived
      - Clean dimensions (Portfolio_std, ProcessName, Scheme, TeamName, TeamManager, Location)
      - ReviewResult + FailFlag created
      - Normalized join keys (PortfolioKey, ProcessKey)
    """
    fpa_dir = Path(fpa_dir)
    if not fpa_dir.exists():
        return pd.DataFrame()

    frames: List[pd.DataFrame] = []
    for f in sorted(fpa_dir.glob("**/*")):
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

        df = _apply_aliases(df, FPA_ALIASES)

        # date → Month
        if "Review_Date" in df.columns:
            df["Review_Date"] = _to_datetime(df["Review_Date"])
            df["Month"] = df["Review_Date"].dt.strftime("%b %y")

        # strip dims
        _strip_cols(df, ["Portfolio_std","ProcessName","Scheme","TeamName","TeamManager","Location","ReviewResult","CaseComment"])

        # Fail flag
        if "ReviewResult" in df.columns:
            df["FailFlag"] = _is_fail(df["ReviewResult"])
        else:
            df["FailFlag"] = False

        # normalized join keys
        if "Portfolio_std" in df.columns:
            df["PortfolioKey"] = _norm_key(df["Portfolio_std"])
        else:
            df["PortfolioKey"] = pd.NA
        if "ProcessName" in df.columns:
            df["ProcessKey"] = _norm_key(df["ProcessName"])
        else:
            df["ProcessKey"] = pd.NA

        frames.append(df)

    if not frames:
        return pd.DataFrame()

    data = pd.concat(frames, ignore_index=True)

    keep = [
        "Case_ID","Review_Date","Month",
        "Portfolio_std","PortfolioKey",
        "ProcessName","ProcessKey",
        "Scheme","TeamName","TeamManager","Location",
        "ReviewResult","FailFlag","CaseComment"
    ]
    keep = [c for c in keep if c in data.columns]
    return data[keep].copy()
