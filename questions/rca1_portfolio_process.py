# questions/rca1_portfolio_process.py
from __future__ import annotations
import re
import pandas as pd
from typing import Optional, Tuple, Dict, Any

# Helpers ---------------------------------------------------------------------

def _pick(df: pd.DataFrame, *cands: str) -> Optional[str]:
    for c in cands:
        if c in df.columns:
            return c
    return None

def _month_floor(series: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(series):
        return series.dt.to_period("M").dt.to_timestamp()
    # if string -> try parse
    s = pd.to_datetime(series, errors="coerce", dayfirst=True)
    return s.dt.to_period("M").dt.to_timestamp()

def _parse_process(user_text: str, default: str) -> str:
    if not user_text:
        return default
    m = re.search(r"process\s+([A-Za-z][\w\s/&-]+)", user_text, flags=re.I)
    return (m.group(1).strip() if m else default) or default

# Public API ------------------------------------------------------------------

def run(store: Dict[str, pd.DataFrame],
        user_text: str = "",
        months: int = 3,
        process_name: Optional[str] = None,
        **kwargs: Any):
    """
    RCA1 by Portfolio x Process (last N months, default 3)

    Required columns in complaints:
      - Process: ['Process Name', 'Process', 'Process_Name']
      - Portfolio: ['Portfolio', 'Portfolio Name']
      - Month/Date: precomputed 'month_dt' by data_store OR a raw date col we floor here
      - RCA1: must exist (labels)
    """

    complaints = store.get("complaints")
    if complaints is None or complaints.empty:
        return "No complaints data available."

    # Column detection
    c_process = _pick(complaints, "Process Name", "Process", "Process_Name")
    c_port    = _pick(complaints, "Portfolio", "Portfolio Name", "Portfolio_Name")
    c_month   = "month_dt" if "month_dt" in complaints.columns else _pick(
        complaints, "Date Complaint Received - DD/MM/YY", "Report Date", "Report_Date", "Report Dt"
    )
    c_rca1    = _pick(complaints, "RCA1", "RCA_1")

    if not all([c_process, c_port, c_month]):
        return "Required columns not found in complaints (need Process, Portfolio and a Date)."

    # Fail fast if labels are missing
    if not c_rca1 or c_rca1 not in complaints.columns:
        return "RCA labels not found. Please run the complaints labeller so 'RCA1' exists."

    df = complaints.copy()

    # Ensure month column exists
    if c_month != "month_dt":
        df["month_dt"] = _month_floor(df[c_month])
    else:
        # sanity
        df["month_dt"] = _month_floor(df["month_dt"])

    # Filter last N months early
    max_month = df["month_dt"].max()
    if pd.notna(max_month) and months and months > 0:
        start = (max_month.to_period("M") - (months - 1)).to_timestamp()
        df = df.loc[df["month_dt"] >= start]

    # Which process?
    proc = process_name or _parse_process(user_text, default="Member Enquiry")
    df = df.loc[df[c_process].str.strip().str.casefold() == proc.strip().casefold()]

    if df.empty:
        return f"No complaints for process '{proc}' in the selected period."

    # Group & pivot: Portfolio x RCA1 (count)
    out = (
        df.groupby([c_port, c_rca1], dropna=False)
          .size()
          .reset_index(name="count")
          .sort_values(["count", c_port], ascending=[False, True])
    )

    # Pretty render (Streamlit-friendly dict)
    return {
        "title": f"RCA1 by Portfolio × {proc} (last {months} mo.)",
        "data": out,
        "index_cols": [c_port, c_rca1],
        "value_col": "count",
    }
