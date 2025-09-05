# questions/_utils.py
from __future__ import annotations
import pandas as pd

def ensure_month(df: pd.DataFrame, date_col: str, out_col="month") -> pd.DataFrame:
    out = df.copy()
    if out_col not in out.columns:
        out[out_col] = pd.to_datetime(out[date_col]).dt.strftime("%Y-%m")
    return out

def month_index(start_end):  # (YYYY-MM, YYYY-MM) -> list of months
    s, e = start_end
    s2 = pd.to_datetime(s+"-01"); e2 = pd.to_datetime(e+"-01")
    return [d.strftime("%Y-%m") for d in pd.date_range(s2, e2, freq="MS")]
