# questions/unique_cases_mom.py
from __future__ import annotations
import pandas as pd
from typing import Dict, Any, Optional, Tuple

def _pick(df: pd.DataFrame, *cands: str) -> Optional[str]:
    for c in cands:
        if c in df.columns:
            return c
    return None

def _month_floor(series: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(series):
        return series.dt.to_period("M").dt.to_timestamp()
    s = pd.to_datetime(series, errors="coerce", dayfirst=True)
    return s.dt.to_period("M").dt.to_timestamp()

def _maybe_date_range(params: Dict[str, Any]) -> Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]:
    # Expect params like {'start_month':
