# questions/_utils.py
from __future__ import annotations
from typing import List, Optional, Tuple
import pandas as pd
from rapidfuzz import process, fuzz

# ---- column helpers ----
def pick_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None

def ensure_month_series(df: pd.DataFrame, date_candidates: List[str]) -> pd.Series:
    """
    Return a normalized month series (Timestamp at start of month).
    Prefers a prebuilt 'month_dt' if available, else parses any of the candidates.
    """
    if "month_dt" in df.columns and pd.api.types.is_datetime64_any_dtype(df["month_dt"]):
        s = df["month_dt"]
    else:
        s = None
        for c in date_candidates:
            if c in df.columns:
                s_try = pd.to_datetime(df[c], dayfirst=True, errors="coerce")
                if s_try.notna().any():
                    s = s_try
                    break
        if s is None:
            raise ValueError("No parsable date column found.")

    # normalize to month (timestamp at month start)
    return s.dt.to_period("M").dt.to_timestamp()

# ---- text helpers ----
def _norm(x) -> str:
    return str(x).strip().lower() if x is not None else ""

def fuzzy_pick(query: Optional[str], choices: List[str], cutoff: int = 80) -> Tuple[Optional[str], int]:
    if not query or not choices:
        return None, 0
    # map normalized -> original
    lut = {_norm(c): c for c in choices if pd.notna(c)}
    keys = list(lut.keys())
    got = process.extractOne(_norm(query), keys, scorer=fuzz.WRatio, score_cutoff=cutoff)
    if got is None:
        return None, 0
    return lut[got[0]], int(got[1])

def available_values(df: pd.DataFrame,
                     proc_cands: List[str],
                     port_cands: List[str],
                     date_cands: List[str]) -> dict:
    procc = pick_col(df, proc_cands)
    portc = pick_col(df, port_cands)
    months = ensure_month_series(df, date_cands)
    return {
        "processes": sorted(df[procc].dropna().unique().tolist()) if procc else [],
        "portfolios": sorted(df[portc].dropna().unique().tolist()) if portc else [],
        "months": months.dropna().drop_duplicates().sort_values().tolist()
    }
