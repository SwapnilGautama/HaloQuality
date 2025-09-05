# question_engine/drivers.py
from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import pandas as pd

def drivers_of_fails(
    fpa_df: pd.DataFrame,
    group_by: List[str] = ("ProcessName", "TeamName"),
    month_from: Optional[str] = None,
    month_to: Optional[str] = None,
    use_reasons: bool = True,   # include FPA_PrimaryTag if available
    top_n: int = 10
) -> Dict[str, pd.DataFrame]:
    """
    Biggest drivers of case fails:
      - Ranks segments by Fails (level) and Fail_Rate.
      - If Month present and a range is provided, computes delta vs previous period (optional extension).
    Returns dict of tables.
    """
    if fpa_df is None or fpa_df.empty:
        return {"summary": pd.DataFrame()}

    df = fpa_df.copy()
    if month_from:
        df = df[df["Month"] >= month_from]
    if month_to:
        df = df[df["Month"] <= month_to]

    dims = [c for c in group_by if c in df.columns]
    if "Month" in df.columns and "Month" not in dims:
        dims = ["Month"] + dims

    # base metrics
    g = df.groupby(dims, dropna=False).agg(
        Reviewed=("Case_ID","count") if "Case_ID" in df.columns else ("ReviewResult","count"),
        Fails=("FailFlag","sum") if "FailFlag" in df.columns else ("ReviewResult","count")
    ).reset_index()
    g["Fail_Rate"] = (g["Fails"] / g["Reviewed"]).fillna(0.0)

    # overall rank by fails (latest month if Month present & range single-month)
    if "Month" in g.columns:
        latest = g["Month"].max()
        latest_slice = g[g["Month"] == latest]
        top = latest_slice.sort_values(["Fails","Fail_Rate"], ascending=[False, False]).head(top_n)
    else:
        top = g.sort_values(["Fails","Fail_Rate"], ascending=[False, False]).head(top_n)

    out = {"top_segments": top}

    # Add reasons if available
    if use_reasons and "FPA_PrimaryTag" in df.columns:
        r = df[df["FailFlag"] == True].groupby(dims + ["FPA_PrimaryTag"], dropna=False).size() \
                                       .reset_index(name="Failures")
        # restrict to top segments (latest month slice if present)
        key_cols = [c for c in dims if c != "Month"]
        if not key_cols:
            out["reasons"] = r.sort_values("Failures", ascending=False).head(top_n)
        else:
            keys = top[key_cols].drop_duplicates()
            r2 = r.merge(keys, on=key_cols, how="inner")
            # within-segment shares
            base = r2.groupby(dims)["Failures"].transform("sum")
            r2["Share"] = (r2["Failures"] / base).fillna(0.0)
            out["reasons"] = r2.sort_values(["Failures"], ascending=False).groupby(key_cols).head(5)
    return out
