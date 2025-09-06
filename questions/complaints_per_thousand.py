# questions/complaints_per_thousand.py
from __future__ import annotations
import pandas as pd

def _month_str(s: pd.Series) -> pd.Series:
    s = pd.to_datetime(s, errors="coerce")
    return s.dt.strftime("%b %y")

def run(store: dict, params: dict, user_text: str | None = None) -> dict:
    """
    Inputs (params):
      - portfolio: str (optional; default All)
      - start_month: 'YYYY-MM' or Timestamp (optional)
      - end_month:   'YYYY-MM' or Timestamp (optional)
    """
    cases = store.get("cases", pd.DataFrame()).copy()
    cmps  = store.get("complaints", pd.DataFrame()).copy()

    # Defensive dtypes
    for df in (cases, cmps):
        if "_month_dt" in df:
            df["_month_dt"] = pd.to_datetime(df["_month_dt"], errors="coerce")
        for col in ("Process", "Portfolio", "Case ID"):
            if col in df:
                df[col] = df[col].astype("string").str.strip()

    # Resolve filters
    portfolio = (params or {}).get("portfolio")
    start = pd.to_datetime((params or {}).get("start_month"), errors="coerce")
    end   = pd.to_datetime((params or {}).get("end_month"),   errors="coerce")

    # If no explicit range, use intersection of available months
    if start is pd.NaT or end is pd.NaT:
        m1 = cases["_month_dt"].min()
        m2 = cases["_month_dt"].max()
        m3 = cmps["_month_dt"].min()
        m4 = cmps["_month_dt"].max()
        start = pd.to_datetime(min([x for x in [m1,m3] if pd.notna(x)]), errors="coerce")
        end   = pd.to_datetime(max([x for x in [m2,m4] if pd.notna(x)]), errors="coerce")

    # Filter portfolio
    if portfolio and portfolio != "All":
        if "Portfolio" in cases: cases = cases[cases["Portfolio"].eq(portfolio)]
        if "Portfolio" in cmps:  cmps  = cmps[ cmps["Portfolio"].eq(portfolio)]

    # Filter month range
    if pd.notna(start): cases = cases[cases["_month_dt"] >= start]
    if pd.notna(end):   cases = cases[cases["_month_dt"] <= end]
    if pd.notna(start): cmps  = cmps[ cmps["_month_dt"] >= start]
    if pd.notna(end):   cmps  = cmps[ cmps["_month_dt"] <= end]

    # Aggregate
    cases_g = (
        cases.dropna(subset=["_month_dt"])
             .groupby(["_month_dt", "Process"], dropna=False, as_index=False)
             .agg(cases=("Case ID", "nunique"))
    )
    cmps_g = (
        cmps.dropna(subset=["_month_dt"])
            .groupby(["_month_dt", "Process"], dropna=False, as_index=False)
            .agg(complaints=("Case ID", "nunique"))
    )

    out = cases_g.merge(cmps_g, on=["_month_dt", "Process"], how="outer")
    out["cases"] = out["cases"].fillna(0)
    out["complaints"] = out["complaints"].fillna(0)
    out["per_1000"] = (out["complaints"] / out["cases"].where(out["cases"].ne(0), pd.NA)) * 1000
    out["per_1000"] = out["per_1000"].fillna(0).round(1)
    out["month"] = _month_str(out["_month_dt"])

    out = out.sort_values(["_month_dt", "Process"], kind="stable")[["month", "Process", "cases", "complaints", "per_1000"]]

    meta = {
        "title": "Complaints per 1,000 cases",
        "filters": {
            "portfolio": portfolio or "All",
            "start_month": pd.to_datetime(start).strftime("%Y-%m") if pd.notna(start) else None,
            "end_month": pd.to_datetime(end).strftime("%Y-%m") if pd.notna(end) else None,
        },
    }
    return {"dataframe": out, "meta": meta}
