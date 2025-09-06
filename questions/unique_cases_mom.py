# questions/unique_cases_mom.py
from __future__ import annotations
import pandas as pd

def run(store: dict, params: dict, user_text: str | None = None) -> dict:
    """
    Show unique case volume month-over-month (optionally by portfolio).
    params:
      - start_month, end_month (optional)
      - portfolio (optional)
    """
    cases = store.get("cases", pd.DataFrame()).copy()
    if cases.empty:
        return {"dataframe": pd.DataFrame([{"_month":"", "unique_cases":0}]),
                "meta": {"title":"Unique cases (MoM)", "filters": {}}}

    cases["_month_dt"] = pd.to_datetime(cases["_month_dt"], errors="coerce")
    cases["Case ID"] = cases["Case ID"].astype("string")

    portfolio = (params or {}).get("portfolio")
    start = pd.to_datetime((params or {}).get("start_month"), errors="coerce")
    end   = pd.to_datetime((params or {}).get("end_month"),   errors="coerce")

    if portfolio and portfolio != "All" and "Portfolio" in cases:
        cases = cases[cases["Portfolio"].eq(portfolio)]

    if pd.notna(start): cases = cases[cases["_month_dt"] >= start]
    if pd.notna(end):   cases = cases[cases["_month_dt"] <= end]

    g = (cases.dropna(subset=["_month_dt"])
              .groupby("_month_dt", as_index=False)
              .agg(unique_cases=("Case ID", "nunique")))
    g["_month"] = pd.to_datetime(g["_month_dt"]).dt.strftime("%b %y")
    g = g.sort_values("_month_dt")[["_month", "unique_cases"]]

    meta = {
        "title": "Unique cases (MoM)",
        "filters": {
            "portfolio": portfolio or "All",
            "start_month": pd.to_datetime(start).strftime("%Y-%m") if pd.notna(start) else None,
            "end_month": pd.to_datetime(end).strftime("%Y-%m") if pd.notna(end) else None,
        },
    }
    return {"dataframe": g, "meta": meta}
