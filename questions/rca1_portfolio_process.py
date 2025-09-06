# questions/rca1_portfolio_process.py
from __future__ import annotations
import pandas as pd

def run(store: dict, params: dict, user_text: str | None = None) -> dict:
    """
    Show RCA1 (first-level reason) counts by portfolio × process for last N months (default 3).
    Requires a column 'RCA1' in complaints. If missing, return an informative table.
    params:
      - months: int = 3
      - portfolio: optional
    """
    cmps = store.get("complaints", pd.DataFrame()).copy()
    if cmps.empty:
        return {"dataframe": pd.DataFrame([{"portfolio":"", "process":"", "rca1":"", "complaints":0}]),
                "meta": {"title":"RCA1 by Portfolio × Process — last 3 months", "filters": {}}}

    if "RCA1" not in cmps.columns:
        msg = pd.DataFrame(
            [{"portfolio":"", "process":"", "rca1":"",
              "complaints":"RCA1 labels not found. Add an 'RCA1' column to complaints (e.g., via your labeller)."}]
        )
        return {"dataframe": msg, "meta": {"title":"RCA1 by Portfolio × Process — last 3 months", "filters": {}}}

    months = int((params or {}).get("months") or 3)
    cmps["_month_dt"] = pd.to_datetime(cmps["_month_dt"], errors="coerce")
    last = cmps["_month_dt"].max()
    if pd.isna(last):
        return {"dataframe": pd.DataFrame([{"portfolio":"", "process":"", "rca1":"", "complaints":0}]),
                "meta": {"title":"RCA1 by Portfolio × Process — last 3 months", "filters": {}}}
    start = (last.to_period("M") - (months-1)).to_timestamp()

    portfolio = (params or {}).get("portfolio")
    if portfolio and portfolio != "All" and "Portfolio" in cmps:
        cmps = cmps[cmps["Portfolio"].eq(portfolio)]

    cmps = cmps[(cmps["_month_dt"] >= start) & (cmps["_month_dt"] <= last)]

    g = (cmps.groupby(["Portfolio","Process","RCA1"], dropna=False, as_index=False)
              .agg(complaints=("Case ID", "nunique"))
              .sort_values(["Portfolio","complaints"], ascending=[True, False]))
    if g.empty:
        g = pd.DataFrame([{"portfolio":"", "process":"", "rca1":"", "complaints":0}])
    else:
        g = g.rename(columns={"Portfolio":"portfolio", "RCA1":"rca1", "Process":"process"})
    meta = {
        "title": f"RCA1 by Portfolio × Process — last {months} months",
        "filters": {"portfolio": portfolio or "All", "months": months},
    }
    return {"dataframe": g[["portfolio","process","rca1","complaints"]], "meta": meta}
