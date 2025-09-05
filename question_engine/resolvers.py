# question_engine/resolvers.py
from __future__ import annotations
import pandas as pd
import plotly.express as px

def _slice_month(df: pd.DataFrame, col: str, start, end) -> pd.DataFrame:
    if start is None and end is None:
        return df
    if start is None:
        return df[df[col] <= end]
    if end is None:
        return df[df[col] >= start]
    if start > end:
        start, end = end, start
    return df[(df[col] >= start) & (df[col] <= end)]

def complaints_per_1000_by_process(
    store: dict,
    portfolio: str | None = None,
    start_month: pd.Period | None = None,
    end_month: pd.Period | None = None,
) -> dict:
    """
    Uses store['joined_summary'] produced by your join builder.
    Expected columns: Month (Period[M]), Portfolio_std, ProcessName,
                      Unique_Cases, Complaints, Complaints_per_1000
    """
    js = store.get("joined_summary")
    if js is None or js.empty:
        return {"kind": "text", "text": "No joined summary available yet."}

    df = js.copy()
    if portfolio:
        df = df[df["Portfolio_std"].str.lower() == portfolio.lower()]

    # month slice
    if "Month" in df.columns:
        df = _slice_month(df, "Month", start_month, end_month)

    # aggregate at ProcessName
    grp = (
        df.groupby(["ProcessName"], dropna=False)
          .agg(Unique_Cases=("Unique_Cases", "sum"),
               Complaints=("Complaints", "sum"))
          .reset_index()
    )
    if grp.empty:
        return {"kind": "text", "text": "No data after applying those filters."}

    grp["Complaints_per_1000"] = (grp["Complaints"] / grp["Unique_Cases"].clip(lower=1)) * 1000
    grp = grp.sort_values("Complaints_per_1000", ascending=False)

    fig = px.bar(
        grp,
        x="ProcessName",
        y="Complaints_per_1000",
        hover_data=["Unique_Cases", "Complaints"],
        title=f"Complaints per 1000 by Process — Portfolio: {portfolio or 'All'}",
    )

    return {
        "kind": "figure",
        "fig": fig,
        "df": grp.round(2),
        "caption": "Aggregated over the selected month range.",
    }

def rca1_by_portfolio_for_process(store: dict, process_name: str) -> dict:
    """
    Uses store['complaints'] with engineered 'RCA1' & standardized joins.
    Expected columns in complaints: Month, Portfolio_std, Parent_Case_Type -> mapped to ProcessName
                                   RCA1
    """
    comp = store.get("complaints")
    if comp is None or comp.empty:
        return {"kind": "text", "text": "No complaints data available."}

    # harmonize process
    df = comp.copy()
    # If a normalized ProcessName was already created in your loader for complaints, use it.
    # Otherwise map Parent_Case_Type to ProcessName for grouping:
    if "ProcessName" not in df.columns and "Parent_Case_Type" in df.columns:
        df["ProcessName"] = df["Parent_Case_Type"]

    df = df[df["ProcessName"].str.lower() == process_name.lower()] if process_name else df

    if df.empty:
        return {"kind": "text", "text": "No complaints found for that process."}

    # RCA1 share by portfolio
    # Treat each complaint row as 1 complaint unless a 'Complaints' column exists.
    if "Complaints" not in df.columns:
        df["Complaints"] = 1

    grp = (
        df.groupby(["Portfolio_std", "RCA1"], dropna=False)["Complaints"]
          .sum()
          .reset_index()
    )
    if grp.empty:
        return {"kind": "text", "text": "No RCA records after filters."}

    # percentage within each portfolio
    grp["pct"] = grp.groupby("Portfolio_std")["Complaints"].apply(
        lambda s: (s / s.sum()) * 100
    ).values

    # Plot (stacked) + table
    fig = px.bar(
        grp,
        x="Portfolio_std",
        y="pct",
        color="RCA1",
        title=f"RCA1 mix by Portfolio — Process: {process_name}",
        labels={"pct": "% of complaints"},
    )

    # Table view (wide)
    table = (
        grp.pivot_table(index="Portfolio_std", columns="RCA1", values="pct", aggfunc="sum")
          .fillna(0)
          .round(2)
          .reset_index()
    )

    return {
        "kind": "figure",
        "fig": fig,
        "df": table,
        "caption": "Percent share of RCA1 within each portfolio.",
    }
