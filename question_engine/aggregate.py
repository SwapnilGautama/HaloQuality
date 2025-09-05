# question_engine/aggregate.py
from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import pandas as pd
import plotly.express as px

def _apply_month_window(df: pd.DataFrame, month_from: Optional[str], month_to: Optional[str]) -> pd.DataFrame:
    if df is None or df.empty or "Month" not in df.columns:
        return df
    if month_from:
        df = df[df["Month"] >= month_from]
    if month_to:
        df = df[df["Month"] <= month_to]
    return df

def _apply_filters(df: pd.DataFrame, filters: Dict[str, List[str]]) -> pd.DataFrame:
    if not filters: return df
    out = df.copy()
    for col, vals in filters.items():
        if col in out.columns:
            out = out[out[col].astype(str).isin([str(v) for v in vals])]
    return out

def aggregate_generic(
    domain: str,              # 'complaints' | 'cases' | 'fpa'
    metric: str,              # canonical metric name (e.g., 'Complaints_per_1000', 'Fail_Rate', 'Unique_Cases', 'Complaints')
    group_by: List[str],      # dims to group
    store: Dict[str, pd.DataFrame],
    month_from: Optional[str] = None,
    month_to: Optional[str] = None,
    filters: Optional[Dict[str, List[str]]] = None,
) -> Tuple[pd.DataFrame, Optional[object]]:
    """
    Returns (table_df, plotly_fig). Chooses sensible default charts per metric.
    """
    df = None
    if domain == "complaints":
        # Prefers the joined summary if available in store; else raw complaints
        df = store.get("complaints_join") or store.get("complaints")
    elif domain == "cases":
        df = store.get("cases")
    elif domain == "fpa":
        df = store.get("fpa")
    if df is None or df.empty:
        return pd.DataFrame(), None

    df = _apply_month_window(df, month_from, month_to)
    df = _apply_filters(df, filters or {})

    # If using complaints join, metrics likely exist already
    value_col = metric

    # Group & aggregate
    gb_cols = [c for c in (["Month"] + group_by) if c in df.columns]
    if not gb_cols:
        gb_cols = ["Month"]
    if value_col not in df.columns:
        # fallbacks for raw domains
        if domain == "complaints" and metric == "Complaints":
            agg = df.groupby(gb_cols, dropna=False).size().reset_index(name="Complaints")
            value_col = "Complaints"
        elif domain == "cases" and metric == "Unique_Cases":
            if "Case_ID" in df.columns:
                agg = df.groupby(gb_cols, dropna=False)["Case_ID"].nunique().reset_index(name="Unique_Cases")
                value_col = "Unique_Cases"
            else:
                agg = df.groupby(gb_cols, dropna=False).size().reset_index(name="Unique_Cases")
                value_col = "Unique_Cases"
        elif domain == "fpa":
            # build reviewed/fails/fail_rate on the fly
            tmp = df.groupby(gb_cols, dropna=False).agg(
                Reviewed=("Case_ID","count") if "Case_ID" in df.columns else ("ReviewResult","count"),
                Fails=("FailFlag","sum") if "FailFlag" in df.columns else ("ReviewResult", "count")
            ).reset_index()
            tmp["Fail_Rate"] = (tmp["Fails"] / tmp["Reviewed"]).fillna(0.0)
            agg = tmp
            if metric not in agg.columns:
                metric = "Fail_Rate"
            value_col = metric
        else:
            # default count
            agg = df.groupby(gb_cols, dropna=False).size().reset_index(name=metric)
            value_col = metric
    else:
        agg = df.groupby(gb_cols, dropna=False)[value_col].sum().reset_index()

    # Chart choice
    fig = None
    if "Month" in agg.columns and len(gb_cols) <= 2:
        # simple time line
        fig = px.line(agg.sort_values("Month"), x="Month", y=value_col, color=group_by[0] if group_by else None, markers=True)
    else:
        # bar chart
        color = group_by[1] if len(group_by) > 1 else (group_by[0] if group_by else None)
        fig = px.bar(agg, x=group_by[0] if group_by else "Month", y=value_col, color=color, barmode="group")

    return agg, fig
