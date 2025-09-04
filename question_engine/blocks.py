from typing import Dict, Any, List
import numpy as np
import pandas as pd

def _fmt(value, kind="int"):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if kind == "int":
        return int(round(float(value)))
    if kind == "rate1":
        return round(float(value), 1)
    if kind == "nps":
        return round(float(value), 1)
    if kind == "float1":
        return round(float(value), 1)
    return value

def _signed(value):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    v = float(value)
    sign = "+" if v >= 0 else "−"
    return f"{sign}{abs(round(v,1))}"

def make_metric_cards(mom_df: pd.DataFrame) -> Dict[str, Any]:
    if mom_df is None or mom_df.empty:
        return {
            "rate": None, "rate_delta": None,
            "complaints": None, "complaints_delta": None,
            "cases": None, "cases_delta": None,
            "nps": None, "nps_delta": None
        }

    sums = mom_df[["Complaints","Unique_Cases","Complaints_prev","Unique_Cases_prev"]].apply(
        pd.to_numeric, errors="coerce"
    ).fillna(0).sum()

    rate = (sums["Complaints"] / sums["Unique_Cases"] * 1000.0) if sums["Unique_Cases"] > 0 else np.nan
    rate_prev = (sums["Complaints_prev"] / sums["Unique_Cases_prev"] * 1000.0) if sums["Unique_Cases_prev"] > 0 else np.nan
    rate_delta = rate - rate_prev if not np.isnan(rate) and not np.isnan(rate_prev) else np.nan

    nps = None
    nps_delta = None
    if "NPS" in mom_df.columns:
        if "Total_Responses" in mom_df.columns and mom_df["Total_Responses"].notna().any():
            w = pd.to_numeric(mom_df["Total_Responses"], errors="coerce").fillna(0)
            n = pd.to_numeric(mom_df["NPS"], errors="coerce")
            nps = (n * w).sum() / w.sum() if w.sum() > 0 else float("nan")
        else:
            nps = pd.to_numeric(mom_df["NPS"], errors="coerce").mean()

        if "NPS_prev" in mom_df.columns:
            if "Total_Responses_prev" in mom_df.columns and mom_df["Total_Responses_prev"].notna().any():
                w2 = pd.to_numeric(mom_df["Total_Responses_prev"], errors="coerce").fillna(0)
                n2 = pd.to_numeric(mom_df["NPS_prev"], errors="coerce")
                nps_prev = (n2 * w2).sum() / w2.sum() if w2.sum() > 0 else float("nan")
            else:
                nps_prev = pd.to_numeric(mom_df["NPS_prev"], errors="coerce").mean()
            nps_delta = nps - nps_prev if not np.isnan(nps) and not np.isnan(nps_prev) else np.nan

    return {
        "rate": _fmt(rate, "rate1"),
        "rate_delta": _fmt(rate_delta, "float1"),
        "complaints": _fmt(mom_df["Complaints"].sum(), "int"),
        "complaints_delta": _fmt(mom_df.get("Complaints_delta", pd.Series([np.nan])).sum(), "float1"),
        "cases": _fmt(mom_df["Unique_Cases"].sum(), "int"),
        "cases_delta": _fmt(mom_df.get("Unique_Cases_delta", pd.Series([np.nan])).sum(), "float1"),
        "nps": _fmt(nps, "nps"),
        "nps_delta": _fmt(nps_delta, "float1")
    }

def table_from_df(df: pd.DataFrame, columns: List[str] = None) -> Dict[str, Any]:
    if df is None or df.empty:
        return {"columns": columns or [], "rows": []}
    if columns:
        safe_cols = [c for c in columns if c in df.columns]
        df = df[safe_cols]
    return {"columns": list(df.columns), "rows": df.to_dict(orient="records")}

def chart_spec(name: str, chart_type: str, x: str, y: str, data_ref: str, sort: str = None) -> Dict[str, Any]:
    spec = {"name": name, "type": chart_type, "x": x, "y": y, "dataRef": data_ref}
    if sort:
        spec["sort"] = sort
    return spec

def signed(v):
    return _signed(v)
