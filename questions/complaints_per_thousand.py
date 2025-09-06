# questions/complaints_per_thousand.py
from __future__ import annotations
import re
import pandas as pd
from datetime import datetime
from typing import Dict, Any, Tuple

def _to_month(dt_like) -> pd.Series:
    return pd.to_datetime(dt_like, errors="coerce").dt.to_period("M").dt.to_timestamp()

def _proc_key(s: pd.Series) -> pd.Series:
    # normalize process labels so complaints & cases match even if minor wording differs
    x = s.astype(str).str.lower()
    # remove punctuation -> spaces
    x = x.str.replace(r"[^a-z0-9]+", " ", regex=True)
    # common tidies (extend if needed)
    x = x.str.replace(r"\b(enquiryqa|enquiry qa)\b", "enquiry", regex=True)
    x = x.str.replace(r"\s+", " ", regex=True).str.strip()
    return x

def _window_from_params(params: Dict[str, Any], fallback_last_n=3) -> Tuple[pd.Timestamp, pd.Timestamp, str]:
    if "start_month" in params and "end_month" in params:
        sm = pd.to_datetime(params["start_month"])  # normalized to YYYY-MM-01
        em = pd.to_datetime(params["end_month"])
        title = f"{sm.strftime('%b %Y')} to {em.strftime('%b %Y')}"
        return sm, em, title
    if "last_n_months" in params:
        n = int(params["last_n_months"])
        end = pd.Timestamp.today().to_period("M").to_timestamp()
        start = (end.to_period("M") - n + 1).to_timestamp()
        title = f"last {n} months"
        return start, end, title
    # default = last 3 months
    end = pd.Timestamp.today().to_period("M").to_timestamp()
    start = (end.to_period("M") - fallback_last_n + 1).to_timestamp()
    title = "last 3 months"
    return start, end, title

def run(store: Dict[str, pd.DataFrame], params: Dict[str, Any] | None = None, user_text: str = ""):
    params = params or {}
    cases = store.get("cases", pd.DataFrame()).copy()
    cmpl  = store.get("complaints", pd.DataFrame()).copy()

    notes = []
    portfolio = (params.get("portfolio") or "all").strip().lower()

    # date window
    start_m, end_m, range_title = _window_from_params(params)

    # safety: required columns
    need_cases = {"id", "portfolio", "process", "_month_dt"}
    miss_cases = sorted(list(need_cases - set(cases.columns)))
    if miss_cases:
        return ("Complaints per 1,000 cases", pd.DataFrame(), [f"Missing columns in cases: {miss_cases}"])

    need_cmpl = {"portfolio", "process", "_month_dt"}
    miss_cmpl = sorted(list(need_cmpl - set(cmpl.columns)))
    if miss_cmpl:
        return ("Complaints per 1,000 cases", pd.DataFrame(), [f"Missing columns in complaints: {miss_cmpl}"])

    # normalize
    cases["portfolio"] = cases["portfolio"].astype(str).str.strip().str.lower()
    cmpl["portfolio"]  = cmpl["portfolio"].astype(str).str.strip().str.lower()

    cases["_month_dt"] = _to_month(cases["_month_dt"])
    cmpl["_month_dt"]  = _to_month(cmpl["_month_dt"])

    cases["proc_key"] = _proc_key(cases["process"])
    cmpl["proc_key"]  = _proc_key(cmpl["process"])

    # filter by portfolio (if supplied)
    if portfolio != "all":
        cases = cases[cases["portfolio"] == portfolio]
        cmpl  = cmpl[cmpl["portfolio"] == portfolio]

    # filter by month window
    cases = cases[(cases["_month_dt"] >= start_m) & (cases["_month_dt"] <= end_m)]
    cmpl  = cmpl[(cmpl["_month_dt"] >= start_m) & (cmpl["_month_dt"] <= end_m)]

    # diagnostics
    notes.append(
        f"Filtered cases rows: {len(cases)} | complaints rows: {len(cmpl)} "
        f"| months: {start_m.strftime('%b %Y')} → {end_m.strftime('%b %Y')} | portfolio: {portfolio}"
    )
    if len(cases):
        topc = cases["proc_key"].value_counts().head(5).to_dict()
        notes.append(f"Top case processes (normalized): {topc}")
    if len(cmpl):
        topm = cmpl["proc_key"].value_counts().head(5).to_dict()
        notes.append(f"Top complaint processes (normalized): {topm}")

    # aggregate
    den = (cases.groupby(["proc_key", "_month_dt"], dropna=False)["id"]
                 .count()
                 .rename("cases")
                 .reset_index())
    num = (cmpl.groupby(["proc_key", "_month_dt"], dropna=False)
                .size()
                .rename("complaints")
                .reset_index())

    out = pd.merge(num, den, on=["proc_key", "_month_dt"], how="outer")
    out["cases"] = out["cases"].fillna(0).astype("Int64")
    out["complaints"] = out["complaints"].fillna(0).astype("Int64")
    out["per_1000"] = (out["complaints"] / out["cases"].replace({0: pd.NA})) * 1000

    # if absolutely no overlap on (proc_key, month), warn explicitly
    if out[["complaints", "cases"]].sum(numeric_only=True).sum() == 0:
        notes.append("No overlapping data for cases and complaints.")
        return (f"Complaints per 1,000 cases — {portfolio.title()} — {range_title}", pd.DataFrame(), notes)

    # prettier columns
    out = out.sort_values(["_month_dt", "proc_key"])
    out = out.rename(columns={"proc_key": "process_key", "_month_dt": "month"})

    title = f"Complaints per 1,000 cases — {portfolio.title()} — {range_title}"
    return title, out, notes
