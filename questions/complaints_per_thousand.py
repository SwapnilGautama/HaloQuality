# questions/complaints_per_thousand.py
from __future__ import annotations
import re
import pandas as pd

TITLE = "Complaints per 1,000 cases"

# --------------------------
# Small internal helpers
# --------------------------
def _norm(name: str) -> str:
    return re.sub(r'[^a-z0-9]+', '', str(name).strip().lower())

def _pick_col(df: pd.DataFrame, candidates=None, regex: str | None = None) -> str | None:
    """
    Pick a column from df using case/space-insensitive matching.
    - candidates: list of names to try (leniently matched)
    - regex: optional regex fallback
    """
    if candidates is None:
        candidates = []
    norm_map = {_norm(c): c for c in df.columns}
    # try provided candidates first
    for c in candidates:
        key = _norm(c)
        if key in norm_map:
            return norm_map[key]
    # fallback to regex
    if regex:
        pat = re.compile(regex, re.I)
        for c in df.columns:
            if pat.search(str(c)):
                return c
    return None

def _ensure_datetime(series: pd.Series) -> pd.Series:
    # dayfirst=True handles DD/MM/YY formats (e.g., complaints date)
    return pd.to_datetime(series, errors="coerce", dayfirst=True, infer_datetime_format=True)

def _portfolio_selector(series: pd.Series, wanted: str) -> pd.Series:
    s = series.fillna("").astype(str)
    w = (wanted or "").strip()
    # Exact match first
    exact = s.str.casefold().eq(w.casefold())
    if w and exact.any():
        return exact
    # Then a word-boundary "contains"
    return s.str.contains(rf"\b{re.escape(w)}\b", case=False, na=False) if w else pd.Series([True]*len(s), index=s.index)

def _month_bounds(start_month: str | None, end_month: str | None):
    """
    Build inclusive month window [start, end].
    Params can be 'YYYY-MM' or a date string; we normalize to month start/end.
    Defaults to the last 3 months if not provided.
    """
    if start_month and end_month:
        start = pd.to_datetime(start_month, errors="coerce")
        end = pd.to_datetime(end_month, errors="coerce")
    else:
        end = pd.Timestamp.today().normalize() + pd.offsets.MonthEnd(0)
        start = (end - pd.offsets.MonthBegin(3)) + pd.offsets.MonthBegin(1)

    start = (start.normalize() if pd.notna(start) else pd.Timestamp.today()).replace(day=1)
    end = (end.normalize() if pd.notna(end) else pd.Timestamp.today()) + pd.offsets.MonthEnd(0)
    return start, end

# --------------------------
# Main entry point
# --------------------------
def run(store, params=None, user_text: str | None = None):
    """
    Inputs expected in store:
      store["cases"]       : DF with (Case ID, Portfolio, Process/Process Name, and a date column like 'Create Date' or 'Report Date')
      store["complaints"]  : DF with ('Original Process Affected Case ID' -> Case ID link,
                                      'Parent Case Type' (process),
                                      'Date Complaint Received - DD/MM/YY' (date))

    Params (parsed already by app/semantic router):
      - portfolio (e.g., "London")
      - start_month, end_month (optional; inclusive window)
    """
    params = params or {}
    portfolio_val = (params.get("portfolio") or params.get("site") or "").strip()

    # Month window
    start, end = _month_bounds(params.get("start_month"), params.get("end_month"))

    # --- Load dataframes
    cases: pd.DataFrame = store["cases"]
    complaints: pd.DataFrame = store["complaints"]

    # --- Column selection: CASES
    case_id_col   = _pick_col(cases, ["Case ID", "CaseID", "Case_Id", "Original Case ID"], regex=r"\bcase\b.*\bid\b")
    portfolio_col = _pick_col(cases, ["Portfolio"])
    case_proc_col = _pick_col(cases, ["Process", "Process Name"], regex=r"\bprocess\b")
    case_date_col = _pick_col(
        cases,
        ["Create Date", "Report Date", "Start Date", "Report_Date"],
        regex=r"(create|report|start)[ _-]*date",
    )

    if not case_id_col or not portfolio_col or not case_proc_col or not case_date_col:
        return (
            TITLE,
            pd.DataFrame(),
            f"Missing columns in cases. Found: id={case_id_col}, portfolio={portfolio_col}, process={case_proc_col}, date={case_date_col}",
        )

    c = cases.copy()
    c[case_date_col] = _ensure_datetime(c[case_date_col])
    c["month"] = c[case_date_col].dt.to_period("M").dt.to_timestamp()

    mask = (c[case_date_col] >= start) & (c[case_date_col] <= end)
    if portfolio_val:
        mask &= _portfolio_selector(c[portfolio_col], portfolio_val)

    cases_f = c.loc[mask, [case_id_col, portfolio_col, case_proc_col, "month"]].dropna(subset=["month"])

    if cases_f.empty:
        return (
            TITLE,
            pd.DataFrame(),
            f"No cases after applying filters (portfolio='{portfolio_val}' months={start:%b %Y}–{end:%b %Y}).",
        )

    cases_by = (
        cases_f.groupby(["month", case_proc_col], dropna=False, as_index=False)
               .size()
               .rename(columns={"size": "cases", case_proc_col: "process"})
    )

    # --- Column selection: COMPLAINTS
    comp_id_col = _pick_col(
        complaints,
        ["Original Process Affected Case ID", "Original Case ID", "Case ID"],
        regex=r"(original.*affected.*case.*id)|(original.*case.*id)|(^case.*id$)",
    )
    comp_proc_col = _pick_col(complaints, ["Parent Case Type", "Process Name"], regex=r"\b(parent)?\s*case\s*type\b|\bprocess\b")
    # specifically handle column "Date Complaint Received - DD/MM/YY"
    comp_date_col = _pick_col(
        complaints,
        ["Date Complaint Received - DD/MM/YY", "Date Complaint Received", "Date Received"],
        regex=r"(date).*?(complaint).*?(received)|(^date.*received$)",
    )

    if not comp_id_col or not comp_proc_col or not comp_date_col:
        # Return cases with zero complaints rather than error hard-stop
        out = cases_by.copy()
        out["complaints"] = 0
        out["per_1000"] = 0.0
        out["_month"] = out["month"].dt.strftime("%b %y")
        out = out[["_month", "process", "cases", "complaints", "per_1000"]].sort_values(["_month", "process"])
        return (
            TITLE,
            out,
            f"Missing columns in complaints. Found: id={comp_id_col}, process={comp_proc_col}, date={comp_date_col}",
        )

    comp = complaints[[comp_id_col, comp_proc_col, comp_date_col]].copy()
    comp[comp_date_col] = _ensure_datetime(comp[comp_date_col])
    comp = comp[(comp[comp_date_col] >= start) & (comp[comp_date_col] <= end)]
    comp = comp.rename(columns={comp_proc_col: "process", comp_id_col: case_id_col})
    comp["month"] = comp[comp_date_col].dt.to_period("M").dt.to_timestamp()

    # Bring portfolio onto complaints via Case ID, then filter to the requested portfolio
    comp = comp.merge(cases[[case_id_col, portfolio_col]], on=case_id_col, how="left")
    if portfolio_val:
        comp = comp[_portfolio_selector(comp[portfolio_col], portfolio_val)]

    complaints_by = (
        comp.groupby(["month", "process"], dropna=False, as_index=False)
            .size()
            .rename(columns={"size": "complaints"})
    )

    out = cases_by.merge(complaints_by, on=["month", "process"], how="left")
    out["complaints"] = out["complaints"].fillna(0).astype(int)
    out["per_1000"] = (out["complaints"] / out["cases"]).fillna(0) * 1000

    out["_month"] = out["month"].dt.strftime("%b %y")
    out = out[["_month", "process", "cases", "complaints", "per_1000"]].sort_values(["_month", "process"])

    return TITLE, out, None
