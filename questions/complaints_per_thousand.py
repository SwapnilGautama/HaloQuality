# questions/complaints_per_thousand.py
from __future__ import annotations
import pandas as pd

def _lower(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower()

def _parse_month_range(params: dict) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    """
    Returns (start_month_dt, end_month_dt) inclusive (both month starts).
    Accepts params like {'start_month': '2025-06-01', 'end_month': '2025-08-01'} or last_n_months.
    """
    start = params.get("start_month")
    end = params.get("end_month")
    if start and end:
        s = pd.to_datetime(start, errors="coerce")
        e = pd.to_datetime(end, errors="coerce")
        # normalize to month-start
        return s.to_period("M").to_timestamp(), e.to_period("M").to_timestamp()
    # Fallback: last_n_months
    n = params.get("last_n_months")
    if n:
        try:
            n = int(n)
        except Exception:
            n = 3
        e = pd.Timestamp.today().to_period("M").to_timestamp()
        s = (e - pd.offsets.MonthBegin(n-1))
        return s, e
    return None, None

def run(store: dict, params: dict, user_text: str):
    """
    Compute complaints per 1,000 cases by process for a portfolio over a month range.
    - Cases date: Create Date (normalized to _month_dt)
    - Complaints date: Date Complaint Received - DD/MM/YY -> _month_dt
      or fallback 'Month' text -> assume year handled in data_store
    - Join key: (process, portfolio, month) — no per-case ID join required
    """
    cases = store.get("cases", pd.DataFrame()).copy()
    cmpl = store.get("complaints", pd.DataFrame()).copy()

    notes = []

    if cases.empty:
        notes.append("Cases data is empty.")
    if cmpl.empty:
        notes.append("Complaints data is empty.")

    # Portfolio from params or default to 'London'
    portfolio = (params or {}).get("portfolio")
    if not portfolio:
        portfolio = "London"
    portfolio_norm = str(portfolio).strip().lower()

    # Month range
    s_month, e_month = _parse_month_range(params or {})
    if s_month is None or e_month is None:
        # If not provided, try to infer from data: use min/max overlap
        if cases["_month_dt"].notna().any() and cmpl["_month_dt"].notna().any():
            s_month = max(cases["_month_dt"].min(), cmpl["_month_dt"].min())
            e_month = min(cases["_month_dt"].max(), cmpl["_month_dt"].max())
        else:
            notes.append("Could not infer month range; please specify.")
            s_month = None
            e_month = None

    title = f"Complaints per 1,000 cases"
    if s_month is not None and e_month is not None:
        title += f" — {portfolio} — {s_month.strftime('%b %Y')} to {e_month.strftime('%b %Y')}"

    # Filter by portfolio + month range
    if "portfolio" in cases:
        cases["portfolio"] = _lower(cases["portfolio"])
    if "portfolio" in cmpl:
        cmpl["portfolio"] = _lower(cmpl["portfolio"])

    if s_month is not None and e_month is not None:
        cases = cases[(cases["_month_dt"] >= s_month) & (cases["_month_dt"] <= e_month)]
        cmpl = cmpl[(cmpl["_month_dt"] >= s_month) & (cmpl["_month_dt"] <= e_month)]

    cases = cases[cases["portfolio"] == portfolio_norm]
    cmpl = cmpl[cmpl["portfolio"] == portfolio_norm]

    # Sanity messages
    if cases.empty or cmpl.empty:
        notes.append("No overlapping data for cases and complaints.")
        return title, pd.DataFrame(), notes

    # Group by process + month
    if "process" not in cases or "process" not in cmpl:
        notes.append("Missing process columns (cases['process'] or complaints['process']).")
        return title, pd.DataFrame(), notes

    cases_g = (
        cases
        .dropna(subset=["process", "_month_dt"])
        .groupby(["process", "_month_dt"], as_index=False)
        .agg(cases=("id", "count"))
    )

    cmpl_g = (
        cmpl
        .dropna(subset=["process", "_month_dt"])
        .groupby(["process", "_month_dt"], as_index=False)
        .agg(complaints=("process", "count"))
    )

    out = pd.merge(
        cases_g,
        cmpl_g,
        on=["process", "_month_dt"],
        how="outer",
        validate="one_to_one"
    ).fillna({"cases": 0, "complaints": 0})

    # Per-1000
    out["per_1000"] = out.apply(
        lambda r: (r["complaints"] * 1000.0 / r["cases"]) if r["cases"] else 0.0,
        axis=1
    )

    # Friendly month label
    out["month"] = out["_month_dt"].dt.strftime("%b %y")
    out = out[["month", "process", "cases", "complaints", "per_1000"]].sort_values(["month", "process"]).reset_index(drop=True)

    return title, out, notes
