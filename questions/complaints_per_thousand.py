# questions/complaints_per_thousand.py
from __future__ import annotations

from typing import Dict, Any, Optional
import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta


# -----------------------------
# Small utilities
# -----------------------------
def _first_present(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    if df is None or df.empty:
        return None
    lower = {c.lower(): c for c in df.columns}
    for name in candidates:
        if name.lower() in lower:
            return lower[name.lower()]
    return None


def _detect_cases_date(df: pd.DataFrame) -> Optional[str]:
    """
    Broader cases date detector (we've seen many headers here).
    """
    if df is None or df.empty:
        return None

    known = [
        # very common
        "Case Created Date", "Created Date", "Creation Date",
        "Case Opened Date", "Opened Date",
        "Received Date", "Start Date", "Date",
        "Case Creation Date", "First Touch Date", "Case Start Date",
        # other systems we encountered
        "Effective Date", "Service Date", "Logged Date",
        "Policy Effective Date", "Interaction Date", "Event Date",
        "FPA Date", "FPA Received Date"
    ]
    col = _first_present(df, known)
    if col:
        return col

    # keyword based
    for c in df.columns:
        name = c.lower()
        if any(k in name for k in ["date", "created", "opened", "received", "start", "logged", "effective", "event", "interaction"]):
            return c

    # heuristic: pick column with best datetime parse ratio (>50%)
    best, best_ratio = None, 0.0
    for c in df.columns:
        try:
            dt = pd.to_datetime(df[c], errors="coerce", dayfirst=True)
            ratio = float(dt.notna().mean())
            if ratio > 0.5 and ratio > best_ratio:
                best, best_ratio = c, ratio
        except Exception:
            pass
    return best


def _monthify(series: pd.Series, *, dayfirst: bool = True) -> pd.Series:
    dt = pd.to_datetime(series, errors="coerce", dayfirst=dayfirst)
    return dt.dt.to_period("M").dt.to_timestamp()


def _norm_str(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip()


def _month_window(params: Dict[str, Any], cases: pd.DataFrame, date_col: str):
    # end = max month in data; default last 3 months
    data_max = _monthify(cases[date_col]).max()
    if pd.isna(data_max):
        data_max = pd.Timestamp(datetime.utcnow()).to_period("M").to_timestamp()

    start = params.get("start_month")
    end = params.get("end_month")
    rel = params.get("relative_months")

    if start and end:
        s = pd.to_datetime(start, errors="coerce")
        e = pd.to_datetime(end, errors="coerce")
        if not (pd.isna(s) or pd.isna(e)):
            return s.to_period("M").to_timestamp(), e.to_period("M").to_timestamp()

    if isinstance(rel, (int, float)) and rel > 0:
        e = data_max
        s = (e - relativedelta(months=int(rel) - 1)).to_period("M").to_timestamp()
        return s, e

    e = data_max
    s = (e - relativedelta(months=2)).to_period("M").to_timestamp()
    return s, e


def _safe_msg(message: str) -> Dict[str, Any]:
    # The app will show this nicely instead of a red failure box.
    return {"message": message, "dataframe": pd.DataFrame()}


# -----------------------------
# Main entrypoint
# -----------------------------
def run(store: Dict[str, Any], params: Dict[str, Any] = None, user_text: str = "") -> Dict[str, Any]:
    """
    Build complaints per 1,000 cases by month × process.

    Data requirements:
      cases:
        - date column (auto-detected)
        - 'Process Name' (or close variants)
        - optional 'Portfolio'
        - optional 'Case ID' (for joining)
      complaints:
        - 'Date Complaint Received - DD/MM/YY' (preferred; day-first)
        - 'Original Process Affected Case ID' for join (optional)
        - 'Parent Case Type' as process if join not possible (fallback)
    """
    try:
        params = params or {}

        cases = store.get("cases")
        complaints = store.get("complaints")
        if cases is None or cases.empty:
            return _safe_msg("No cases available.")
        if complaints is None or complaints.empty:
            return _safe_msg("No complaints available.")

        # ----- CASES: detect columns
        cases_date_col = _detect_cases_date(cases)
        if not cases_date_col:
            return _safe_msg("Could not find a date column in cases.")

        cases_proc_col = _first_present(cases, ["Process Name", "Process", "Parent Case Type"])
        if not cases_proc_col:
            return _safe_msg("Could not find a process column in cases (e.g. 'Process Name').")

        portfolio_col = _first_present(cases, ["Portfolio", "portfolio"])
        portfolio_val = params.get("portfolio")

        # monthify & filter cases
        cases = cases.copy()
        cases["_month"] = _monthify(cases[cases_date_col], dayfirst=True)

        start_m, end_m = _month_window(params, cases, cases_date_col)
        mask = (cases["_month"] >= start_m) & (cases["_month"] <= end_m)
        if portfolio_col and portfolio_val:
            mask &= _norm_str(cases[portfolio_col]).str.lower().eq(str(portfolio_val).strip().lower())
        cases = cases.loc[mask]

        if cases.empty:
            return _safe_msg("No cases after applying the selected filters/date window.")

        # denominator: cases by month x process
        cases_proc = (
            cases.assign(_proc=_norm_str(cases[cases_proc_col]))
                 .dropna(subset=["_month"])
                 .groupby(["_month", "_proc"], dropna=False)
                 .size()
                 .rename("cases")
                 .reset_index()
        )

        # ----- COMPLAINTS: date & keys
        complaints = complaints.copy()

        # Prefer exact complaints date column (day-first)
        comp_date_col = _first_present(complaints, ["Date Complaint Received - DD/MM/YY"])
        if comp_date_col:
            complaints["_comp_month"] = _monthify(complaints[comp_date_col], dayfirst=True)
        else:
            # fallback to best guess to keep older files working
            guess = _detect_cases_date(complaints)
            if not guess:
                return _safe_msg("Could not find a complaints date column (‘Date Complaint Received - DD/MM/YY’).")
            complaints["_comp_month"] = _monthify(complaints[guess], dayfirst=True)

        comp_case_id_col = _first_present(
            complaints,
            ["Original Process Affected Case ID", "Original Case ID", "Case ID", "Linked Case ID", "Case Ref", "Case Reference"]
        )
        comp_proc_col = _first_present(complaints, ["Parent Case Type", "Process Name", "Process"])

        if comp_proc_col:
            complaints["_comp_proc"] = _norm_str(complaints[comp_proc_col])
        else:
            complaints["_comp_proc"] = ""

        # Try to join to cases on Case ID
        join_case_id = _first_present(cases, ["Case ID", "Case Ref", "Case Reference", "Id"])

        if comp_case_id_col and join_case_id:
            lookup_cols = [join_case_id, "_month", cases_proc_col]
            if portfolio_col:
                lookup_cols.append(portfolio_col)
            cases_lookup = cases[lookup_cols].drop_duplicates(subset=[join_case_id])

            merged = complaints.merge(
                cases_lookup,
                left_on=comp_case_id_col,
                right_on=join_case_id,
                how="left",
                suffixes=("", "_case"),
            )

            # Month preference: complaints month, else case month
            merged["_final_month"] = merged["_comp_month"]
            needs_month = merged["_final_month"].isna()
            merged.loc[needs_month, "_final_month"] = merged["_month"]

            # Process preference: cases process, else complaints process
            merged["_final_proc"] = _norm_str(merged.get(cases_proc_col, ""))
            needs_proc = merged["_final_proc"].eq("").fillna(True)
            merged.loc[needs_proc, "_final_proc"] = merged["_comp_proc"]

            # portfolio filter (post-join) if requested
            if portfolio_col and portfolio_val:
                merged = merged[
                    _norm_str(merged[portfolio_col]).str.lower().eq(str(portfolio_val).strip().lower())
                ]
        else:
            # No join path → rely on complaints' own month & process
            merged = complaints.rename(
                columns={"_comp_month": "_final_month", "_comp_proc": "_final_proc"}
            )

        # Restrict to date window
        merged = merged[(merged["_final_month"] >= start_m) & (merged["_final_month"] <= end_m)]

        complaints_agg = (
            merged.dropna(subset=["_final_month"])
                  .assign(_final_proc=_norm_str(merged["_final_proc"]))
                  .groupby(["_final_month", "_final_proc"], dropna=False)
                  .size()
                  .rename("complaints")
                  .reset_index()
                  .rename(columns={"_final_month": "_month", "_final_proc": "_proc"})
        )

        # Combine cases & complaints
        out = cases_proc.merge(complaints_agg, on=["_month", "_proc"], how="outer")
        out["cases"] = out["cases"].fillna(0).astype(int)
        out["complaints"] = out["complaints"].fillna(0).astype(int)
        out["per_1000"] = out.apply(
            lambda r: (r["complaints"] / r["cases"] * 1000) if r["cases"] > 0 else 0.0, axis=1
        )

        # Present
        out = (
            out.assign(month=lambda d: d["_month"].dt.strftime("%b %y"))
               .rename(columns={"_proc": "process"})
               .loc[:, ["month", "process", "cases", "complaints", "per_1000"]]
               .sort_values(["month", "process"])
               .reset_index(drop=True)
        )

        if out.empty:
            return _safe_msg("No overlapping data between cases and complaints for the current filters.")

        title_bits = []
        if portfolio_val:
            title_bits.append(f"Portfolio: {portfolio_val}")
        title_bits.append(f"Range: {start_m.strftime('%b %y')} → {end_m.strftime('%b %y')}")
        return {"table_title": " | ".join(title_bits), "dataframe": out}

    except Exception as e:
        # Final safety net: return a message instead of throwing
        return _safe_msg(f"Failed to build Complaints per 1,000 cases. Details: {e}")
