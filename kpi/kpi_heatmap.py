
# kpi/kpi_heatmap.py — KPI 8: Complaint Heatmap (Reason × Dimension), optional MoM compare
import pandas as pd
import numpy as np
import re
from typing import List, Optional, Tuple

# Reuse mapping/priorities from KPI 2 if available
try:
    from kpi.kpi_reason_mix import CATEGORY_RULES as _CATEGORY_RULES
    from kpi.kpi_reason_mix import DEFAULT_REASON_SOURCES as _DEFAULT_SOURCES
except Exception:
    # Fallback definitions (kept consistent with KPI 2 defaults)
    _DEFAULT_SOURCES = [
        "Complaint Reason - Why is the member complaining ? ",
        "Current Activity Reason",
        "Root Cause",
        "Process Category",
        "Event Type"
    ]
    _CATEGORY_RULES = [
        ("Delay", re.compile(r"\b(delay|late|timescale|turn\s*around|tat|await|waiting|hold up)\b", re.I)),
        ("Communication", re.compile(r"\b(communicat|letter|email|mail|call|phone|contact|clarit|explain|response)\b", re.I)),
        ("Incorrect/Incomplete Information", re.compile(r"\b(incorrect|incomplete|unclear|wrong|error|mismatch|inaccura|typo|missing)\b", re.I)),
        ("System/Portal", re.compile(r"\b(system|portal|website|login|access|bug|crash|outage|best|bizflow|it\s)\b", re.I)),
        ("Procedure/Policy", re.compile(r"\b(procedure|process|rule|requirement|policy|sop|compliance)\b", re.I)),
        ("Scheme/Benefit", re.compile(r"\b(overpayment|benefit|pension\s*increase|value|calculation|quote|estimate|payment|transfer|contribution)\b", re.I)),
        ("Dispute", re.compile(r"\b(dispute|client|trustee|complainant)\b", re.I)),
        ("Other", re.compile(r".+", re.I)),
    ]

def _first_non_empty(row: pd.Series, fields: List[str]) -> Optional[str]:
    for f in fields:
        if f in row.index:
            v = row[f]
            if pd.isna(v):
                continue
            s = str(v).strip()
            if s and s.lower() not in ("nan", "none", "na", "null"):
                return s
    return None

def _map_reason(txt: Optional[str]) -> str:
    if txt is None:
        return "Unknown"
    for label, pat in _CATEGORY_RULES:
        if pat.search(txt):
            return label
    return "Unknown"

def _prev_month(month: str) -> str:
    p = pd.Period(month, freq="M")
    return (p - 1).strftime("%Y-%m")

def _aggregate(complaints_df: pd.DataFrame, month: str, rows: List[str], sources: List[str], include_unknown: bool):
    c = complaints_df[complaints_df["month"] == month].copy()
    if c.empty:
        return pd.DataFrame(columns=rows + ["Reason","Count","Row_Total","Col_Total","Grand_Total"])
    # ensure rows exist
    for col in rows:
        if col not in c.columns:
            c[col] = np.nan
    # extract & map
    c["__txt__"] = c.apply(lambda r: _first_non_empty(r, sources), axis=1)
    c["Reason"] = c["__txt__"].map(_map_reason)
    if not include_unknown:
        c = c[c["Reason"] != "Unknown"]
    # counts
    agg = c.groupby(rows + ["Reason"], dropna=False).size().reset_index(name="Count")
    row_tot = agg.groupby(rows, dropna=False)["Count"].sum().reset_index(name="Row_Total")
    col_tot = agg.groupby(["Reason"], dropna=False)["Count"].sum().reset_index(name="Col_Total")
    grand = agg["Count"].sum()
    out = (agg
           .merge(row_tot, on=rows, how="left")
           .merge(col_tot, on="Reason", how="left"))
    out["Grand_Total"] = grand
    return out

def complaint_heatmap(
    complaints_df: pd.DataFrame,
    month: str,
    rows_dim: List[str],
    source_cols: Optional[List[str]] = None,
    normalize: str = "row",             # "none"|"row"|"col"|"overall"
    include_unknown: bool = False,
    top_n_rows: int = 50,
    min_count: int = 1,
    compare_prev: bool = False
) -> Tuple[pd.DataFrame, str]:
    """
    Produce a tidy heatmap table for Reason × rows_dim.
    normalize:
      - "none": Value = Count
      - "row":  Value = Count / Row_Total * 100
      - "col":  Value = Count / Col_Total * 100
      - "overall": Value = Count / Grand_Total * 100
    Returns (DataFrame, prev_month_string)
    Columns: rows_dim..., Reason, Count, Value[, Prev_Count, Prev_Value, Delta]
    """
    if not rows_dim:
        raise ValueError("rows_dim must contain at least one column.")
    sources = source_cols if (source_cols and len(source_cols) > 0) else _DEFAULT_SOURCES
    prev_m = _prev_month(month)

    curr = _aggregate(complaints_df, month, rows_dim, sources, include_unknown)
    if curr.empty:
        return curr, prev_m

    # Apply min_count
    curr = curr[curr["Count"] >= int(min_count)]

    # Compute Value
    norm = (normalize or "row").lower()
    if norm not in {"none","row","col","overall"}:
        raise ValueError("normalize must be one of: none, row, col, overall")

    if norm == "none":
        curr["Value"] = curr["Count"].astype(float)
    elif norm == "row":
        curr["Value"] = (curr["Count"] / curr["Row_Total"] * 100.0).where(curr["Row_Total"] > 0)
    elif norm == "col":
        curr["Value"] = (curr["Count"] / curr["Col_Total"] * 100.0).where(curr["Col_Total"] > 0)
    else:  # overall
        gt = curr["Grand_Total"].iloc[0] if not curr.empty else 0
        curr["Value"] = (curr["Count"] / gt * 100.0) if gt > 0 else np.nan

    # Limit top rows by row total
    row_totals = (curr.groupby(rows_dim, dropna=False)["Count"].sum()
                      .sort_values(ascending=False)
                      .reset_index(name="Row_Total_All"))
    keep_rows = set(row_totals.head(int(top_n_rows))                    .apply(lambda r: tuple(r[c] for c in rows_dim), axis=1).tolist())

    curr["__rowkey__"] = curr.apply(lambda r: tuple(r[c] for c in rows_dim), axis=1)
    curr = curr[curr["__rowkey__"].isin(keep_rows)].drop(columns="__rowkey__")

    # Compare with previous month
    if compare_prev:
        prev = _aggregate(complaints_df, prev_m, rows_dim, sources, include_unknown)
        if not prev.empty:
            # compute prev Value
            if norm == "none":
                prev["Value"] = prev["Count"].astype(float)
            elif norm == "row":
                prev["Value"] = (prev["Count"] / prev["Row_Total"] * 100.0).where(prev["Row_Total"] > 0)
            elif norm == "col":
                prev["Value"] = (prev["Count"] / prev["Col_Total"] * 100.0).where(prev["Col_Total"] > 0)
            else:
                gt = prev["Grand_Total"].iloc[0] if not prev.empty else 0
                prev["Value"] = (prev["Count"] / gt * 100.0) if gt > 0 else np.nan

            # join
            on_cols = rows_dim + ["Reason"]
            merged = curr.merge(prev[on_cols + ["Count","Value"]],
                                on=on_cols, how="left", suffixes=("","_prev"))
            merged["Prev_Count"] = merged["Count_prev"]
            merged["Prev_Value"] = merged["Value_prev"]
            merged.drop(columns=["Count_prev","Value_prev"], inplace=True)
            merged["Delta"] = merged["Value"] - merged["Prev_Value"]
            curr = merged
        else:
            # no previous month: still return curr with NaN prev/delta
            curr["Prev_Count"] = np.nan
            curr["Prev_Value"] = np.nan
            curr["Delta"] = np.nan

    # Order columns
    lead = list(rows_dim) + ["Reason"]
    tail = [c for c in ["Count","Value","Prev_Count","Prev_Value","Delta","Row_Total","Col_Total","Grand_Total"] if c in curr.columns]
    curr = curr[lead + tail]
    # Round
    for col in ["Value","Prev_Value","Delta"]:
        if col in curr.columns:
            curr[col] = curr[col].astype(float).round(1)
    return curr.reset_index(drop=True), prev_m
