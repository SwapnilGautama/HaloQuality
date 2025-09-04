# kpis/kpi_reason_mix.py — Reason Mix% (free-text aware)

import re
from typing import List, Optional, Tuple, Dict
import pandas as pd
import numpy as np

REQUIRED_COMPLAINTS_COLS = ["month"]

# Default field priority (first non-empty wins)
DEFAULT_REASON_SOURCES = [
    "Complaint Reason - Why is the member complaining ? ",  # note: trailing space as in your file
    "Current Activity Reason",
    "Root Cause",
    "Process Category",
    "Event Type"
]

# Compile keyword rules for mapping messy text -> canonical categories
# Order matters: first match wins
CATEGORY_RULES: List[Tuple[str, re.Pattern]] = [
    ("Delay", re.compile(r"\b(delay|late|timescale|turn\s*around|tat|await|waiting|hold up)\b", re.I)),
    ("Communication", re.compile(r"\b(communicat|letter|email|mail|call|phone|contact|clarit|explain|response)\b", re.I)),
    ("Incorrect/Incomplete Information", re.compile(r"\b(incorrect|incomplete|unclear|wrong|error|mismatch|inaccura|typo|missing)\b", re.I)),
    ("System/Portal", re.compile(r"\b(system|portal|website|login|access|bug|crash|outage|best|bizflow|it\s)\b", re.I)),
    ("Procedure/Policy", re.compile(r"\b(procedure|process|rule|requirement|policy|sop|compliance)\b", re.I)),
    ("Scheme/Benefit", re.compile(r"\b(overpayment|benefit|pension\s*increase|value|calculation|quote|estimate|payment|transfer|contribution)\b", re.I)),
    ("Dispute", re.compile(r"\b(dispute|client|trustee|complainant)\b", re.I)),
    ("Other", re.compile(r".+", re.I)),  # catch-all non-empty
]

def _validate(df: pd.DataFrame, required_cols: List[str], name: str):
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name} missing required columns: {missing}")

def _first_non_empty_text(row: pd.Series, fields: List[str]) -> Optional[str]:
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
    for label, pattern in CATEGORY_RULES:
        if pattern.search(txt):
            return label
    return "Unknown"

def reason_mix_percent(
    complaints_df: pd.DataFrame,
    month: str,
    group_by: List[str],
    source_cols: Optional[List[str]] = None,
    top_n: int = 10,
    include_unknown: bool = True,
) -> Tuple[pd.DataFrame, str]:
    """
    Compute Reason Mix% from complaints (free-text aware).
    - Picks the first non-empty value across `source_cols` per row.
    - Maps text to canonical categories using keyword regex rules.
    - Groups by `group_by` + Reason, returning counts and % share within each group.

    Returns: (DataFrame, used_field_info)
      DataFrame columns: group_by + ["Reason","Count","Percent"]
      used_field_info: a string describing which fields were used in priority order
    """
    _validate(complaints_df, REQUIRED_COMPLAINTS_COLS, "complaints_df")
    if not group_by:
        raise ValueError("group_by must contain at least one column name.")

    # Filter month
    c = complaints_df[complaints_df["month"] == month].copy()
    if c.empty:
        return pd.DataFrame(columns=group_by + ["Reason","Count","Percent"]), "No data for month"

    # Ensure group_by columns exist
    for col in group_by:
        if col not in c.columns:
            c[col] = np.nan

    fields = source_cols if (source_cols and len(source_cols) > 0) else DEFAULT_REASON_SOURCES
    used_field_info = " | ".join(fields)

    # Extract the primary text reason per row (first non-empty field)
    c["__reason_text__"] = c.apply(lambda row: _first_non_empty_text(row, fields), axis=1)

    # Map to canonical buckets
    c["Reason"] = c["__reason_text__"].apply(_map_reason)

    # Optionally drop Unknown
    if not include_unknown:
        c = c[c["Reason"] != "Unknown"]

    # Aggregate
    agg = (
        c
        .groupby(group_by + ["Reason"], dropna=False)
        .size()
        .reset_index(name="Count")
    )

    # Percent within each group
    totals = agg.groupby(group_by, dropna=False)["Count"].sum().reset_index(name="__group_total__")
    out = agg.merge(totals, on=group_by, how="left")
    out["Percent"] = (out["Count"] / out["__group_total__"]) * 100.0
    out.drop(columns="__group_total__", inplace=True)

    # Sort and top_n per group
    out = out.sort_values(group_by + ["Percent"], ascending=[True]*len(group_by) + [False])

    if top_n is not None and top_n > 0:
        # keep top_n per group
        out["__rank__"] = out.groupby(group_by)["Percent"].rank(method="first", ascending=False)
        out = out[out["__rank__"] <= top_n].drop(columns="__rank__")

    # Round Percent for presentation
    out["Percent"] = out["Percent"].round(1)

    # Make sure output types are clean
    return out.reset_index(drop=True), used_field_info
