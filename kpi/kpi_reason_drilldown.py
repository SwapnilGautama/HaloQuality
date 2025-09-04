
# kpi/kpi_reason_drilldown.py — KPI 7: Reasons Drill-Down (free-text aware)
import re
from typing import List, Optional, Tuple, Dict
import pandas as pd
import numpy as np

REQUIRED_COMPLAINTS_COLS = ["month"]

# Default field priority (first non-empty wins)
DEFAULT_REASON_SOURCES = [
    "Complaint Reason - Why is the member complaining ? ",  # trailing space per file
    "Current Activity Reason",
    "Root Cause",
    "Process Category",
    "Event Type"
]

# Category keyword rules (first match wins)
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

# Simple sub-reason normalizer (keeps short text but reduces noise)
_PUNCT = re.compile(r"[\t\n\r\-–—_/\\]+")
_MULTISPACE = re.compile(r"\s{2,}")
def _normalize_subreason(text: str) -> str:
    s = str(text).strip()
    s = _PUNCT.sub(" ", s)
    s = _MULTISPACE.sub(" ", s)
    # keep original casing for readability, but trim length
    return s[:160]

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

def _map_category(txt: Optional[str]) -> str:
    if txt is None:
        return "Unknown"
    for label, pattern in CATEGORY_RULES:
        if pattern.search(txt):
            return label
    return "Unknown"

def reason_drilldown(
    complaints_df: pd.DataFrame,
    month: str,
    group_by: List[str],
    target_category: str,
    source_cols: Optional[List[str]] = None,
    top_n: int = 20,
    min_count: int = 3,
    include_unknown: bool = False
) -> Tuple[pd.DataFrame, str]:
    """
    Drill into a given reason category and show top free-text subreasons by group.
    - Picks first non-empty text across `source_cols`, maps to broad category via keyword rules.
    - Filters to rows in `target_category` (exact match to canonical label).
    - Aggregates by group_by + SubReason (normalized free text), producing Counts + % within group and within category.
    Returns (DataFrame, used_field_info)
    Columns: group_by + ['SubReason','Count','Percent_within_group','Percent_within_category','Group_Total','Category_Total']
    """
    _validate(complaints_df, REQUIRED_COMPLAINTS_COLS, "complaints_df")
    if not group_by:
        raise ValueError("group_by must contain at least one column name.")
    if not target_category or not isinstance(target_category, str):
        raise ValueError("target_category must be a non-empty string.")

    fields = source_cols if (source_cols and len(source_cols) > 0) else DEFAULT_REASON_SOURCES
    used_field_info = " | ".join(fields)

    c = complaints_df[complaints_df["month"] == month].copy()
    if c.empty:
        return pd.DataFrame(columns=group_by + ["SubReason","Count","Percent_within_group","Percent_within_category","Group_Total","Category_Total"]), used_field_info

    # Ensure group_by columns exist
    for col in group_by:
        if col not in c.columns:
            c[col] = np.nan

    # Extract primary text
    c["__reason_text__"] = c.apply(lambda row: _first_non_empty_text(row, fields), axis=1)
    c["Category"] = c["__reason_text__"].apply(_map_category)

    if not include_unknown:
        c = c[c["Category"] != "Unknown"]

    # Filter target category
    c = c[c["Category"].str.lower() == target_category.strip().lower()]
    if c.empty:
        return pd.DataFrame(columns=group_by + ["SubReason","Count","Percent_within_group","Percent_within_category","Group_Total","Category_Total"]), used_field_info

    # Build SubReason from text (normalized)
    c["SubReason"] = c["__reason_text__"].map(lambda x: _normalize_subreason(x) if x is not None else "Unknown")
    if not include_unknown:
        c = c[c["SubReason"].str.lower() != "unknown"]

    # Aggregate
    agg = (
        c.groupby(group_by + ["SubReason"], dropna=False)
         .size()
         .reset_index(name="Count")
    )

    group_totals = agg.groupby(group_by, dropna=False)["Count"].sum().reset_index(name="Group_Total")
    overall_category_total = agg["Count"].sum()
    out = agg.merge(group_totals, on=group_by, how="left")
    out["Percent_within_group"] = (out["Count"] / out["Group_Total"]) * 100.0
    out["Percent_within_category"] = (out["Count"] / overall_category_total * 100.0) if overall_category_total > 0 else np.nan
    out["Category_Total"] = overall_category_total

    # Rank by overall contribution within category
    overall_rank = (
        out.groupby("SubReason", dropna=False)["Count"].sum()
          .sort_values(ascending=False)
          .reset_index()
    )
    overall_rank["Overall_Rank"] = overall_rank["Count"].rank(method="first", ascending=False).astype(int)
    out = out.merge(overall_rank[["SubReason","Overall_Rank"]], on="SubReason", how="left")

    # Enforce min_count and top_n by overall rank
    out = out[out["Count"] >= int(min_count)]
    out = out.sort_values(["Overall_Rank","Percent_within_group"], ascending=[True, False])
    if top_n is not None and top_n > 0:
        # keep only rows whose SubReason overall rank <= top_n
        allowed = set(overall_rank["SubReason"].head(top_n).tolist())
        out = out[out["SubReason"].isin(allowed)]

    # Round percentages
    out["Percent_within_group"] = out["Percent_within_group"].round(1)
    out["Percent_within_category"] = out["Percent_within_category"].round(1)

    return out.reset_index(drop=True), used_field_info
