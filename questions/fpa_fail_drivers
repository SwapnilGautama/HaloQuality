# questions/fpa_fail_drivers.py
"""
FPA Fail Drivers
----------------
Analyze the biggest drivers (root causes) of First Pass Accuracy (FPA) failures.

- Reads FPA records from the in-memory store (expects `store["fpa"]`).
- Uses labeled RCA columns if they exist (from core.fpa_labeller.label_fpa_comments);
  otherwise applies a small internal fallback labeler.
- Supports natural filters in the user query (portfolio, process, scheme, team,
  manager, location, month range).
- Shows Top N drivers (bar chart) + a detailed table (count, % of fails).
- Returns a dict with the dataframe and plotly figure, *and* renders to Streamlit
  if `st` is provided.

Signature (expected by your app/router):
    run(store, query=None, st=None, top_n=10)
"""

from __future__ import annotations

import re
from typing import Dict, Tuple, Optional, List

import pandas as pd
import numpy as np

# Plotly for visualization
import plotly.express as px

# Optional: shared utils if present
try:
    from questions._utils import ensure_month_col, month_to_label  # type: ignore
except Exception:
    ensure_month_col = None
    month_to_label = None

# Optional: your labeler (preferred)
try:
    from core.fpa_labeller import label_fpa_comments  # type: ignore
except Exception:
    label_fpa_comments = None


# -----------------------------
# Helpers
# -----------------------------

_MONTH_RX = re.compile(
    r"(?P<m1>(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*)[\s\-\/]?(?P<y1>\d{2,4})"
    r"|(?P<y2>\d{4})[\-\/](?P<m2>\d{1,2})", re.IGNORECASE
)

def _parse_months(text: str) -> Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]:
    """
    Try to find one or two months in free text.
    Returns (start_month, end_month) as MonthBegin timestamps.
    """
    if not text:
        return None, None

    hits: List[pd.Timestamp] = []
    for m in _MONTH_RX.finditer(text):
        if m.group("m1") and m.group("y1"):
            mon = m.group("m1")[:3].title()
            y = m.group("y1")
            if len(y) == 2:
                y = "20" + y
            try:
                hits.append(pd.Timestamp(f"{mon} {y}").to_period("M").to_timestamp())
            except Exception:
                pass
        elif m.group("y2") and m.group("m2"):
            y = m.group("y2")
            mm = m.group("m2").zfill(2)
            try:
                hits.append(pd.Timestamp(f"{y}-{mm}-01").to_period("M").to_timestamp())
            except Exception:
                pass

    if not hits:
        return None, None
    if len(hits) == 1:
        return hits[0], hits[0]
    # pick the first two mentions in order
    return min(hits[0], hits[1]), max(hits[0], hits[1])


def _extract_filter(text: str, key: str, aliases: List[str]) -> Optional[str]:
    """
    Look for 'key <value>' or alias phrase '<alias> <value>'.
    Very lightweight; intended to catch simple prompts like:
      "fpa fail drivers for portfolio London, process Member Enquiry, Jun 2025 to Aug 2025"
    """
    if not text:
        return None
    rx = re.compile(rf"(?:\b{key}\b|{'|'.join(map(re.escape, aliases))})\s*[:\-]?\s*([^\.,;|]+)", re.IGNORECASE)
    m = rx.search(text)
    if m:
        return m.group(1).strip()
    return None


def _to_month_col(df: pd.DataFrame) -> pd.Series:
    """
    Ensure a month column exists. Prefer existing 'month' if present; else
    derive from likely date columns in FPA.
    """
    # If user has a helper, use it.
    if ensure_month_col:
        try:
            return ensure_month_col(df)
        except Exception:
            pass

    # Common date columns to try, in order
    candidates = [
        "month",
        "Create Date",
        "Create_Date",
        "Review Date",
        "Review_Date",
        "Date",
        "CreatedOn",
    ]
    dt = None
    for c in candidates:
        if c in df.columns:
            try:
                dt = pd.to_datetime(df[c], errors="coerce")
                break
            except Exception:
                continue
    if dt is None:
        # synthesize a dummy 'month' if nothing is usable
        return pd.Series(pd.NaT, index=df.index)

    return dt.dt.to_period("M").dt.to_timestamp()


def _fallback_labeler(text: str) -> str:
    """
    Minimal keyword-based labeler used only if your core labeling
    isn't available. You can extend these keywords later.
    """
    if not isinstance(text, str):
        return "Unclassified"

    t = text.lower()

    # examples; adjust as per your data patterns
    if any(k in t for k in ["missing", "not provided", "no doc", "document", "doc not"]):
        return "Missing documentation"
    if any(k in t for k in ["wrong", "incorrect", "mismatch", "typo", "error entry"]):
        return "Incorrect data entry"
    if any(k in t for k in ["delay", "late", "sla", "breach"]):
        return "Delay / SLA"
    if any(k in t for k in ["system", "tool", "access", "portal", "technical"]):
        return "System/tool issue"
    if any(k in t for k in ["process", "procedure", "step", "followed"]):
        return "Process not followed"
    return "Unclassified"


def _label_fail_rca(fail_df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure fail_df has a primary RCA column: 'FPA_RCA1'.
    Preferred: use your `core.fpa_labeller.label_fpa_comments`.
    Fallback: minimal keyword labeler on 'Case Comment' (or text-like fields).
    """
    # Already labeled?
    for col in ["FPA_RCA1", "RCA1", "rca1", "fpa_rca1"]:
        if col in fail_df.columns:
            fail_df["FPA_RCA1"] = fail_df[col].astype(str).str.strip().replace({"": "Unclassified"})
            return fail_df

    # Try your labeler if available
    if label_fpa_comments is not None:
        try:
            lab = label_fpa_comments(fail_df)  # should return df with 'FPA_RCA1' or similar
            if "FPA_RCA1" not in lab.columns:
                # If your labeler uses a different name, normalize
                for alt in ["RCA1", "rca1", "label_1", "topic1", "cluster1"]:
                    if alt in lab.columns:
                        lab = lab.rename(columns={alt: "FPA_RCA1"})
                        break
            lab["FPA_RCA1"] = lab["FPA_RCA1"].astype(str).str.strip().replace({"": "Unclassified"})
            return lab
        except Exception:
            pass

    # Fallback: one-column keyword label from 'Case Comment' / 'Comments'
    comment_col = None
    for c in ["Case Comment", "Comments", "Comment", "Case_Comment", "Review Comments", "Review_Comments"]:
        if c in fail_df.columns:
            comment_col = c
            break

    if comment_col is None:
        # Last resort: create a neutral label
        fail_df["FPA_RCA1"] = "Unclassified"
        return fail_df

    fail_df = fail_df.copy()
    fail_df["FPA_RCA1"] = fail_df[comment_col].apply(_fallback_labeler).astype(str)
    return fail_df


def _apply_free_text_filters(df: pd.DataFrame, text: Optional[str]) -> pd.DataFrame:
    """
    Apply simple free-text filters from the prompt.
    Supported keys: portfolio, process, scheme, team, manager, location, month range
    """
    if not text:
        return df

    # Textual filters
    portfolio = _extract_filter(text, "portfolio", ["portfolio"])
    process   = _extract_filter(text, "process", ["process name", "process"])
    scheme    = _extract_filter(text, "scheme", ["scheme"])
    team      = _extract_filter(text, "team", ["team name", "team"])
    manager   = _extract_filter(text, "manager", ["team manager", "manager"])
    location  = _extract_filter(text, "location", ["location", "site"])

    # Month range
    m_start, m_end = _parse_months(text)

    # Best-effort column mapping
    colmap = {
        "Portfolio_std": ["Portfolio_std", "Portfolio", "portfolio", "PORTFOLIO"],
        "Process Name": ["Process Name", "Process_Name", "process", "process name", "Process"],
        "Scheme": ["Scheme", "scheme"],
        "Team": ["Team", "team", "Team Name", "Team_Name"],
        "Team Manager": ["Team Manager", "Manager", "Team_Manager", "manager"],
        "Location": ["Location", "Site", "Office", "location"],
    }

    def first_col(options: List[str]) -> Optional[str]:
        for c in options:
            if c in df.columns:
                return c
        return None

    out = df
    for val, key in [
        (portfolio, "Portfolio_std"),
        (process,   "Process Name"),
        (scheme,    "Scheme"),
        (team,      "Team"),
        (manager,   "Team Manager"),
        (location,  "Location"),
    ]:
        if val:
            col = first_col(colmap[key])
            if col:
                out = out[out[col].astype(str).str.contains(re.escape(val), case=False, na=False)]

    # Month filter
    if m_start is not None or m_end is not None:
        months = _to_month_col(out)
        out = out.assign(__month=months)
        if m_start is not None:
            out = out[out["__month"] >= m_start]
        if m_end is not None:
            out = out[out["__month"] <= m_end]
        out = out.drop(columns="__month", errors="ignore")

    return out


# -----------------------------
# Main entrypoint
# -----------------------------

def run(store: Dict, query: Optional[str] = None, st=None, top_n: int = 10) -> Dict[str, object]:
    """
    Render & return top FPA fail drivers.
    - store: expects store["fpa"] as a pandas DataFrame
    - query: free text containing optional filters
    - st:    Optional Streamlit module for rendering
    - top_n: number of top drivers to display

    Returns: {"table": DataFrame, "fig": PlotlyFigure}
    """
    if "fpa" not in store or not isinstance(store["fpa"], pd.DataFrame):
        msg = "FPA dataframe not found in store['fpa']."
        if st is not None:
            st.error(msg)
        return {"table": pd.DataFrame(), "fig": None}

    fpa = store["fpa"].copy()

    # Normalize month column (no-op if you already have one)
    fpa["_month"] = _to_month_col(fpa)

    # Fail-only subset
    review_col = None
    for c in ["Review Result", "Review_Result", "Result", "Status", "review result"]:
        if c in fpa.columns:
            review_col = c
            break
    if review_col is None:
        # no explicit review result — assume all are fails (conservative)
        fail_df = fpa.copy()
    else:
        fail_df = fpa[fpa[review_col].astype(str).str.lower().isin(["fail", "failed", "f"])]  # type: ignore

    if fail_df.empty:
        if st is not None:
            st.info("No FPA failures found for the current data/filters.")
        return {"table": pd.DataFrame(), "fig": None}

    # Apply free-text filters
    fail_df = _apply_free_text_filters(fail_df, query)

    if fail_df.empty:
        if st is not None:
            st.warning("No FPA failures after applying filters in your question.")
        return {"table": pd.DataFrame(), "fig": None}

    # Ensure RCA labels
    fail_df = _label_fail_rca(fail_df)

    # Aggregate Top N drivers by RCA1
    grp = (
        fail_df.assign(__one=1)
        .groupby("FPA_RCA1", dropna=False)["__one"]
        .sum()
        .sort_values(ascending=False)
        .rename("Fail_Count")
        .to_frame()
    )
    total_fails = int(grp["Fail_Count"].sum())
    grp["Share_%"] = (grp["Fail_Count"] / total_fails * 100).round(2)
    grp = grp.reset_index()
    grp = grp.rename(columns={"FPA_RCA1": "Driver"})
    grp_top = grp.head(top_n).copy()

    # Plot
    if not grp_top.empty:
        fig = px.bar(
            grp_top.sort_values("Fail_Count"),
            x="Fail_Count",
            y="Driver",
            orientation="h",
            title=f"Top FPA Fail Drivers (Top {len(grp_top)})  —  Total fails: {total_fails}",
            text="Share_%"
        )
        fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside", cliponaxis=False)
        fig.update_layout(yaxis_title="", xaxis_title="Fail count", margin=dict(l=10, r=10, t=60, b=10))
    else:
        fig = None

    # Render to Streamlit (if provided)
    if st is not None:
        st.subheader("FPA Fail Drivers")
        # If month range present, echo it:
        ms, me = _parse_months(query or "")
        if ms or me:
            lbl_start = (ms.strftime("%b %Y") if ms else "…")
            lbl_end   = (me.strftime("%b %Y") if me else "…")
            st.caption(f"Period: {lbl_start} to {lbl_end}")
        st.plotly_chart(fig, use_container_width=True) if fig is not None else st.info("No drivers to plot.")
        st.dataframe(grp_top, use_container_width=True)

    return {"table": grp_top, "fig": fig}
