# app.py — Halo Quality (Streamlit) — minimal chat UI, auto-load data, Q1 + MoM overview
# - Autoloads latest files from ./data
# - Single prompt box; routes to:
#     1) complaints vs NPS correlation by portfolio (latest or specified month)
#     2) month-on-month overview (complaints, NPS, complaints/1k) + tabs by dimension
# ------------------------------------------------------------------------------

from __future__ import annotations
import re
import sys
from pathlib import Path
from typing import Optional, Tuple, Dict, List

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Optional: keep existing working question modules if present in the repo
try:
    from questions import corr_nps as q_corr_nps  # your working Q1 implementation
except Exception:
    q_corr_nps = None  # we'll fall back to a built-in implementation if missing

try:
    from questions import mom_overview  # new MoM question file you added
except Exception:
    mom_overview = None  # (will warn if route hits this and the file isn't present)


# ------------------------------------------------------------------------------
# Paths / page config
# ------------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
CASES_DIR = DATA_DIR / "cases"
COMPLAINTS_DIR = DATA_DIR / "complaints"
SURVEY_DIR = DATA_DIR / "surveys"

st.set_page_config(page_title="Conversational Analytics Assistant", layout="wide")


# ------------------------------------------------------------------------------
# Helpers — month parsing / normalization
# ------------------------------------------------------------------------------
MONTH_RE = re.compile(r"\b(20\d{2})[-/](0[1-9]|1[0-2])\b")

def _to_month_str(x) -> Optional[str]:
    """Return YYYY-MM or None. Accepts datetime/Period/str."""
    if x is None:
        return None
    # datetime-like
    if hasattr(x, "year") and hasattr(x, "month"):
        try:
            return f"{int(x.year):04d}-{int(x.month):02d}"
        except Exception:
            pass
    # pandas Period
    if hasattr(x, "strftime"):
        try:
            return pd.Period(x, freq="M").strftime("%Y-%m")
        except Exception:
            pass
    s = str(x)
    # try direct regex YYYY-MM
    m = MONTH_RE.search(s)
    if m:
        return f"{m.group(1)}-{m.group(2)}"
    # try common strings like "Jun'25", "June 2025"
    try:
        dt = pd.to_datetime(s, errors="coerce", dayfirst=False)
        if not pd.isna(dt):
            return f"{dt.year:04d}-{dt.month:02d}"
    except Exception:
        pass
    return None


def _months_set(df: pd.DataFrame) -> List[str]:
    if df is None or df.empty or "month" not in df.columns:
        return []
    return sorted({ _to_month_str(x) for x in df["month"].dropna() if _to_month_str(x) })


def _latest_common_month(*dfs: pd.DataFrame) -> Optional[str]:
    sets = [set(_months_set(d)) for d in dfs if d is not None and not d.empty]
    sets = [s for s in sets if s]
    if not sets:
        return None
    common = set.intersection(*sets) if len(sets) > 1 else sets[0]
    return sorted(common)[-1] if common else None


def _find_month_in_prompt(prompt: str) -> Optional[str]:
    if not prompt:
        return None
    m = MONTH_RE.search(prompt)
    if m:
        return f"{m.group(1)}-{m.group(2)}"
    if "last month" in prompt.lower():
        # The caller will replace this with latest common month
        return "__LAST__"
    return None


# ------------------------------------------------------------------------------
# Data loading (cached)
# Expectation:
#   - Cases files under data/cases/** contain columns: "Case ID", "month", and dimensions (e.g., "Portfolio_std")
#   - Complaints under data/complaints/** contain "month" and dimensions (e.g., "Portfolio_std")
#   - Survey under data/surveys/** contain "month", "NPS" (+ optional same dimensions)
# ------------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_cases() -> pd.DataFrame:
    rows = []
    for p in sorted(CASES_DIR.rglob("*.xls*")):
        try:
            df = pd.read_excel(p)
        except Exception:
            continue
        # column normalization
        cols = {c.strip(): c for c in df.columns}
        # month
        if "month" not in df.columns:
            # try to infer from any date-like column
            for cand in ["Month", "Created Month", "Date", "Created", "case_month"]:
                if cand in df.columns:
                    df["month"] = df[cand]
                    break
        df["month"] = df["month"].map(_to_month_str) if "month" in df.columns else _to_month_str(p.stem)
        # Case ID must exist
        if "Case ID" not in df.columns:
            # try common variants
            for cand in ["CaseID", "Case_Id", "CaseId", "Case id", "Unique_Case_ID"]:
                if cand in df.columns:
                    df = df.rename(columns={cand: "Case ID"})
                    break
        # keep minimal columns + all dims
        if "Case ID" in df.columns and "month" in df.columns:
            rows.append(df)
    if not rows:
        return pd.DataFrame()
    cases = pd.concat(rows, ignore_index=True)
    # Coerce month
    cases["month"] = cases["month"].map(_to_month_str)
    return cases.dropna(subset=["month"])


@st.cache_data(show_spinner=False)
def load_complaints() -> pd.DataFrame:
    rows = []
    for p in sorted(COMPLAINTS_DIR.rglob("*.xls*")):
        try:
            df = pd.read_excel(p)
        except Exception:
            continue
        if "month" not in df.columns:
            for cand in ["Month", "Complaints Date", "Date", "Created"]:
                if cand in df.columns:
                    df["month"] = df[cand]
                    break
        df["month"] = df["month"].map(_to_month_str) if "month" in df.columns else _to_month_str(p.stem)
        rows.append(df)
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    out["month"] = out["month"].map(_to_month_str)
    return out.dropna(subset=["month"])


@st.cache_data(show_spinner=False)
def load_survey() -> pd.DataFrame:
    rows = []
    for p in sorted(SURVEY_DIR.rglob("*.xls*")):
        try:
            df = pd.read_excel(p)
        except Exception:
            continue
        # ensure NPS column
        if "NPS" not in df.columns:
            for cand in ["nps", "Nps", "NPS Score", "NPS_score"]:
                if cand in df.columns:
                    df = df.rename(columns={cand: "NPS"})
                    break
        if "month" not in df.columns:
            for cand in ["Month", "Survey Date", "Date", "Response Month"]:
                if cand in df.columns:
                    df["month"] = df[cand]
                    break
        df["month"] = df["month"].map(_to_month_str) if "month" in df.columns else _to_month_str(p.stem)
        rows.append(df)
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    out["month"] = out["month"].map(_to_month_str)
    return out.dropna(subset=["month"])


# ------------------------------------------------------------------------------
# Built-in “corr_nps” fallback (used only if your existing q_corr_nps module is missing)
# ------------------------------------------------------------------------------
def _corr_nps_builtin(store: Dict[str, pd.DataFrame], month: str, group_by: str = "Portfolio_std"):
    complaints = store["complaints"]
    cases = store["cases"]
    survey = store["survey"]

    need_msg = []
    if complaints.empty: need_msg.append("complaints")
    if cases.empty: need_msg.append("cases")
    if survey.empty: need_msg.append("survey")
    if need_msg:
        st.warning(f"Missing data: {', '.join(need_msg)}.")
        return

    # Month filter (use common latest if requested)
    if month == "__AUTO__":
        month = _latest_common_month(complaints, cases, survey)
        if not month:
            st.warning("No overlapping month across datasets.")
            return

    complaints_m = complaints[complaints["month"] == month].copy()
    cases_m = cases[cases["month"] == month].copy()
    survey_m = survey[survey["month"] == month].copy()

    if complaints_m.empty or cases_m.empty or survey_m.empty:
        st.warning(f"No overlapping rows found for {month}.")
        return

    # Per-group
    comp_g = complaints_m.groupby(group_by, dropna=False).size().reset_index(name="Complaints")
    uniq_g = cases_m.groupby(group_by, dropna=False)["Case ID"].nunique().reset_index(name="Unique_Cases")
    nps_g = survey_m.groupby(group_by, dropna=False)["NPS"].mean().reset_index()

    df = comp_g.merge(uniq_g, on=group_by, how="outer").merge(nps_g, on=group_by, how="outer").fillna(0.0)
    df["Complaints_per_1000"] = np.where(df["Unique_Cases"] > 0,
                                         (df["Complaints"] / df["Unique_Cases"]) * 1000.0,
                                         np.nan)

    # Plot
    title = f"NPS vs Complaints per 1,000 — {month}"
    fig = px.scatter(
        df, x="Complaints_per_1000", y="NPS", text=group_by,
        title=title, labels={"Complaints_per_1000": "Complaints per 1,000"},
    )
    fig.update_traces(mode="markers+text", textposition="top center")
    st.plotly_chart(fig, use_container_width=True)

    # Table
    st.subheader("By portfolio")
    show_cols = [group_by, "Complaints", "Unique_Cases", "Complaints_per_1000", "NPS"]
    st.dataframe(df[show_cols].sort_values("Complaints_per_1000", ascending=False), use_container_width=True)


# ------------------------------------------------------------------------------
# Intent Router
# ------------------------------------------------------------------------------
MOM_KEYWORDS = [
    "month on month", "mom", "mo m", "trend", "trends",
    "time series", "over time", "by month", "monthly"
]

def route(prompt: str) -> str | None:
    p = (prompt or "").lower().strip()
    if not p:
        return None
    # Q1 correlation
    if "nps" in p and "complaint" in p and "correl" in p:
        return "corr_nps"
    # MoM overview
    if any(k in p for k in MOM_KEYWORDS) or ("complaint" in p and "month" in p):
        return "mom_overview"
    return None


# ------------------------------------------------------------------------------
# UI: Header + Data status
# ------------------------------------------------------------------------------
def header_and_status(complaints: pd.DataFrame, cases: pd.DataFrame, survey: pd.DataFrame):
    st.markdown("## Conversational Analytics Assistant")
    st.caption(
        "Welcome to **Halo** — we auto-load the latest files from the "
        "`data/` folder. Try: “**complaints nps correlation**” or add a month like "
        "“**complaints nps correlation 2025-06**” and press **Enter**."
    )

    # Status line
    c_rows = 0 if complaints is None else len(complaints)
    k_rows = 0 if cases is None else len(cases)
    s_rows = 0 if survey is None else len(survey)
    months = sorted(set(_months_set(complaints)) | set(_months_set(cases)) | set(_months_set(survey)))
    if months:
        month_span = f"{months[0]} … {months[-1]}" if len(months) > 1 else months[0]
        st.caption(f"Data status — Complaints: **{c_rows}** • Cases: **{k_rows}** • Survey: **{s_rows}** • Months: {', '.join(months)}")
    else:
        st.caption(f"Data status — Complaints: **{c_rows}** • Cases: **{k_rows}** • Survey: **{s_rows}**")


# ------------------------------------------------------------------------------
# Main app
# ------------------------------------------------------------------------------
def main():
    # Load data (cached)
    complaints = load_complaints()
    cases = load_cases()
    survey = load_survey()

    header_and_status(complaints, cases, survey)

    # Prompt box (enter to run)
    prompt = st.text_input(
        "Start by typing your business question:",
        placeholder="e.g., complaints nps correlation",
        key="prompt_input",
    ).strip()

    if not prompt:
        return

    # Route
    qid = route(prompt)
    if not qid:
        st.info("I couldn't match your question. Try “complaints nps correlation” or “complaints month on month”.")
        return

    # Store
    store = {"complaints": complaints, "cases": cases, "survey": survey}

    # Month handling (for correlation route)
    month_in_prompt = _find_month_in_prompt(prompt)
    if month_in_prompt == "__LAST__":
        month_in_prompt = _latest_common_month(complaints, cases, survey)

    # Execute
    if qid == "corr_nps":
        month = month_in_prompt or _latest_common_month(complaints, cases, survey) or "__AUTO__"
        if q_corr_nps is not None:
            # Use your working implementation if present
            try:
                q_corr_nps.run(store, month=month, group_by="Portfolio_std")
            except TypeError:
                # In case your module uses a slightly different signature
                q_corr_nps.run(store, month)
        else:
            # Built-in fallback
            _corr_nps_builtin(store, month=month, group_by="Portfolio_std")
        return

    if qid == "mom_overview":
        if mom_overview is not None:
            mom_overview.run(store)
        else:
            st.warning("MoM module is not available. Please add `questions/mom_overview.py`.")
        return


if __name__ == "__main__":
    main()
