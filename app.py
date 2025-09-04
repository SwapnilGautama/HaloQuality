# app.py — Halo Quality (Streamlit) — minimal chat UI
# Q1 implemented: correlation between complaints rate (per 1k cases) and NPS by portfolio (latest or specified month)

import sys, io, glob, re
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

st.set_page_config(page_title="Halo Quality", layout="wide")

DATA_DIR = ROOT / "data"
CASES_DIR = DATA_DIR / "cases"
DEFAULT_GB = ["Portfolio_std"]

# -------------------- Utilities --------------------
def _to_month_str(dt):
    try:
        if pd.isna(dt):
            return None
        d = pd.to_datetime(dt, errors="coerce", dayfirst=True)
        if pd.isna(d):
            return None
        return d.to_period("M").strftime("%Y-%m")
    except Exception:
        return None

def _std_portfolio(s):
    if pd.isna(s):
        return s
    t = str(s).strip()
    low = t.lower()
    low = low.replace("leatherhead - baes", "baes leatherhead").replace("baes-leatherhead", "baes leatherhead")
    low = low.replace("north west", "northwest")
    return low.title()

def _latest_file(patterns):
    cands = []
    for pat in patterns:
        cands += list(DATA_DIR.glob(pat))
    if not cands:
        return None
    return max(cands, key=lambda p: p.stat().st_mtime)

def _collect_case_files():
    return sorted(glob.glob(str(CASES_DIR / "*.xlsx"))) if CASES_DIR.exists() else []

def latest_month_from(df: pd.DataFrame):
    if df is None or df.empty or "month" not in df.columns:
        return None
    vals = df["month"].dropna().astype(str).tolist()
    return max(vals) if vals else None

def _has_cols(df, cols):
    return df is not None and not df.empty and all(c in df.columns for c in cols)

# -------------------- Loaders (cached) --------------------
@st.cache_data(show_spinner=False)
def load_complaints_auto():
    f = _latest_file(["Complaints*.xlsx", "complaints*.xlsx"])
    if not f:
        return pd.DataFrame()
    df = pd.read_excel(f)
    # guess date col
    candidates = [
        "Date Complaint Received - DD/MM/YY", "Date Complaint Received",
        "Complaint Received Date", "Received Date", "Date"
    ]
    date_col = next((c for c in candidates if c in df.columns), None)
    if not date_col:
        poss = [c for c in df.columns if "date" in c.lower()]
        if poss:
            date_col = poss[0]
    if date_col:
        df["month"] = df[date_col].apply(_to_month_str)
    else:
        df["month"] = np.nan
    df["Portfolio_std"] = df.get("Portfolio", np.nan).apply(_std_portfolio)
    return df

@st.cache_data(show_spinner=False)
def load_cases_auto():
    files = _collect_case_files()
    if not files: return pd.DataFrame()
    frames = []
    for f in files:
        try:
            d = pd.read_excel(f)
        except Exception:
            continue
        if "Case ID" not in d.columns: 
            continue
        if "Report_Date" in d.columns:
            d["month"] = d["Report_Date"].apply(_to_month_str)
        elif "month" not in d.columns:
            d["month"] = np.nan
        d["Portfolio_std"] = d.get("Portfolio", np.nan).apply(_std_portfolio)
        frames.append(d)
    if not frames: return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out = out.dropna(subset=["Case ID"])
    out["Case ID"] = out["Case ID"].astype(str)
    out = out.drop_duplicates(subset=["month", "Case ID"], keep="first")
    return out

@st.cache_data(show_spinner=False)
def load_survey_auto():
    f = _latest_file(["Overall raw data*.xlsx", "Survey*.xlsx"])
    if not f: return pd.DataFrame()
    df = pd.read_excel(f)
    df["Portfolio_std"] = df.get("Portfolio", np.nan).apply(_std_portfolio)
    mcol = next((c for c in df.columns if "month" in c.lower()), None)
    if mcol:
        df["month"] = df[mcol].apply(_to_month_str)
    return df

# Detect or compute NPS from a survey dataframe grouped by group_by + month
def compute_nps(df: pd.DataFrame, group_by, month):
    if df is None or df.empty: 
        return pd.DataFrame()

    gcols = list(group_by) + (["month"] if "month" in df.columns else [])
    work = df.copy()

    # 1) direct NPS column?
    nps_col = next((c for c in work.columns if c.lower() in {"nps", "net promoter score", "netpromoterscore"}), None)
    if nps_col:
        if "month" in work.columns:
            work = work[work["month"] == month]
        out = work.groupby(group_by, dropna=False)[nps_col].mean().reset_index()
        out = out.rename(columns={nps_col: "NPS"})
        return out

    # 2) compute from counts if available
    cols_l = {c.lower(): c for c in work.columns}
    prom = next((cols_l[c] for c in cols_l if "promoter" in c and "count" in c or c=="promoters"), None)
    detr = next((cols_l[c] for c in cols_l if "detractor" in c and "count" in c or c=="detractors"), None)
    passv = next((cols_l[c] for c in cols_l if "passive" in c and "count" in c or c=="passives"), None)
    if prom and detr and passv:
        if "month" in work.columns:
            work = work[work["month"] == month]
        agg = work.groupby(group_by, dropna=False)[[prom, detr, passv]].sum().reset_index()
        agg["total"] = agg[prom] + agg[detr] + agg[passv]
        agg = agg[agg["total"] > 0]
        agg["NPS"] = ((agg[prom] - agg[detr]) / agg["total"]) * 100.0
        return agg[group_by + ["NPS"]]

    # Nothing matched
    return pd.DataFrame()

# -------------------- UI: minimal chat --------------------
st.markdown("### Conversational Analytics Assistant")
st.caption("Welcome to **Halo** — we auto-load the latest files from the `data/` folder and answer your question. For now, try: *“complaints nps correlation”*.")

with st.expander("Upload data (optional)"):
    up_complaints = st.file_uploader("Complaints (.xlsx)", type=["xlsx"])
    up_cases      = st.file_uploader("Cases (.xlsx) — multiple allowed", type=["xlsx"], accept_multiple_files=True)
    up_survey     = st.file_uploader("Survey / NPS (.xlsx)", type=["xlsx"])

# Load repo data
complaints_df = load_complaints_auto()
cases_df      = load_cases_auto()
survey_df     = load_survey_auto()

# Override with uploads (no caching)
if up_complaints is not None:
    d = pd.read_excel(up_complaints)
    poss = [c for c in d.columns if "date" in c.lower()]
    d["month"] = d[poss[0]].apply(_to_month_str) if poss else np.nan
    d["Portfolio_std"] = d.get("Portfolio", np.nan).apply(_std_portfolio)
    complaints_df = d

if up_cases:
    frames = []
    for f in up_cases:
        d = pd.read_excel(f)
        if "Case ID" not in d.columns: 
            continue
        if "Report_Date" in d.columns:
            d["month"] = d["Report_Date"].apply(_to_month_str)
        elif "month" not in d.columns:
            d["month"] = np.nan
        d["Portfolio_std"] = d.get("Portfolio", np.nan).apply(_std_portfolio)
        frames.append(d)
    if frames:
        out = pd.concat(frames, ignore_index=True)
        out = out.dropna(subset=["Case ID"])
        out["Case ID"] = out["Case ID"].astype(str)
        out = out.drop_duplicates(subset=["month", "Case ID"], keep="first")
        cases_df = out

if up_survey is not None:
    d = pd.read_excel(up_survey)
    d["Portfolio_std"] = d.get("Portfolio", np.nan).apply(_std_portfolio)
    mcol = next((c for c in d.columns if "month" in c.lower()), None)
    if mcol:
        d["month"] = d[mcol].apply(_to_month_str)
    survey_df = d

latest_month = latest_month_from(complaints_df) or latest_month_from(cases_df) or latest_month_from(survey_df) or "2025-06"

# Session state for prompt + run
st.session_state.setdefault("prompt", "")
st.session_state.setdefault("do_run", False)

def _on_enter():
    st.session_state["do_run"] = True

st.text_input(
    "Ask:",
    placeholder="e.g., complaints nps correlation for latest month",
    label_visibility="collapsed",
    key="prompt",
    on_change=_on_enter,
)

# -------------------- Q1: Complaints vs NPS correlation --------------------
def run_corr_complaints_nps(prompt: str):
    # month inference from prompt like 2025-06
    m = re.search(r"(20\d{2})[-/](0[1-9]|1[0-2])", (prompt or "").lower())
    month = f"{m.group(1)}-{m.group(2)}" if m else latest_month
    gb = DEFAULT_GB  # keep it simple for now

    # Guards
    if not _has_cols(complaints_df, ["month", "Portfolio_std"]):
        st.warning("Complaints data with `month` and `Portfolio` is required. Upload a complaints file or add one under `data/`.")
        return
    if cases_df.empty or "Case ID" not in cases_df.columns:
        st
