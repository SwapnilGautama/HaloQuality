# app.py — Halo HQ (Streamlit, minimal UI, auto data + uploader)

# --- ensure repo root is on sys.path ---
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ---- std libs / deps ----
import io
import glob
import re
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px

# ---- local imports (already in your repo) ----
from question_engine.registry import get_spec_path
from question_engine.runner import run_question

# -------------------- Config --------------------
st.set_page_config(page_title="Halo Quality — Streamlit", layout="wide")
DATA_DIR = ROOT / "data"
CASES_DIR = DATA_DIR / "cases"
DEFAULT_GROUP_BY = ["Portfolio_std"]

# --- session state init (separate widget vs. logical keys) ---
st.session_state.setdefault("halo_prompt", "")
st.session_state.setdefault("prompt_input", st.session_state.get("halo_prompt", ""))
st.session_state.setdefault("do_run", False)

def _queue_run_from_input():
    st.session_state["halo_prompt"] = st.session_state.get("prompt_input", "")
    st.session_state["do_run"] = True

def _set_prompt_and_run(v: str):
    st.session_state["prompt_input"] = v
    st.session_state["halo_prompt"]  = v
    st.session_state["do_run"]       = True

# -------------------- Helpers --------------------
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
    candidates = []
    for pat in patterns:
        candidates += list(DATA_DIR.glob(pat))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)

def _collect_case_files():
    return sorted(glob.glob(str(CASES_DIR / "*.xlsx"))) if CASES_DIR.exists() else []

# -------------------- Loaders (cached, silent) --------------------
@st.cache_data(show_spinner=False)
def load_complaints_auto():
    f = _latest_file(["Complaints*.xlsx", "complaints*.xlsx"])
    if not f:
        return pd.DataFrame(), None
    df = pd.read_excel(f)
    # try to find a complaints date column
    candidates = [
        "Date Complaint Received - DD/MM/YY",
        "Date Complaint Received",
        "Complaint Received Date",
        "Received Date",
        "Date",
    ]
    date_col = next((c for c in candidates if c in df.columns), None)
    if not date_col:
        poss = [c for c in df.columns if "date" in c.lower()]
        if poss:
            date_col = poss[0]
        else:
            return df, str(f)
    df["month"] = df[date_col].apply(_to_month_str)
    df["Portfolio_std"] = df.get("Portfolio", np.nan).apply(_std_portfolio)
    return df, str(f)

@st.cache_data(show_spinner=False)
def load_cases_auto():
    files = _collect_case_files()
    if not files:
        return pd.DataFrame(), []
    frames = []
    for f in files:
        try:
            df = pd.read_excel(f)
        except Exception:
            continue
        if "Report_Date" not in df.columns or "Case ID" not in df.columns:
            continue
        df["month"] = df["Report_Date"].apply(_to_month_str)
        df["Portfolio_std"] = df.get("Portfolio", np.nan).apply(_std_portfolio)
        frames.append(df)
    if not frames:
        return pd.DataFrame(), files
    out = pd.concat(frames, ignore_index=True)
    out = out.dropna(subset=["Case ID"])
    out["Case ID"] = out["Case ID"].astype(str)
    out = out.drop_duplicates(subset=["month", "Case ID"], keep="first")
    return out, files

@st.cache_data(show_spinner=False)
def load_survey_auto():
    f = _latest_file(["Overall raw data*.xlsx", "Survey*.xlsx"])
    if not f:
        return pd.DataFrame(), None
    df = pd.read_excel(f)
    df["Portfolio_std"] = df.get("Portfolio", np.nan).apply(_std_portfolio)
    mcol = next((c for c in df.columns if "month" in c.lower()), None)
    if mcol:
        df["month"] = df[mcol].apply(_to_month_str)
    return df, str(f)

def latest_month_from(df: pd.DataFrame) -> str | None:
    if df is None or df.empty or "month" not in df.columns:
        return None
    vals = df["month"].dropna().astype(str).tolist()
    if not vals:
        return None
    return max(vals)

def _has_cols(df: pd.DataFrame, cols: list[str]) -> bool:
    return df is not None and not df.empty and all(c in df.columns for c in cols)

# -------------------- Minimal header --------------------
st.markdown("### Conversational Analytics Assistant")
st.caption("Welcome to **Halo** — we auto-load the latest files from the `data/` folder and answer your question.")

# -------------------- Optional uploader (overrides auto-loaders) --------------------
with st.expander("Upload data (optional)"):
    up_complaints = st.file_uploader("Complaints (.xlsx)", type=["xlsx"], key="up_complaints")
    up_survey     = st.file_uploader("Survey / NPS (.xlsx) — optional", type=["xlsx"], key="up_survey")
    up_cases      = st.file_uploader("Cases (.xlsx) — you can select multiple", type=["xlsx"], accept_multiple_files=True, key="up_cases")

# Auto-load from repo
complaints_df, _ = load_complaints_auto()
cases_df, _      = load_cases_auto()
survey_df, _     = load_survey_auto()

# If uploads provided, override with uploaded data (no caching → immediate)
if up_complaints is not None:
    df = pd.read_excel(up_complaints)
    poss = [c for c in df.columns if "date" in c.lower()]
    if poss:
        df["month"] = df[poss[0]].apply(_to_month_str)
    if "month" not in df.columns:
        df["month"] = np.nan
    df["Portfolio_std"] = df.get("Portfolio", np.nan).apply(_std_portfolio)
    complaints_df = df

if up_cases:
    frames = []
    for f in up_cases:
        try:
            df = pd.read_excel(f)
        except Exception:
            continue
        if "Case ID" in df.columns:
            if "Report_Date" in df.columns:
                df["month"] = df["Report_Date"].apply(_to_month_str)
            elif "month" not in df.columns:
                df["month"] = np.nan
            df["Portfolio_std"] = df.get("Portfolio", np.nan).apply(_std_portfolio)
            frames.append(df)
    if frames:
        out = pd.concat(frames, ignore_index=True)
        out = out.dropna(subset=["Case ID"])
        out["Case ID"] = out["Case ID"].astype(str)
        out = out.drop_duplicates(subset=["month", "Case ID"], keep="first")
        cases_df = out

if up_survey is not None:
    df = pd.read_excel(up_survey)
    df["Portfolio_std"] = df.get("Portfolio", np.nan).apply(_std_portfolio)
    mcol = next((c for c in df.columns if "month" in c.lower()), None)
    if mcol:
        df["month"] = df[mcol].apply(_to_month_str)
    survey_df = df

# Determine default latest month
latest_month = latest_month_from(complaints_df) or latest_month_from(cases_df) or "2025-06"

# -------------------- Prompt & chips --------------------
st.markdown("##### Start by typing your business question:")
st.text_input(
    "Ask:",
    placeholder="e.g., complaints analysis latest",
    label_visibility="collapsed",
    key="prompt_input",
    on_change=_queue_run_from_input,
)

c1, c2
