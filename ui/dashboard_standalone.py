# ui/dashboard_standalone.py — Halo HQ (standalone Streamlit, minimal UI, auto data)

# --- ensure repo root is on sys.path for imports when running from ui/ ---
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
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

# ---- local imports ----
from question_engine.registry import list_questions, get_spec_path
from question_engine.runner import run_question

# -------------------- Config --------------------
st.set_page_config(page_title="Halo Quality — Streamlit", layout="wide")
DATA_DIR = ROOT / "data"
CASES_DIR = DATA_DIR / "cases"
DEFAULT_GROUP_BY = ["Portfolio_std"]

# --- session state init (separate widget vs. logical keys) ---
st.session_state.setdefault("halo_prompt", "")     # logical prompt text used for routing
st.session_state.setdefault("prompt_input", "")    # widget-bound text
st.session_state.setdefault("do_run", False)

def _queue_run_from_input():
    """Called when user presses Enter in the prompt box."""
    st.session_state["halo_prompt"] = st.session_state.get("prompt_input", "")
    st.session_state["do_run"] = True  # Streamlit will rerun automatically

def _set_prompt_and_run(v: str):
    """Used by suggestion chips."""
    st.session_state["prompt_input"] = v
    st.session_state["halo_prompt"]  = v
    st.session_state["do_run"]       = True  # no st.rerun() here

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
    """Return the most recently modified file matching any of the patterns."""
    candidates = []
    for pat in patterns:
        candidates += list(DATA_DIR.glob(pat))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)

def _collect_case_files():
    if CASES_DIR.exists():
        files = sorted(glob.glob(str(CASES_DIR / "*.xlsx")))
    else:
        files = []
    return files

# -------------------- Loaders (cached, silent) --------------------
@st.cache_data(show_spinner=False)
def load_complaints_auto():
    f = _latest_file(["Complaints*.xlsx", "complaints*.xlsx"])
    if not f:
        return pd.DataFrame(), None
    df = pd.read_excel(f)
    # expected date column; fallback to heuristic if missing
    date_col = "Date Complaint Received - DD/MM/YY"
    if date_col not in df.columns:
        alt = [c for c in df.columns if "date" in c.lower() and "complaint" in c.lower()]
        if alt:
            date_col = alt[0]
        else:
            # return as-is; runner will check required cols
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
    # infer a month column if present
    mcol = None
    for c in df.columns:
        if "month" in c.lower():
            mcol = c
            break
    if mcol:
        df["month"] = df[mcol].apply(_to_month_str)
    return df, str(f)

def df_from_table(block: dict) -> pd.DataFrame:
    return pd.DataFrame(block.get("data", {}).get("rows", []))

def download_button(df: pd.DataFrame, label: str, fname: str):
    if df is None or df.empty:
        return
    buf = io.StringIO()
    df.to_csv(buf,
