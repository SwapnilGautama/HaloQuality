# app.py — Halo Quality (Streamlit) — minimal chat UI, auto-load only
# Implemented question: complaints vs NPS correlation by portfolio (latest or specified month)

import sys, glob, re
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
DEFAULT_GB = ["Portfolio_std"]  # group-by for v1

# -------------------- helpers --------------------
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
    return max(cands, key=lambda p: p.stat().st_mtime) if cands else None

def _collect_case_files():
    return sorted(glob.glob(str(CASES_DIR / "*.xlsx"))) if CASES_DIR.exists() else []

def latest_month_from(df: pd.DataFrame):
    if df is None or df.empty or "month" not in df.columns:
        return None
    vals = df["month"].dropna().astype(str).tolist()
    return max(vals) if vals else None

def _has_cols(df, cols):
    return df is not None and not df.empty and all(c in df.columns for c in cols)

# -------------------- auto-loaders (cached) --------------------
@st.cache_data(show_spinner=False)
def load_complaints_auto():
    f = _latest_file(["Complaints*.xlsx", "complaints*.xlsx"])
    if not f:
        return pd.DataFrame()
    df = pd.read_excel(f)

    # try to find a complaint date column
    candidates = [
        "Date Complaint Received - DD/MM/YY", "Date Complaint Received",
        "Complaint Received Date", "Received Date", "Date"
    ]
    date_col = next((c for c in candidates if c in df.columns), None)
    if not date_col:
        poss = [c for c in df.columns if "date" in c.lower()]
        date_col = poss[0] if poss else None

    if date_col:
        df["month"] = df[date_col].apply(_to_month_str)
    else:
        df["month"] = np.nan

    df["Portfolio_std"] = df.get("Portfolio", np.nan).apply(_std_portfolio)
    return df

@st.cache_data(show_spinner=False)
def load_cases_auto():
    files = _collect_case_files()
    if not files:
        return pd.DataFrame()
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
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out = out.dropna(subset=["Case ID"])
    out["Case ID"] = out["Case ID"].astype(str)
    out = out.drop_duplicates(subset=["month", "Case ID"], keep="first")
    return out

@st.cache_data(show_spinner=False)
def load_survey_auto():
    f = _latest_file(["Overall raw data*.xlsx", "Survey*.xlsx"])
    if not f:
        return pd.DataFrame()
    df = pd.read_excel(f)
    df["Portfolio_std"] = df.get("Portfolio", np.nan).apply(_std_portfolio)
    mcol = next((c for c in df.columns if "month" in c.lower()), None)
    if mcol:
        df["month"] = df[mcol].apply(_to_month_str)
    return df

def compute_nps(df: pd.DataFrame, group_by, month):
    """Return df[group_by + ['NPS']] for the chosen month.
       Works with either an NPS column or Promoters/Passives/Detractors counts."""
    if df is None or df.empty:
        return pd.DataFrame()
    work = df.copy()

    # restrict to month if available
    if "month" in work.columns:
        work = work[work["month"] == month]

    # direct NPS column?
    nps_col = next((c for c in work.columns if c.lower() in {"nps", "net promoter score", "netpromoterscore"}), None)
    if nps_col:
        return (work.groupby(group_by, dropna=False)[nps_col]
                    .mean().reset_index().rename(columns={nps_col: "NPS"}))

    # compute from counts
    cols_l = {c.lower(): c for c in work.columns}
    prom = next((cols_l[c] for c in cols_l if c in {"promoters", "promoters_count"} or ("promoter" in c and "count" in c)), None)
    detr = next((cols_l[c] for c in cols_l if c in {"detractors", "detractors_count"} or ("detractor" in c and "count" in c)), None)
    pasv = next((cols_l[c] for c in cols_l if c in {"passives", "passives_count"} or ("passive" in c and "count" in c)), None)
    if prom and detr and pasv:
        agg = work.groupby(group_by, dropna=False)[[prom, detr, pasv]].sum().reset_index()
        agg["total"] = agg[prom] + agg[detr] + agg[pasv]
        agg = agg[agg["total"] > 0]
        if agg.empty:
            return pd.DataFrame()
        agg["NPS"] = ((agg[prom] - agg[detr]) / agg["total"]) * 100.0
        return agg[group_by + ["NPS"]]

    return pd.DataFrame()

# -------------------- UI header --------------------
st.markdown("### Conversational Analytics Assistant")
st.caption("Welcome to **Halo** — we auto-load the latest files from the `data/` folder. "
           "Try: *“complaints nps correlation”* or *“complaints nps correlation 2025-06”* and press **Enter**.")

# Load data (auto only)
complaints_df = load_complaints_auto()
cases_df      = load_cases_auto()
survey_df     = load_survey_auto()

# Data status
c_rows = len(complaints_df)
ca_rows = len(cases_df)
s_rows = len(survey_df)
st.caption(f"Data status — Complaints: **{c_rows}** rows   •   Cases: **{ca_rows}** rows   •   Survey: **{s_rows}** rows")

latest_month = (latest_month_from(complaints_df)
                or latest_month_from(cases_df)
                or latest_month_from(survey_df)
                or "2025-06")

# -------------------- Prompt box (Enter-to-run) --------------------
st.session_state.setdefault("prompt", "")
st.session_state.setdefault("do_run", False)

def _on_enter():
    st.session_state["do_run"] = True

st.text_input(
    "Ask:",
    key="prompt",
    placeholder="e.g., complaints nps correlation latest month",
    label_visibility="collapsed",
    on_change=_on_enter,
)

# -------------------- Q1: Complaints vs NPS correlation --------------------
def run_corr_complaints_nps(prompt: str):
    # month parsing (YYYY-MM), else default to latest
    m = re.search(r"(20\d{2})[-/](0[1-9]|1[0-2])", (prompt or "").lower())
    month = f"{m.group(1)}-{m.group(2)}" if m else latest_month
    gb = DEFAULT_GB

    # Guards with helpful messages
    if not _has_cols(complaints_df, ["month", "Portfolio_std"]):
        st.warning("Complaints data with columns `month` and `Portfolio` is required. "
                   "Place a `Complaints*.xlsx` under `data/` with a complaints date and a Portfolio.")
        return
    if cases_df.empty or "Case ID" not in cases_df.columns:
        st.warning("Cases data with `Case ID` is required for denominator. "
                   "Place monthly case files under `data/cases/` with `Report_Date`, `Case ID`, and `Portfolio`.")
        return
    if survey_df.empty:
        st.warning("Survey/NPS data is required. Place a file like `Overall raw data*.xlsx` under `data/`.")
        return

    # Complaints numerator (month)
    comp = complaints_df[complaints_df["month"] == month].copy()
    comp_g = comp.groupby(gb, dropna=False).size().reset_index(name="Complaints")

    # Cases denominator (month, unique Case ID)
    cas = cases_df[cases_df["month"] == month].copy()
    cas_g = cas.groupby(gb, dropna=False)["Case ID"].nunique().reset_index(name="Unique_Cases")

    base = comp_g.merge(cas_g, on=gb, how="inner")
    base = base[base["Unique_Cases"] > 0]
    if base.empty:
        st.warning(f"No overlapping data for {month}.")
        return
    base["Complaints_per_1000"] = (base["Complaints"] / base["Unique_Cases"]) * 1000.0

    # NPS (month)
    nps_g = compute_nps(survey_df, gb, month)
    if nps_g.empty or "NPS" not in nps_g.columns:
        st.warning("Could not detect an `NPS` column or derive it from Promoters/Passives/Detractors.")
        return

    df = base.merge(nps_g, on=gb, how="inner").dropna(subset=["NPS"])
    if df.empty:
        st.warning("No groups have both Complaints rate and NPS for the chosen month.")
        return

    # Correlation + trend
    r = np.corrcoef(df["Complaints_per_1000"], df["NPS"])[0, 1]
    slope, intercept = np.polyfit(df["Complaints_per_1000"], df["NPS"], 1)
    xs = np.linspace(df["Complaints_per_1000"].min(), df["Complaints_per_1000"].max(), 50)
    ys = slope * xs + intercept

    direction = "negative" if r < 0 else "positive"
    strength = "strong" if abs(r) >= 0.7 else "moderate" if abs(r) >= 0.4 else "weak"

    st.subheader(f"Complaints vs NPS — {month}")
    st.write(f"**Correlation:** {r:.2f} ({strength}, {direction})  •  Groups: {len(df)}")

    fig = px.scatter(
        df, x="Complaints_per_1000", y="NPS", hover_data=gb,
        labels={"Complaints_per_1000": "Complaints per 1,000 cases"},
    )
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name="trend"))
    fig.update_layout(height=440, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("**By portfolio**")
    view = df.sort_values("Complaints_per_1000", ascending=False)
    st.dataframe(
        view[gb + ["Complaints", "Unique_Cases", "Complaints_per_1000", "NPS"]],
        hide_index=True, use_container_width=True
    )

# -------------------- Router (only this one question for now) --------------------
def handle_prompt(prompt: str):
    q = (prompt or "").lower().strip()
    # intent match for correlation between complaints and NPS/CSAT
    if (("nps" in q or "csat" in q) and
        any(k in q for k in ["correlation", "relationship", "impact", "association"])) \
        or ("complaint" in q and "correlation" in q):
        run_corr_complaints_nps(prompt)
    else:
        st.info("For now I can answer: **complaints vs NPS correlation**. "
                "Try: *“complaints nps correlation”* (optionally add YYYY-MM).")

# -------------------- Fire on Enter --------------------
if st.session_state.get("do_run"):
    st.session_state["do_run"] = False
    handle_prompt(st.session_state.get("prompt", ""))
