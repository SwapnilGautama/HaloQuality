# ui/dashboard_standalone.py — Halo HQ (standalone Streamlit, auto data, minimal UI)

# --- ensure repo root is on sys.path for imports when running from ui/ ---
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ---- std libs / deps ----
import io
import glob
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

# -------------------- Helpers --------------------
def _to_month_str(dt):
    try:
        if pd.isna(dt): return None
        d = pd.to_datetime(dt, errors="coerce", dayfirst=True)
        if pd.isna(d): return None
        return d.to_period("M").strftime("%Y-%m")
    except Exception:
        return None

def _std_portfolio(s):
    if pd.isna(s): return s
    t = str(s).strip()
    low = t.lower()
    low = low.replace("leatherhead - baes", "baes leatherhead").replace("baes-leatherhead","baes leatherhead")
    low = low.replace("north west", "northwest")
    return low.title()

def _latest_file(patterns):
    "Return the most recently modified file matching any of the patterns."
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

# -------------------- Loaders (cached) --------------------
@st.cache_data(show_spinner=False)
def load_complaints_auto():
    f = _latest_file(["Complaints*.xlsx", "complaints*.xlsx"])
    if not f:
        return pd.DataFrame(), None
    df = pd.read_excel(f)
    date_col = "Date Complaint Received - DD/MM/YY"
    if date_col not in df.columns:
        # try a common alternative
        alt = [c for c in df.columns if "date" in c.lower() and "complaint" in c.lower()]
        if alt:
            date_col = alt[0]
        else:
            return pd.DataFrame(), str(f)
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
    out = out.drop_duplicates(subset=["month","Case ID"], keep="first")
    return out, files

@st.cache_data(show_spinner=False)
def load_survey_auto():
    f = _latest_file(["Overall raw data*.xlsx", "Survey*.xlsx"])
    if not f:
        return pd.DataFrame(), None
    df = pd.read_excel(f)
    df["Portfolio_std"] = df.get("Portfolio", np.nan).apply(_std_portfolio)
    # try to infer a month column if present
    mcol = None
    for c in df.columns:
        if "month" in c.lower():
            mcol = c; break
    if mcol:
        df["month"] = df[mcol].apply(_to_month_str)
    return df, str(f)

def df_from_table(block: dict) -> pd.DataFrame:
    return pd.DataFrame(block.get("data", {}).get("rows", []))

def download_button(df: pd.DataFrame, label: str, fname: str):
    if df is None or df.empty: return
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    st.download_button(label=label, data=buf.getvalue(),
                       file_name=fname, mime="text/csv")

def find_table(payload: dict, name: str):
    for t in payload.get("tables", []):
        if t.get("name") == name:
            return t
    return None

def find_chart(payload: dict, name: str):
    for c in payload.get("charts", []):
        if c.get("name") == name:
            return c
    return None

def dataref_df(payload: dict, ref: str) -> pd.DataFrame:
    return pd.DataFrame(payload.get("dataRefs", {}).get(ref, []))

def latest_month_from(df: pd.DataFrame) -> str | None:
    if df is None or df.empty or "month" not in df.columns:
        return None
    vals = df["month"].dropna().astype(str).tolist()
    if not vals: return None
    return max(vals)

# -------------------- Page header (Halo-like hero) --------------------
st.markdown(
    """
    <style>
      .halo-hero {padding: 12px 0 8px;}
      .halo-input input {font-size:18px; height:48px;}
      .chip {display:inline-block; padding:8px 12px; margin:6px 8px 0 0; border-radius:9999px;
             border:1px solid #e5e7eb; background:#fff; cursor:pointer; font-size:14px;}
      .chip:hover {background:#f3f4f6;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown("### Conversational Analytics Assistant")
st.markdown(
    "Welcome to **Halo** — analyze complaints and NPS with a simple prompt. "
    "We auto-load your latest files from the **data/** folder."
)

# -------------------- Auto data load (no sidebar paths) --------------------
complaints_df, complaints_path = load_complaints_auto()
cases_df, case_files = load_cases_auto()
survey_df, survey_path = load_survey_auto()

# Health row
c1, c2, c3 = st.columns(3)
c1.metric("Complaints rows", len(complaints_df))
c2.metric("Cases rows (unique Case ID × month)", len(cases_df))
c3.metric("Survey rows", len(survey_df))

# Small caption showing which files were picked
with st.expander("Data sources (auto-detected)", expanded=False):
    st.write(f"**Complaints:** {complaints_path or '— not found —'}")
    if case_files:
        st.write("**Cases (directory):** data/cases/  \n" + "<br/>".join(f"- {Path(f).name}" for f in case_files), unsafe_allow_html=True)
    else:
        st.write("**Cases (directory):** — not found — (expected: data/cases/*.xlsx)")
    st.write(f"**Survey (optional):** {survey_path or '— not found —'}")
    if st.button("↻ Reload data"):
        load_complaints_auto.clear(); load_cases_auto.clear(); load_survey_auto.clear()
        st.experimental_rerun()

# Determine default/latest month from available data
latest_month = latest_month_from(complaints_df) or latest_month_from(cases_df) or "2025-06"

# -------------------- Prompt box (Halo-style) --------------------
st.markdown("##### Start by typing your business question:")
user_q = st.text_input(
    "Ask:",
    placeholder="e.g., Show Complaints Analysis for the latest month",
    label_visibility="collapsed",
    key="halo_prompt"
)

# Suggestion chips
colchips = st.container()
with colchips:
    cc1, cc2, cc3, cc4 = st.columns([1,1,1,1])
    with cc1:
        if st.button("Complaints analysis (latest)", key="chip1", help="Runs complaint_analysis for the latest month", use_container_width=True):
            user_q = "complaints analysis latest"
            st.session_state["halo_prompt"] = user_q
    with cc2:
        if st.button("Top drivers of change", key="chip2", use_container_width=True):
            user_q = "drivers of change"
            st.session_state["halo_prompt"] = user_q
    with cc3:
        if st.button("Reasons heatmap by portfolio", key="chip3", use_container_width=True):
            user_q = "reasons heatmap"
            st.session_state["halo_prompt"] = user_q
    with cc4:
        if st.button("Portfolio comparison", key="chip4", use_container_width=True):
            user_q = "portfolio comparison"
            st.session_state["halo_prompt"] = user_q

# Lightweight intent routing (we currently have one question spec)
def route_intent(q: str):
    q = (q or "").lower()
    qid = "complaint_analysis"  # default
    month = latest_month
    group_by = list(DEFAULT_GROUP_BY)
    # minimal parsing
    if "portfolio" in q and "heatmap" in q:
        group_by = ["Portfolio_std"]
    elif "portfolio" in q:
        group_by = ["Portfolio_std"]
    if "last" in q or "latest" in q or "current" in q:
        month = latest_month
    # allow explicit YYYY-MM mention
    import re
    m = re.search(r"(20\d{2})[-/](0[1-9]|1[0-2])", q)
    if m:
        month = f"{m.group(1)}-{m.group(2)}"
    return qid, month, group_by

qid, month_chosen, gb = route_intent(user_q)

# Optional advanced controls (collapsed)
with st.expander("Advanced (month & group_by)"):
    # Offer month choices from data
    months_available = sorted(set(
        [*complaints_df.get("month", pd.Series(dtype=str)).dropna().astype(str).tolist(),
         *cases_df.get("month", pd.Series(dtype=str)).dropna().astype(str).tolist()]
    ))
    if month_chosen not in months_available and months_available:
        month_chosen = months_available[-1]
    month_chosen = st.selectbox("Month", options=months_available or [month_chosen], index=(months_available.index(month_chosen) if months_available and month_chosen in months_available else 0))
    gb_csv = st.text_input("group_by (CSV)", value=",".join(gb))
    gb = [c.strip() for c in gb_csv.split(",") if c.strip()]
    st.caption("You can leave this collapsed; defaults are fine.")

run_clicked = st.button("Run", type="primary", use_container_width=True)

# -------------------- Execute & Render --------------------
def render_payload(payload: dict):
    # Insights
    st.subheader("Insights")
    st.write(payload.get("insights") or "—")

    # Cards
    cards = payload.get("cards", [])
    if cards:
        data = cards[0].get("data", {})
        a, b, c, d = st.columns(4)
        a.metric("Complaints / 1k", data.get("rate"), delta=data.get("rate_delta"))
        b.metric("Complaints", data.get("complaints"), delta=data.get("complaints_delta"))
        c.metric("Unique Cases", data.get("cases"), delta=data.get("cases_delta"))
        d.metric("NPS", data.get("nps"), delta=data.get("nps_delta"))

    # Drivers bar
    bar_spec = find_chart(payload, "drivers_bar")
    if bar_spec:
        src = bar_spec.get("dataRef")
        xcol, ycol = bar_spec.get("x"), bar_spec.get("y")
        df = dataref_df(payload, src)
        if not df.empty and xcol in df.columns and ycol in df.columns:
            if (bar_spec.get("sort") or "").lower().startswith("desc"):
                df = df.sort_values(ycol, ascending=False)
            fig = px.bar(df, x=xcol, y=ycol, title="Top drivers (Rate Δ)")
            fig.update_layout(height=420, margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig, use_container_width=True)

    # Tables
    st.subheader("Tables")
    for t in payload.get("tables", []):
        title = t.get("title", t.get("name", "Table"))
        st.markdown(f"**{title}**")
        df_t = df_from_table(t)
        if df_t.empty:
            st.caption("No data")
        else:
            st.dataframe(df_t, use_container_width=True, hide_index=True)
            download_button(df_t, f"⬇ Download: {t.get('name','table')}.csv", f"{t.get('name','table')}.csv")

    # Heatmap (pretty) if present
    heat_t = find_table(payload, "reasons_heatmap")
    if heat_t:
        df_heat = df_from_table(heat_t)
        metric_cols = {"Reason","Count","Value","Prev_Count","Prev_Value","Delta","Row_Total","Col_Total","Grand_Total"}
        row_cols = [c for c in df_heat.columns if c not in metric_cols]
        if "Reason" in df_heat.columns and "Value" in df_heat.columns and row_cols:
            idx = row_cols[0]
            pivot = df_heat.pivot_table(index=idx, columns="Reason", values="Value", aggfunc="mean")
            if not pivot.empty:
                st.markdown("**Reasons × Group (Row %)**")
                fig = px.imshow(pivot, aspect="auto", labels=dict(x="Reason", y=idx, color="% within row"))
                fig.update_layout(height=520, margin=dict(l=10, r=10, t=30, b=10))
                st.plotly_chart(fig, use_container_width=True)

if run_clicked:
    try:
        if not list_questions():
            st.error("No questions registered.")
        else:
            spec_path = get_spec_path(qid)
            payload = run_question(
                spec_path=spec_path,
                params={"month": month_chosen, "group_by": gb},
                store_data={
                    "complaints_df": complaints_df,
                    "cases_df": cases_df,
                    "survey_df": survey_df
                }
            )
            render_payload(payload)
    except Exception as e:
        st.error(f"Run failed: {e}")
else:
    st.info("Choose a question by typing (or click a chip) and press **Run**.")
