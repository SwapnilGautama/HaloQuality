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
if "halo_prompt" not in st.session_state:
    st.session_state["halo_prompt"] = ""      # logical prompt text used for routing
if "prompt_input" not in st.session_state:
    st.session_state["prompt_input"] = st.session_state["halo_prompt"]
if "do_run" not in st.session_state:
    st.session_state["do_run"] = False

def _queue_run_from_input():
    """Called when user presses Enter in the prompt box."""
    st.session_state["halo_prompt"] = st.session_state.get("prompt_input", "")
    st.session_state["do_run"] = True

def _set_prompt_and_run(v: str):
    """Used by suggestion chips."""
    st.session_state["prompt_input"] = v
    st.session_state["halo_prompt"] = v
    st.session_state["do_run"] = True
    st.rerun()

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
    df.to_csv(buf, index=False)
    st.download_button(label=label, data=buf.getvalue(), file_name=fname, mime="text/csv")

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
    if not vals:
        return None
    return max(vals)

def _has_cols(df: pd.DataFrame, cols: list[str]) -> bool:
    return df is not None and not df.empty and all(c in df.columns for c in cols)

# -------------------- Minimal header (Halo-like) --------------------
st.markdown(
    """
    <style>
      .halo-input input {font-size:18px; height:48px;}
    </style>
    """,
    unsafe_allow_html=True,
)
st.markdown("### Conversational Analytics Assistant")
st.caption("Welcome to **Halo** — we auto-load the latest files from the `data/` folder and answer your question.")

# Load data silently
complaints_df, _ = load_complaints_auto()
cases_df, _ = load_cases_auto()
survey_df, _ = load_survey_auto()

# Determine default month
latest_month = latest_month_from(complaints_df) or latest_month_from(cases_df) or "2025-06"

# -------------------- Prompt box --------------------
st.markdown("##### Start by typing your business question:")
user_q = st.text_input(
    "Ask:",
    value=st.session_state.get("prompt_input", ""),
    placeholder="e.g., complaints analysis latest",
    label_visibility="collapsed",
    key="prompt_input",
    on_change=_queue_run_from_input,  # runs on Enter
)

# Suggestion chips (trigger immediate run)
c1, c2, c3, c4 = st.columns([1, 1, 1, 1])
with c1:
    st.button("Complaints analysis (latest)", use_container_width=True,
              on_click=lambda: _set_prompt_and_run("complaints analysis latest"))
with c2:
    st.button("Top drivers of change", use_container_width=True,
              on_click=lambda: _set_prompt_and_run("drivers of change"))
with c3:
    st.button("Reasons heatmap by portfolio", use_container_width=True,
              on_click=lambda: _set_prompt_and_run("reasons heatmap"))
with c4:
    st.button("Portfolio comparison", use_container_width=True,
              on_click=lambda: _set_prompt_and_run("portfolio comparison"))

# Lightweight intent routing
def route_intent(q: str):
    q = (q or "").lower()
    qid = "complaint_analysis"  # default and only registered for now
    month = latest_month
    group_by = list(DEFAULT_GROUP_BY)

    if "portfolio" in q and "heatmap" in q:
        group_by = ["Portfolio_std"]
    elif "portfolio" in q:
        group_by = ["Portfolio_std"]
    if any(w in q for w in ["last", "latest", "current"]):
        month = latest_month

    m = re.search(r"(20\d{2})[-/](0[1-9]|1[0-2])", q)
    if m:
        month = f"{m.group(1)}-{m.group(2)}"
    return qid, month, group_by

# Tiny advanced override (collapsed)
with st.expander("Advanced (optional: month & group_by)"):
    months_available = sorted(
        set(
            [
                *complaints_df.get("month", pd.Series(dtype=str)).dropna().astype(str).tolist(),
                *cases_df.get("month", pd.Series(dtype=str)).dropna().astype(str).tolist(),
            ]
        )
    )
    adv_month = months_available[-1] if months_available else latest_month
    month_override = st.selectbox(
        "Month",
        options=months_available or [adv_month],
        index=(months_available.index(adv_month) if months_available and adv_month in months_available else 0),
    )
    gb_csv = st.text_input("group_by (CSV)", value=",".join(DEFAULT_GROUP_BY))
    st.session_state["advanced_month"] = month_override
    st.session_state["advanced_gb"] = [c.strip() for c in gb_csv.split(",") if c.strip()]

# -------------------- Execute & Render --------------------
def render_payload(payload: dict):
    st.subheader("Insights")
    st.write(payload.get("insights") or "—")

    cards = payload.get("cards", [])
    if cards:
        data = cards[0].get("data", {})
        a, b, c, d = st.columns(4)
        a.metric("Complaints / 1k", data.get("rate"), delta=data.get("rate_delta"))
        b.metric("Complaints", data.get("complaints"), delta=data.get("complaints_delta"))
        c.metric("Unique Cases", data.get("cases"), delta=data.get("cases_delta"))
        d.metric("NPS", data.get("nps"), delta=data.get("nps_delta"))

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

    heat_t = find_table(payload, "reasons_heatmap")
    if heat_t:
        df_heat = df_from_table(heat_t)
        metric_cols = {"Reason", "Count", "Value", "Prev_Count", "Prev_Value", "Delta", "Row_Total", "Col_Total", "Grand_Total"}
        row_cols = [c for c in df_heat.columns if c not in metric_cols]
        if "Reason" in df_heat.columns and "Value" in df_heat.columns and row_cols:
            idx = row_cols[0]
            pivot = df_heat.pivot_table(index=idx, columns="Reason", values="Value", aggfunc="mean")
            if not pivot.empty:
                st.markdown("**Reasons × Group (Row %)**")
                fig = px.imshow(pivot, aspect="auto", labels=dict(x="Reason", y=idx, color="% within row"))
                fig.update_layout(height=520, margin=dict(l=10, r=10, t=30, b=10))
                st.plotly_chart(fig, use_container_width=True)

def safe_run(prompt_text: str):
    qid, month_chosen, gb = route_intent(prompt_text)
    # If user changed Advanced, prefer those
    month_chosen = st.session_state.get("advanced_month", month_chosen) or month_chosen
    gb = st.session_state.get("advanced_gb", gb) or gb

    # Guard: complaints data must have required cols for this question
    required_cols = ["month", "Portfolio_std"]
    if qid == "complaint_analysis" and not _has_cols(complaints_df, required_cols):
        st.warning(
            "Complaints data not detected with required columns "
            f"{required_cols}. Please add a file like **data/Complaints*.xlsx** "
            "with a complaints date column and a Portfolio column."
        )
        return

    try:
        spec_path = get_spec_path(qid)
        payload = run_question(
            spec_path=spec_path,
            params={"month": month_chosen, "group_by": gb},
            store_data={
                "complaints_df": complaints_df,
                "cases_df": cases_df,
                "survey_df": survey_df,
            },
        )
        render_payload(payload)
    except Exception as e:
        st.error(f"Run failed: {e}")

# Fire when Enter pressed or chip clicked
if st.session_state.get("do_run"):
    st.session_state["do_run"] = False
    safe_run(st.session_state.get("halo_prompt", ""))
