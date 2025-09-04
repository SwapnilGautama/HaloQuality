# ui/dashboard_standalone.py — Halo HQ (standalone Streamlit, no FastAPI)
import os, io, glob
from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
# --- ensure repo root is on sys.path for imports when running from ui/ ---
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ---- local imports from your repo ----
from question_engine.registry import list_questions, get_spec_path
from question_engine.runner import run_question

# -------------------- Helpers & Data Store --------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
CASES_DIR = DATA_DIR / "cases"

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

@st.cache_data(show_spinner=False)
def load_complaints(path: str):
    df = pd.read_excel(path)
    date_col = "Date Complaint Received - DD/MM/YY"
    if date_col not in df.columns:
        raise ValueError(f"Complaints file missing '{date_col}'")
    df["month"] = df[date_col].apply(_to_month_str)
    df["Portfolio_std"] = df.get("Portfolio", np.nan).apply(_std_portfolio)
    return df

@st.cache_data(show_spinner=False)
def _collect_case_files(dir_or_file: str | None):
    p = Path(dir_or_file) if dir_or_file else CASES_DIR
    if p.is_dir():
        files = sorted(glob.glob(str(p / "*.xlsx")))
    elif p.exists():
        files = [str(p)]
    else:
        files = []
    return files

@st.cache_data(show_spinner=False)
def load_cases(dir_or_file: str | None):
    files = _collect_case_files(dir_or_file)
    if not files:
        return pd.DataFrame(columns=["Report_Date","Case ID","Portfolio","month","Portfolio_std"])
    frames = []
    for f in files:
        df = pd.read_excel(f)
        if "Report_Date" not in df.columns or "Case ID" not in df.columns:
            continue
        df["month"] = df["Report_Date"].apply(_to_month_str)
        df["Portfolio_std"] = df.get("Portfolio", np.nan).apply(_std_portfolio)
        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=["Report_Date","Case ID","Portfolio","month","Portfolio_std"])
    out = pd.concat(frames, ignore_index=True)
    out = out.dropna(subset=["Case ID"])
    out["Case ID"] = out["Case ID"].astype(str)
    out = out.drop_duplicates(subset=["month","Case ID"], keep="first")
    return out

@st.cache_data(show_spinner=False)
def load_survey(path: str | None):
    if not path or not Path(path).exists():
        return pd.DataFrame()
    df = pd.read_excel(path)
    df["Portfolio_std"] = df.get("Portfolio", np.nan).apply(_std_portfolio)
    if "Month_received" in df.columns:
        df["month"] = df["Month_received"].apply(_to_month_str)
    return df

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

# -------------------- UI --------------------
st.set_page_config(page_title="Halo HQ — Streamlit", layout="wide")
st.title("Halo Quality — Streamlit (standalone)")

with st.sidebar:
    st.header("Data")
    default_comp = next((str(p) for p in DATA_DIR.glob("Complaints *.xlsx")), "")
    default_survey = next((str(p) for p in DATA_DIR.glob("Overall raw data*.xlsx")), "")
    complaints_path = st.text_input("Complaints file", value=default_comp)
    survey_path = st.text_input("Survey file (optional)", value=default_survey)
    cases_path = st.text_input("Cases dir or file", value=str(CASES_DIR))
    refresh = st.button("↻ Reload data")

    st.header("Question")
    questions = list_questions()
    q_id = st.selectbox("Question", options=questions, index=questions.index("complaint_analysis") if "complaint_analysis" in questions else 0)
    month = st.text_input("Month (YYYY-MM)", value="2025-06")
    group_by = st.text_input("group_by (CSV)", value="Portfolio_std")
    run_now = st.button("▶ Run")

# load data (cached)
try:
    complaints_df = load_complaints(complaints_path) if complaints_path else pd.DataFrame()
    cases_df = load_cases(cases_path)
    survey_df = load_survey(survey_path)
except Exception as e:
    st.error(f"Failed to load data: {e}")
    st.stop()

if refresh:
    load_complaints.clear()
    load_cases.clear()
    load_survey.clear()
    st.experimental_rerun()

# health chips
c1, c2, c3 = st.columns(3)
c1.metric("Complaints rows", len(complaints_df))
c2.metric("Cases rows", len(cases_df))
c3.metric("Survey rows", len(survey_df))

if run_now:
    try:
        spec_path = get_spec_path(q_id)
        gb = [c.strip() for c in group_by.split(",") if c.strip()]
        payload = run_question(
            spec_path=spec_path,
            params={"month": month, "group_by": gb},
            store_data={
                "complaints_df": complaints_df,
                "cases_df": cases_df,
                "survey_df": survey_df
            }
        )
    except Exception as e:
        st.error(f"Run failed: {e}")
        st.stop()

    # Insights
    st.subheader("Insights")
    st.write(payload.get("insights") or "—")

    # Headline cards
    cards = payload.get("cards", [])
    if cards:
        data = cards[0].get("data", {})
        a, b, c, d = st.columns(4)
        a.metric("Complaints / 1k", data.get("rate"), delta=data.get("rate_delta"))
        b.metric("Complaints", data.get("complaints"), delta=data.get("complaints_delta"))
        c.metric("Unique Cases", data.get("cases"), delta=data.get("cases_delta"))
        d.metric("NPS", data.get("nps"), delta=data.get("nps_delta"))

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

    # Drivers bar chart
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

    # Reasons heatmap (if table present)
    heat_t = find_table(payload, "reasons_heatmap")
    if heat_t:
        df_heat = df_from_table(heat_t)
        metric_cols = {"Reason","Count","Value","Prev_Count","Prev_Value","Delta","Row_Total","Col_Total","Grand_Total"}
        row_cols = [c for c in df_heat.columns if c not in metric_cols]
        if "Reason" in df_heat.columns and "Value" in df_heat.columns and row_cols:
            idx = row_cols[0]
            pivot = df_heat.pivot_table(index=idx, columns="Reason", values="Value", aggfunc="mean")
            if not pivot.empty:
                st.markdown("**Heatmap — Row %**")
                fig = px.imshow(pivot, aspect="auto", labels=dict(x="Reason", y=idx, color="% within row"))
                fig.update_layout(height=520, margin=dict(l=10, r=10, t=30, b=10))
                st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Choose a question and click **Run**.")
