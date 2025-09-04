# ui/dashboard.py — Halo HQ demo UI (Streamlit)
import os
import io
import json
import requests
import pandas as pd
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="Halo HQ — Questions", layout="wide")

# -------- Sidebar controls --------
st.sidebar.title("Halo HQ")
default_api = os.getenv("HALO_API_BASE", "http://localhost:8000")
api_base = st.sidebar.text_input("API base URL", value=default_api, help="Your FastAPI base, e.g. http://localhost:8000")
refresh = st.sidebar.button("↻ Refresh questions")

def fetch_questions():
    try:
        r = requests.get(f"{api_base}/question/list", timeout=20)
        r.raise_for_status()
        return r.json().get("questions", [])
    except Exception as e:
        st.sidebar.error(f"Failed to load list: {e}")
        return []

questions = fetch_questions() if refresh or True else []
if not questions:
    st.info("No questions found. Make sure `questions/*.yml` exists and API is running.")
    st.stop()

q_id = st.sidebar.selectbox("Question", options=questions, index=questions.index("complaint_analysis") if "complaint_analysis" in questions else 0)
month = st.sidebar.text_input("Month (YYYY-MM)", value="2025-06")
group_by = st.sidebar.text_input("group_by (CSV)", value="Portfolio_std")
show_raw = st.sidebar.checkbox("Show raw payload", value=False)

run_now = st.sidebar.button("Run question")

st.title("Halo Quality — Question Explorer")

# -------- Helpers --------
def run_question(api_base: str, question_id: str, month: str, group_by: str):
    r = requests.get(
        f"{api_base}/question/{question_id}",
        params={"month": month, "group_by": group_by},
        timeout=60
    )
    r.raise_for_status()
    return r.json()

def df_from_table(table_block: dict) -> pd.DataFrame:
    rows = (table_block or {}).get("data", {}).get("rows", [])
    return pd.DataFrame(rows)

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
    data = payload.get("dataRefs", {}).get(ref, [])
    return pd.DataFrame(data)

# -------- Run & Render --------
if run_now:
    try:
        with st.spinner("Running question…"):
            payload = run_question(api_base, q_id, month, group_by)

        # Insights
        st.subheader("Insights")
        st.write(payload.get("insights") or "—")

        # Cards (headline)
        cards = payload.get("cards", [])
        if cards:
            st.subheader("Headline")
            data = cards[0].get("data", {})
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Complaints / 1k", data.get("rate"), delta=data.get("rate_delta"))
            c2.metric("Complaints", data.get("complaints"), delta=data.get("complaints_delta"))
            c3.metric("Unique Cases", data.get("cases"), delta=data.get("cases_delta"))
            c4.metric("NPS", data.get("nps"), delta=data.get("nps_delta"))

        # Tables
        st.subheader("Tables")
        for t in payload.get("tables", []):
            title = t.get("title", t.get("name", "Table"))
            st.markdown(f"**{title}**")
            df = df_from_table(t)
            if df.empty:
                st.caption("No data")
            else:
                st.dataframe(df, use_container_width=True, hide_index=True)
                download_button(df, f"⬇ Download: {t.get('name','table')}.csv", f"{t.get('name','table')}.csv")

        # Charts
        st.subheader("Charts")
        # 1) Drivers bar (if available)
        bar_spec = find_chart(payload, "drivers_bar")
        if bar_spec:
            src = bar_spec.get("dataRef")
            xcol = bar_spec.get("x")
            ycol = bar_spec.get("y")
            df = dataref_df(payload, src)
            # do a light cleanup: drop rows with NaNs in y
            if not df.empty and xcol in df.columns and ycol in df.columns:
                df_plot = df[[xcol, ycol]].dropna()
                if not df_plot.empty:
                    # Sort by y desc if specified
                    if (bar_spec.get("sort") or "").lower().startswith("desc"):
                        df_plot = df_plot.sort_values(ycol, ascending=False)
                    fig = px.bar(df_plot, x=xcol, y=ycol)
                    fig.update_layout(height=420, margin=dict(l=10, r=10, t=30, b=10))
                    st.plotly_chart(fig, use_container_width=True)

        # 2) Reasons heatmap (if table exists)
        heat_t = find_table(payload, "reasons_heatmap")
        if heat_t:
            df_heat = df_from_table(heat_t)
            # Heuristic: group column = first non-metric, non-Reason column
            metric_cols = {"Reason","Count","Value","Prev_Count","Prev_Value","Delta","Row_Total","Col_Total","Grand_Total"}
            row_cols = [c for c in df_heat.columns if c not in metric_cols]
            if "Reason" in df_heat.columns and "Value" in df_heat.columns and row_cols:
                # use first row col as index
                idx_col = row_cols[0]
                pivot = df_heat.pivot_table(index=idx_col, columns="Reason", values="Value", aggfunc="mean")
                if not pivot.empty:
                    st.markdown("**Heatmap — Row %**")
                    fig = px.imshow(pivot, aspect="auto", labels=dict(x="Reason", y=idx_col, color="% within row"))
                    fig.update_layout(height=520, margin=dict(l=10, r=10, t=30, b=10))
                    st.plotly_chart(fig, use_container_width=True)

        if show_raw:
            st.subheader("Raw payload")
            st.code(json.dumps(payload, indent=2)[:200000], language="json")

    except Exception as e:
        st.error(f"Failed to run question: {e}")
        st.stop()
else:
    st.info("Set the API URL, pick a question, and click **Run question**.")
