# app.py
from __future__ import annotations
import importlib
from pathlib import Path
import streamlit as st
import pandas as pd

from data_store import load_store
from semantic_router import match_query, IntentMatch

st.set_page_config(page_title="Halo Quality — Chat", layout="wide")

# ---------- Helpers
QUESTIONS_DIR = Path(__file__).parent / "questions"
MODULES = {
    "complaints_per_thousand": "questions.complaints_per_thousand",
    "unique_cases_mom": "questions.unique_cases_mom",
    "rca1_portfolio_process": "questions.rca1_portfolio_process",
}

def _data_status(store: dict):
    st.sidebar.subheader("Data status")
    st.sidebar.write(f"Cases rows: **{store.get('cases_rows', 0)}**")
    st.sidebar.write(f"Complaints rows: **{store.get('complaints_rows', 0)}**")

def _parsed_filters_box(title: str, params: dict | None):
    with st.expander("Parsed filters", expanded=False):
        if not params:
            st.write("—")
        else:
            neat = {k: (v if v not in [None, ""] else "—") for k, v in params.items()}
            st.json(neat)

def _run_question(slug: str, params: dict, store: dict, user_text: str | None = None):
    mod = importlib.import_module(MODULES[slug])
    # All question modules accept (store, params, user_text=None)
    res = mod.run(store, params or {}, user_text=user_text)
    df = res.get("dataframe", pd.DataFrame())
    meta = res.get("meta", {})
    title = meta.get("title", "")
    if title:
        st.subheader(title)
    _parsed_filters_box("Parsed filters", meta.get("filters"))
    if df is None or df.empty:
        st.info("No data for the current filters.")
    else:
        st.dataframe(df, use_container_width=True, hide_index=True)

# ---------- UI
st.title("Halo Quality — Chat")
st.caption("Hi! Ask me about cases, complaints (incl. RCA), or first-pass accuracy.")

store = load_store()
_data_status(store)

# Quick actions
cols = st.columns(3)
with cols[0]:
    if st.button("complaints per 1000 by process for portfolio London Jun 2025 to Aug 2025", use_container_width=True):
        match = IntentMatch(
            slug="complaints_per_thousand",
            params={"portfolio": "London", "start_month": "2025-06", "end_month": "2025-08"},
        )
        _run_question(match.slug, match.params, store, user_text=None)
with cols[1]:
    if st.button("show rca1 by portfolio for process Member Enquiry last 3 months", use_container_width=True):
        match = IntentMatch(slug="rca1_portfolio_process", params={"portfolio": None, "months": 3})
        _run_question(match.slug, match.params, store, user_text=None)
with cols[2]:
    if st.button("unique cases by process and portfolio Apr 2025 to Jun 2025", use_container_width=True):
        match = IntentMatch(slug="unique_cases_mom", params={"start_month": "2025-04", "end_month": "2025-06"})
        _run_question(match.slug, match.params, store, user_text=None)

# Free text
query = st.text_input("Type your question (e.g., 'complaints per 1000 by process last 3 months')", "")
if query:
    match = match_query(query)
    if not match:
        st.error("Sorry—couldn't understand that question.")
    else:
        _run_question(match.slug, match.params, store, user_text=query)
