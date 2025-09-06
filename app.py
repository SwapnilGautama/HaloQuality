# app.py
from __future__ import annotations

import importlib
import sys
from pathlib import Path
import traceback
import streamlit as st
import pandas as pd

# ---------- tolerant imports (works whether files live in root or core/) ----------
def _try_import(module_name: str, attr: str | None = None):
    try:
        mod = importlib.import_module(module_name)
        return getattr(mod, attr) if attr else mod
    except Exception:
        return None

load_store = None
for cand in ("data_store", "core.data_store"):
    load_store = _try_import(cand, "load_store")
    if load_store:
        break
if load_store is None:
    st.stop()  # Will show the red box from Streamlit with ModuleNotFound

match_query = None
for cand in ("semantic_router", "core.semantic_router"):
    match_query = _try_import(cand, "match_query")
    if match_query:
        break
if match_query is None:
    st.stop()

# ---------- UI ----------
st.set_page_config(page_title="Halo Quality — Chat", layout="wide")
st.title("Halo Quality — Chat")
st.caption("Hi! Ask me about cases, complaints (incl. RCA), or first-pass accuracy.")

with st.sidebar:
    st.subheader("Data status")

# Load data (cache handled inside data_store)
with st.spinner("Loading data store…"):
    store = load_store(assume_year_for_complaints=2025)  # <— assume 2025 if only ‘Month’

# Sidebar counts
with st.sidebar:
    cases_rows = len(store.get("cases", pd.DataFrame()))
    cmpl_rows = len(store.get("complaints", pd.DataFrame()))
    fpa_rows = len(store.get("fpa", pd.DataFrame()))
    st.write(f"Cases rows: **{cases_rows:,}**")
    st.write(f"Complaints rows: **{cmpl_rows:,}**")
    st.write(f"FPA rows: **{fpa_rows:,}**")

# Quick chips
st.write("")
cols = st.columns(3)
with cols[0]:
    st.button(
        "complaints per 1000 by process for portfolio London Jun 2025 to Aug 2025",
        key="chip_cpt",
        on_click=lambda: st.session_state.update(
            q="complaints per 1000 by process for portfolio London Jun 2025 to Aug 2025"
        ),
    )
with cols[1]:
    st.button(
        "show rca1 by portfolio for process Member Enquiry last 3 months",
        key="chip_rca",
        on_click=lambda: st.session_state.update(
            q="show rca1 by portfolio for process Member Enquiry last 3 months"
        ),
    )
with cols[2]:
    st.button(
        "unique cases by process and portfolio Apr 2025 to Jun 2025",
        key="chip_uc",
        on_click=lambda: st.session_state.update(
            q="unique cases by process and portfolio Apr 2025 to Jun 2025"
        ),
    )

query = st.text_input(
    "Type your question (e.g., 'complaints per 1000 by process last 3 months')",
    value=st.session_state.get("q", ""),
    key="q_input",
)

def _run_question(slug: str, params: dict, user_text: str):
    """
    Dynamically import and execute a question in questions/<slug>.py
    Returns (title, df, notes)
    """
    mod = None
    err = None
    try:
        # NOTE: questions live in the root 'questions' directory
        mod = importlib.import_module(f"questions.{slug}")
    except Exception as e:
        err = f"Could not import questions.{slug}: {e}"

    if err:
        return None, None, [err]

    try:
        return mod.run(store, params=params or {}, user_text=user_text)
    except Exception as e:
        tb = traceback.format_exc()
        return None, None, [f"{e}", tb]

# Route + run
if query:
    match = match_query(query)
    with st.expander("Parsed filters", expanded=False):
        st.json(match.params or {})

    if match.slug == "complaints_per_thousand":
        title, df, notes = _run_question("complaints_per_thousand", match.params, query)
    elif match.slug == "rca1_portfolio_process":
        title, df, notes = _run_question("rca1_portfolio_process", match.params, query)
    elif match.slug == "unique_cases_mom":
        title, df, notes = _run_question("unique_cases_mom", match.params, query)
    else:
        title, df, notes = None, None, ["Sorry—couldn't understand that question."]

    st.subheader(title or "Result")
    if notes:
        for n in notes:
            st.info(n)

    if df is not None and len(df):
        st.dataframe(df, use_container_width=True)
    elif df is not None and len(df) == 0:
        st.info("No rows returned for the current filters.")
