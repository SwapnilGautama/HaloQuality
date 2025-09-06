# app.py
from __future__ import annotations

import importlib
import traceback
from pathlib import Path

import pandas as pd
import streamlit as st


# -----------------------------
# Tolerant dynamic import helper
# -----------------------------
def _try_import(module_name: str, attr: str | None = None):
    try:
        mod = importlib.import_module(module_name)
        return getattr(mod, attr) if attr else mod
    except Exception:
        return None


# -----------------------------
# Resolve data_store.load_store
# (works whether file is in root or core/)
# -----------------------------
load_store = None
for cand in ("data_store", "core.data_store"):
    load_store = _try_import(cand, "load_store")
    if load_store:
        break

if load_store is None:
    st.error("Could not import load_store from data_store or core.data_store.")
    st.stop()


# -----------------------------
# Resolve semantic_router.match_query
# -----------------------------
match_query = None
for cand in ("semantic_router", "core.semantic_router"):
    match_query = _try_import(cand, "match_query")
    if match_query:
        break

if match_query is None:
    st.error("Could not import match_query from semantic_router or core.semantic_router.")
    st.stop()


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Halo Quality — Chat", layout="wide")
st.title("Halo Quality — Chat")
st.caption("Hi! Ask me about cases, complaints (incl. RCA), or first-pass accuracy.")


# -----------------------------
# Load the data store (SAFE)
# -----------------------------
with st.spinner("Loading data store…"):
    try:
        # Preferred: pass the assumption for complaints month-only files
        store = load_store(assume_year_for_complaints=2025)
    except TypeError:
        # Fallback: your current load_store doesn't accept the kwarg
        store = load_store()

# Sidebar data status
with st.sidebar:
    st.subheader("Data status")
    cases_rows = len(store.get("cases", pd.DataFrame()))
    compl_rows = len(store.get("complaints", pd.DataFrame()))
    fpa_rows = len(store.get("fpa", pd.DataFrame()))
    st.write(f"Cases rows: **{cases_rows:,}**")
    st.write(f"Complaints rows: **{compl_rows:,}**")
    st.write(f"FPA rows: **{fpa_rows:,}**")


# -----------------------------
# Quick action chips
# -----------------------------
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


# -----------------------------
# Query input
# -----------------------------
query = st.text_input(
    "Type your question (e.g., 'complaints per 1000 by process last 3 months')",
    value=st.session_state.get("q", ""),
    key="q_input",
)


# -----------------------------
# Runner for questions/<slug>.py
# -----------------------------
def _run_question(slug: str, params: dict, user_text: str):
    """
    Imports questions/<slug>.py and calls its run(store, params, user_text)
    Returns: (title, dataframe, notes[list[str]])
    """
    try:
        mod = importlib.import_module(f"questions.{slug}")
    except Exception as e:
        return None, None, [f"Could not import questions.{slug}: {e}"]

    try:
        return mod.run(store, params=params or {}, user_text=user_text)
    except Exception as e:
        tb = traceback.format_exc()
        return None, None, [f"{e}", tb]


# -----------------------------
# Route & execute
# -----------------------------
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
