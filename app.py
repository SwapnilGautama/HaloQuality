# app.py
from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path
import traceback
import streamlit as st
import pandas as pd


# -----------------------------
# Safe / resilient module import
# -----------------------------
THIS_DIR = Path(__file__).parent

def _import_or_find(module_name: str, prefer_attr: str | None = None):
    """
    Try 'import module_name'. If it fails, search for a file that looks like it
    (e.g., 'data_store.py', 'data_store (1).py') in the current folder and load it
    dynamically. If prefer_attr is provided, return that attribute from the module.
    """
    try:
        mod = importlib.import_module(module_name)
    except ModuleNotFoundError:
        # Try to find a similarly-named file in the same directory
        candidates = list(THIS_DIR.glob(f"{module_name}*.py"))
        if not candidates:
            raise
        path = candidates[0]
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise
        mod = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = mod
        spec.loader.exec_module(mod)
    if prefer_attr:
        return getattr(mod, prefer_attr)
    return mod


# Import data store loader & router — with fallbacks that won’t crash the UI
try:
    load_store = _import_or_find("data_store", "load_store")
except Exception as e:
    load_store = None
    _data_store_import_err = e
else:
    _data_store_import_err = None

try:
    match_query = _import_or_find("semantic_router", "match_query")
    IntentMatch = _import_or_find("semantic_router", "IntentMatch")
except Exception as e:
    match_query = None
    IntentMatch = None
    _router_import_err = e
else:
    _router_import_err = None


# -----------------------------
# Streamlit page config / helpers
# -----------------------------
st.set_page_config(page_title="Halo Quality — Chat", layout="wide")

MODULES = {
    "complaints_per_thousand": "questions.complaints_per_thousand",
    "unique_cases_mom": "questions.unique_cases_mom",
    "rca1_portfolio_process": "questions.rca1_portfolio_process",
}


def _parsed_filters_box(params: dict | None):
    with st.expander("Parsed filters", expanded=False):
        if not params:
            st.write("—")
        else:
            neat = {k: (v if v not in [None, ""] else "—") for k, v in params.items()}
            st.json(neat)


def _run_question(slug: str, params: dict, store: dict, user_text: str | None = None):
    try:
        mod = importlib.import_module(MODULES[slug])
    except Exception:
        st.error(f"Could not import module for '{slug}'.")
        st.code(traceback.format_exc())
        return
    try:
        res = mod.run(store, params or {}, user_text=user_text)
    except Exception:
        st.error("This question failed.")
        st.code(traceback.format_exc())
        return

    df = res.get("dataframe", pd.DataFrame())
    meta = res.get("meta", {})
    title = meta.get("title", "")
    if title:
        st.subheader(title)
    _parsed_filters_box(meta.get("filters"))

    if df is None or df.empty:
        st.info("No data for the current filters.")
    else:
        st.dataframe(df, use_container_width=True, hide_index=True)


def _sidebar_status(store: dict | None):
    st.sidebar.subheader("Data status")
    if not store:
        st.sidebar.write("Cases rows: **0**")
        st.sidebar.write("Complaints rows: **0**")
        return
    st.sidebar.write(f"Cases rows: **{store.get('cases_rows', 0)}**")
    st.sidebar.write(f"Complaints rows: **{store.get('complaints_rows', 0)}**")


# -----------------------------
# UI
# -----------------------------
st.title("Halo Quality — Chat")
st.caption("Hi! Ask me about cases, complaints (incl. RCA), or first-pass accuracy.")

# Warn early if imports failed
if _data_store_import_err:
    st.error("I couldn’t import `data_store`. Make sure **data_store.py** is in the same folder as app.py.")
    st.code("from data_store import load_store", language="python")
    st.code(str(_data_store_import_err))
    st.stop()

if _router_import_err:
    st.error("I couldn’t import `semantic_router`. Make sure **semantic_router.py** is in the same folder as app.py.")
    st.code("from semantic_router import match_query, IntentMatch", language="python")
    st.code(str(_router_import_err))
    st.stop()

# Load data
with st.spinner("Loading data store…"):
    try:
        store = load_store()
    except Exception:
        store = None
        st.error("Failed to load data.")
        st.code(traceback.format_exc())

_sidebar_status(store)

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

# Free-text query
query = st.text_input(
    "Type your question (e.g., 'complaints per 1000 by process last 3 months')",
    "",
)
if query:
    match = match_query(query)
    if not match:
        st.error("Sorry—couldn't understand that question.")
    else:
        _run_question(match.slug, match.params, store, user_text=query)
