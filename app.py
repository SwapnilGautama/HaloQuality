# app.py
from __future__ import annotations

import importlib
import traceback
from pathlib import Path
from typing import Any, Dict

import streamlit as st

# ---- Safe import for data_store from root or core ----
def _import_load_store():
    try:
        return importlib.import_module("data_store").load_store
    except ModuleNotFoundError:
        return importlib.import_module("core.data_store").load_store


load_store = _import_load_store()

# ---- Router (root semantic_router.py) ----
router = importlib.import_module("semantic_router")


# ---------- helpers ----------
def _import_question_module(slug: str):
    """
    Import question module by slug from 'questions.<slug>'.
    Fallback to 'core.questions.<slug>' if needed.
    """
    module_path = f"questions.{slug}"
    try:
        return importlib.import_module(module_path)
    except ModuleNotFoundError:
        # fallback, in case you keep older structure in core/
        module_path = f"core.questions.{slug}"
        return importlib.import_module(module_path)


def _run_question(mod, store: Dict[str, Any], params: Dict[str, Any], user_text: str):
    """
    Be liberal with signatures: try the common shapes used in older question files.
    """
    try:
        return mod.run(store, params)
    except TypeError:
        try:
            return mod.run(params, store)
        except TypeError:
            try:
                return mod.run(store, params=params, user_text=user_text)
            except TypeError:
                # Last resort: pass store and expand params if run(**kwargs) style.
                return mod.run(store, **(params or {}))


# ---------- UI ----------
st.set_page_config(page_title="Halo Quality — Chat", layout="wide")
st.title("Halo Quality — Chat")
st.caption("Hi! Ask me about cases, complaints (incl. RCA), or first-pass accuracy.")

# Load data (assume 2025 for complaints when only 'Month' is provided)
store = load_store(assume_year_for_complaints=2025)

# Sidebar stats
with st.sidebar:
    st.subheader("Data status")
    st.write(f"Cases rows: **{store.get('cases_rows', 0):,}**")
    st.write(f"Complaints rows: **{store.get('complaints_rows', 0):,}**")
    st.write(f"FPA rows: **{store.get('fpa_rows', 0):,}**")

# Quick chips (examples)
cols = st.columns(3)
with cols[0]:
    if st.button("complaint analysis — June 2025 (by portfolio)"):
        st.session_state["quick_query"] = "complaint analysis for June 2025 by portfolio"
with cols[1]:
    if st.button("show rca1 by portfolio for process Member Enquiry last 3 months"):
        st.session_state["quick_query"] = "show rca1 by portfolio for process Member Enquiry last 3 months"
with cols[2]:
    if st.button("unique cases by process and portfolio Apr 2025 to Jun 2025"):
        st.session_state["quick_query"] = "unique cases by process and portfolio Apr 2025 to Jun 2025"

# Freeform question
default_q = st.session_state.get("quick_query", "")
query = st.text_input(
    "Type your question (e.g., 'complaint analysis for June 2025 by portfolio')",
    value=default_q,
    placeholder="complaint analysis for June 2025 by portfolio",
)

if query:
    try:
        match = router.match(query)
        slug = match.get("slug")
        params = match.get("params", {}) or {}

        # Resolve and run the question module
        mod = _import_question_module(slug)
        _run_question(mod, store, params, user_text=query)

    except ModuleNotFoundError as e:
        st.error(f"That question module failed to import.\n\n**Import error details**\n```\n{e}\n```")
    except Exception as e:
        st.error("This question failed.")
        with st.expander("Traceback", expanded=False):
            st.code("".join(traceback.format_exc()))
