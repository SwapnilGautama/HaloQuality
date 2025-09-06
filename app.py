# app.py
from __future__ import annotations

import importlib
import inspect
import traceback

import streamlit as st

from core.data_store import load_store
from semantic_router import match_query, IntentMatch

st.set_page_config(page_title="Halo Quality — Chat", layout="wide")

# --- data store ------------------------------------------------------------
@st.cache_data(show_spinner=True, ttl=60 * 15)
def _load_store_cached():
    return load_store()

try:
    store = _load_store_cached()
    left = st.sidebar
    cases_rows = len(store["cases"]) if "cases" in store else 0
    comp_rows = len(store["complaints"]) if "complaints" in store else 0
    fpa_rows = len(store["fpa"]) if "fpa" in store else 0
    left.markdown("### Data status")
    left.markdown(f"**Cases rows:** {cases_rows:,}")
    left.markdown(f"**Complaints rows:** {comp_rows:,}")
    left.markdown(f"**FPA rows:** {fpa_rows:,}")
except Exception as e:
    st.error("Failed to load data store.")
    st.exception(e)
    st.stop()

st.markdown("## Halo Quality — Chat")
st.caption("Hi! Ask me about cases, complaints (incl. RCA), or first-pass accuracy.")

# quick examples
col = st.container()
col.write(
    st.text_input(
        "Type your question (e.g., 'complaints per 1000 by process last 3 months')",
        key="free_text",
        label_visibility="collapsed",
        placeholder="Type your question…",
    )
)

chip1, chip2, chip3 = st.columns(3)
if chip1.button("complaints per 1000 by process for portfolio London Jun 2025 to Aug 2025"):
    st.session_state.free_text = "complaints per 1000 by process for portfolio London Jun 2025 to Aug 2025"
if chip2.button("show rca1 by portfolio for process Member Enquiry last 3 months"):
    st.session_state.free_text = "show rca1 by portfolio for process Member Enquiry last 3 months"
if chip3.button("unique cases by process and portfolio Apr 2025 to Jun 2025"):
    st.session_state.free_text = "unique cases by process and portfolio Apr 2025 to Jun 2025"

query = st.session_state.get("free_text") or ""

# --- question runner -------------------------------------------------------
def _import_question_module(slug: str):
    """
    Import questions.<slug>. If import fails for canonical slugs,
    try a direct import of the provided slug.
    """
    try:
        return importlib.import_module(f"questions.{slug}")
    except Exception:
        # Surface the inner error in the UI for debugging
        raise

def _run_question(slug: str, args: dict, store, user_text: str):
    mod = _import_question_module(slug)
    if not hasattr(mod, "run"):
        raise RuntimeError(f"Question module questions.{slug} has no run()")

    # filter kwargs to what run() accepts
    spec = inspect.signature(mod.run)
    safe = {k: v for k, v in (args or {}).items() if k in spec.parameters}
    if "store" in spec.parameters:
        safe["store"] = store
    if "user_text" in spec.parameters:
        safe["user_text"] = user_text

    return mod.run(**safe)

# --- handle input ----------------------------------------------------------
if query.strip():
    with st.container():
        try:
            m: IntentMatch | None = match_query(query)
            if not m:
                st.error("Sorry—couldn't understand that question.")
            else:
                st.subheader(m.title)
                _run_question(m.slug, m.args, store, query)
        except Exception:
            st.error("Sorry—this question failed.")
            with st.expander("Traceback"):
                st.code(traceback.format_exc())
