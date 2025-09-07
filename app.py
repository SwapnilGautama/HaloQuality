# app.py
from __future__ import annotations
import importlib
import traceback
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd
import streamlit as st

# -----------------------------
# Resilient import helper
# -----------------------------
def _imp(mod: str, attr: str | None = None):
    """
    Import a module (prefers repo root). If that fails, import from core.<mod>.
    Optionally return a named attribute.
    """
    try:
        m = importlib.import_module(mod)
    except ModuleNotFoundError:
        m = importlib.import_module(f"core.{mod}")
    return getattr(m, attr) if attr else m


# These must exist in your repo (root or under core/)
load_store = _imp("data_store", "load_store")
sem_router = _imp("semantic_router")  # must define match(q) -> {"slug": ..., "params": {...}}

# Search paths for question modules
QUESTION_MODULE_PREFIXES = ("questions", "core.questions")


def _run_question(store: Dict[str, Any], slug: str, params: Dict[str, Any], user_text: Optional[str] = None):
    """
    Dynamically import a question module and run it.
    Returns (title/subtitle, dataframe) or (message, empty df) on failure.
    """
    last_exc = None
    for prefix in QUESTION_MODULE_PREFIXES:
        mod_name = f"{prefix}.{slug}"
        try:
            mod = importlib.import_module(mod_name)
            # Each question module must expose: run(store, params, user_text=None)
            return mod.run(store, params, user_text=user_text)
        except Exception as e:
            last_exc = e
            # Try next prefix
            continue

    # If all imports failed, show a helpful message
    err = f"That question module failed to import.\n\nslug={slug}\n\n{traceback.format_exc()}"
    return err, pd.DataFrame()


# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="Halo Quality — Chat", layout="wide")
st.title("Halo Quality — Chat")
st.caption("Hi! Ask me about cases, complaints (incl. RCA), or first-pass accuracy.")

# Load data (defensively accept/ignore kwargs)
with st.spinner("Reading Excel / parquet sources"):
    try:
        store = load_store(assume_year_for_complaints=2025)  # prefer assuming 2025 for complaints 'Report Month'
    except TypeError:
        store = load_store()

cases: pd.DataFrame = store.get("cases", pd.DataFrame())
complaints: pd.DataFrame = store.get("complaints", pd.DataFrame())

# Sidebar status
with st.sidebar:
    st.header("Data status")
    st.write(f"Cases rows: **{len(cases):,}**")
    st.write(f"Complaints rows: **{len(complaints):,}**")

# Quick chips
c1, c2, c3 = st.columns(3)
with c1:
    if st.button("complaint analysis — June 2025 (by portfolio)", use_container_width=True):
        st.session_state["q"] = "complaint analysis — June 2025 (by portfolio)"
with c2:
    st.button("show rca1 by portfolio for process Member Enquiry last 3 months", use_container_width=True)
with c3:
    st.button("unique cases by process and portfolio Apr 2025 to Jun 2025", use_container_width=True)

# Query box (✅ fixed quotes/closing parenthesis)
default_q = st.session_state.get("q", "complaint analysis for June 2025 by portfolio")
q = st.text_input(
    "Type your question (e.g., 'complaint analysis for June 2025 by portfolio')",
    value=default_q,
)

# Route the query
match = sem_router.match(q) if hasattr(sem_router, "match") else {"slug": "complaints_june_by_portfolio", "params": {}}
slug = match.get("slug", "complaints_june_by_portfolio")
params = match.get("params", {}) or {}

# Show parsed filters for debugging
with st.expander("Parsed filters"):
    st.json(params)

# Run the chosen question
try:
    result, df = _run_question(store, slug, params, user_text=q)
except Exception:
    st.error("Sorry—couldn't run that question.")
    st.code(traceback.format_exc())
else:
    # Render result
    if isinstance(result, tuple) and len(result) in (1, 2):
        title = result[0]
        subtitle = result[1] if len(result) == 2 else None
        if isinstance(title, str):
            st.subheader(title)
        elif isinstance(title, (list, tuple)) and title:
            st.subheader(str(title[0]))
            if len(title) > 1:
                st.caption(str(title[1]))
        if subtitle:
            st.caption(subtitle)
    elif isinstance(result, str):
        # A message string came back (e.g., “No data loaded.”)
        st.info(result)

    if isinstance(df, pd.DataFrame) and not df.empty:
        st.dataframe(df, use_container_width=True)
    else:
        st.info("No rows returned for the current filters.")
