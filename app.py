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

# Question modules are always looked up in these packages (Q1 and Q2 are isolated here)
QUESTION_MODULE_PREFIXES = ("questions", "core.questions")


def _run_question(store: Dict[str, Any], slug: str, params: Dict[str, Any], user_text: Optional[str] = None):
    """
    Dynamically import a question module and run it.
    Each question module must expose: run(store, params, user_text=None)
    """
    last_exc = None
    for prefix in QUESTION_MODULE_PREFIXES:
        mod_name = f"{prefix}.{slug}"
        try:
            mod = importlib.import_module(mod_name)
            return mod.run(store, params, user_text=user_text)
        except Exception as e:
            last_exc = e
            continue

    err = f"That question module failed to import.\n\nslug={slug}\n\n{traceback.format_exc()}"
    return err, pd.DataFrame()


# -----------------------------
# UI (keep the HALO branding/size from your working version)
# -----------------------------
st.set_page_config(page_title="Halo Quality — Chat", layout="wide")
st.title("Halo Quality — Chat")
st.caption("Hi! Ask me about cases, complaints (incl. RCA), or first-pass accuracy.")

# Load data (compatible with both questions)
with st.spinner("Reading Excel / parquet sources"):
    try:
        # Q1 likes assuming 2025 for string months — the loader supports this kwarg in your repo
        store = load_store(assume_year_for_complaints=2025)  # falls back below if unsupported
    except TypeError:
        store = load_store()

cases: pd.DataFrame = store.get("cases", pd.DataFrame())
complaints: pd.DataFrame = store.get("complaints", pd.DataFrame())

# Small sidebar status (same as your base)
with st.sidebar:
    st.header("Data status")
    st.write(f"Cases rows: **{len(cases):,}**")
    st.write(f"Complaints rows: **{len(complaints):,}**")
    # FPA rows will be counted in Q2 module (it loads its own workbook), so we keep sidebar simple here.

# Two safe chips (do not mutate anything except the text box value)
c1, c2 = st.columns(2)
with c1:
    if st.button("complaint analysis — June 2025 (by portfolio)", use_container_width=True):
        st.session_state["q"] = "complaint analysis — June 2025 by portfolio"
with c2:
    if st.button("first pass accuracy analysis", use_container_width=True):
        st.session_state["q"] = "first pass accuracy analysis"

# Query box
default_q = st.session_state.get("q", "complaint analysis — June 2025 by portfolio")
q = st.text_input(
    "Type your question (e.g., 'complaint analysis — June 2025 by portfolio' or 'first pass accuracy analysis')",
    value=default_q,
)

# Route the query (router never touches app state; it only returns a slug+params)
match = sem_router.match(q) if hasattr(sem_router, "match") else {"slug": "complaints_june_by_portfolio", "params": {}}
slug = match.get("slug", "complaints_june_by_portfolio")
params = match.get("params", {}) or {}

# Run the chosen question
try:
    result, df = _run_question(store, slug, params, user_text=q)
except Exception:
    st.error("Sorry—couldn't run that question.")
    st.code(traceback.format_exc())
else:
    # Render result (question modules render their own charts/tables; returning a df is optional)
    if isinstance(result, tuple) and len(result) in (1, 2):
        title = result[0]
        subtitle = result[1] if len(result) == 2 else None
        if isinstance(title, str) and title.strip():
            st.subheader(title)
        if subtitle:
            st.caption(subtitle)
    elif isinstance(result, str) and result.strip():
        st.info(result)

    if isinstance(df, pd.DataFrame) and not df.empty:
        st.dataframe(df, use_container_width=True)
