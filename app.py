# app.py
from __future__ import annotations
import importlib
import traceback
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd
import streamlit as st

# =============== Small import helper (keeps Q1 & Q2 isolated) ===============
def _imp(mod: str, attr: str | None = None):
    """Import from repo root; if that fails, from core.<mod>."""
    try:
        m = importlib.import_module(mod)
    except ModuleNotFoundError:
        m = importlib.import_module(f"core.{mod}")
    return getattr(m, attr) if attr else m

# These must exist in your repo (root or core/)
load_store = _imp("data_store", "load_store")
sem_router = _imp("semantic_router")  # must define match(q) -> {"slug": ..., "params": {...}}

# Question modules are always looked up here (keeps them sandboxed from each other)
QUESTION_MODULE_PREFIXES = ("questions", "core.questions")


def _run_question(store: Dict[str, Any], slug: str, params: Dict[str, Any], user_text: Optional[str] = None):
    """
    Dynamically import a question module and run it.
    Every question exposes: run(store, params, user_text=None) -> (message|tuple, optional_df)
    """
    last_exc = None
    for prefix in QUESTION_MODULE_PREFIXES:
        try:
            mod = importlib.import_module(f"{prefix}.{slug}")
            return mod.run(store, params, user_text=user_text)
        except Exception as e:
            last_exc = e
            continue

    err = f"That question module failed to import.\n\nslug={slug}\n\n{traceback.format_exc()}"
    return err, pd.DataFrame()


# ===================== Page setup (branding + no sidebar) ====================
st.set_page_config(page_title="Halo Quality — Chat", layout="wide")

# kill the sidebar & toolbar permanently (and keep the clean look)
st.markdown(
    """
    <style>
      [data-testid="stSidebar"], [data-testid="stSidebarNav"] { display: none !important; }
      section[data-testid="stSidebar"] { display: none !important; }
      [data-testid="stToolbar"] { display:none !important; }  /* top-right hamburger */
      /* HALO pill + brand sizes */
      .halo-wrap{display:flex;align-items:baseline;gap:.5rem;margin:8px 0 22px 0;}
      .halo-pill{
        background:#FF7A00; color:white; font-weight:800; letter-spacing:1px;
        border-radius:10px; padding:6px 12px; display:inline-block; font-size:20px;
      }
      .brand-quality{color:#0B3B8C; font-weight:800; font-size:32px;}
      .brand-chat{color:#3c3c3c; font-size:28px; margin-left:.25rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="halo-wrap">
      <div class="halo-pill">HALO</div>
      <div class="brand-quality">Quality</div>
      <div class="brand-chat">— Chat</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ============================== Load “store” ==============================
# Keep compatibility with Q1 loader (assume 2025 if the function supports it)
with st.spinner("Loading data..."):
    try:
        store = load_store(assume_year_for_complaints=2025)
    except TypeError:
        store = load_store()

# ============================== Chips + router ==============================
c1, c2 = st.columns(2)
with c1:
    if st.button("complaint analysis — June 2025 (by portfolio)", use_container_width=True):
        st.session_state["q"] = "complaint analysis — June 2025 by portfolio"
with c2:
    if st.button("first pass accuracy analysis", use_container_width=True):
        st.session_state["q"] = "first pass accuracy analysis"

q_default = st.session_state.get("q", "complaint analysis — June 2025 by portfolio")
q = st.text_input(
    "Type your question (e.g., 'complaint analysis — June 2025 by portfolio' or 'first pass accuracy analysis')",
    value=q_default,
)

match = sem_router.match(q) if hasattr(sem_router, "match") else {"slug": "complaints_june_by_portfolio", "params": {}}
slug = match.get("slug", "complaints_june_by_portfolio")
params = match.get("params", {}) or {}

# ============================== Run question ===============================
try:
    result, df = _run_question(store, slug, params, user_text=q)
except Exception:
    st.error("Sorry—couldn't run that question.")
    st.code(traceback.format_exc())
else:
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
