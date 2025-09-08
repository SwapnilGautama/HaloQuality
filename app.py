# app.py
from __future__ import annotations
import importlib
import os
import traceback
from typing import Any, Dict, Optional

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
st.set_page_config(page_title="Halo - Quality - AI Assistant", layout="wide")

# kill the sidebar & toolbar permanently (and keep the clean look)
st.markdown(
    """
    <style>
      [data-testid="stSidebar"], [data-testid="stSidebarNav"] { display: none !important; }
      section[data-testid="stSidebar"] { display: none !important; }
      [data-testid="stToolbar"] { display:none !important; }  /* top-right hamburger */

      /* HALO branding */
      .halo-wrap{
        display:flex; align-items:baseline; gap:.75rem; margin:8px 0 22px 0;
      }
      .halo-pill{
        background: linear-gradient(90deg, #FF7A00 0%, #FFD54F 50%, #66BB6A 100%);
        color:white; font-weight:900; letter-spacing:1.2px;
        border-radius:12px; padding:8px 16px; display:inline-block; font-size:28px;
        text-shadow: 0 1px 1px rgba(0,0,0,.18);
      }
      .brand-title{
        color:#0B3B8C; font-weight:800; font-size:36px; line-height:1.1;
      }
      .brand-subtle{
        color:#3c3c3c; font-size:20px; margin-left:.25rem;
      }

      /* tighten input spacing a bit */
      div[data-baseweb="input"] { margin-top: -6px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- Header ----------
st.markdown(
    """
    <div class="halo-wrap">
      <div class="halo-pill">HALO</div>
      <div class="brand-title">- Quality - AI Assistant</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ============================== Load “store” ==============================
with st.spinner("Loading data..."):
    try:
        store = load_store(assume_year_for_complaints=2025)
    except TypeError:
        store = load_store()

# ============================== Router + query box ==============================
# (Removed the 3 chips as requested)
q_default = st.session_state.get("q", "first pass accuracy analysis")
q = st.text_input(
    "Type your question (e.g., 'complaint analysis — June 2025 by portfolio' or 'first pass accuracy analysis' or 'fail reasons analysis')",
    value=q_default,
)
st.session_state["q"] = q

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
