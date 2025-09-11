# app.py
from __future__ import annotations
import importlib
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

# ===================== Page setup ====================
st.set_page_config(page_title="Halo - Quality - AI Assistant", layout="wide")

# ---------- Header styles (no sidebar rule here) ----------
st.markdown(
    """
    <style>
      [data-testid="stToolbar"] { display:none !important; }

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
# FIX: initialize once; let the widget own the value thereafter
if "q" not in st.session_state:
    st.session_state["q"] = "fpa"   # one-time default on first load only

q = st.text_input(
    "Type your question (e.g., 'comp', 'complaint', or 'fpa'; say 'with filters' to open the filter pane)",
    key="q",   # <- binds state; no 'value=' so it won't be reset on reruns
)

# Route
user_query = (q or "").strip()
match = sem_router.match(user_query) if hasattr(sem_router, "match") else {"slug": "complaints_june_by_portfolio", "params": {}}
slug = match.get("slug", "complaints_june_by_portfolio")
params = match.get("params", {}) or {}

# ============================== Conditional sidebar visibility ===============================
def _wants_sidebar(text: str, p: Dict[str, Any]) -> bool:
    if p.get("show_sidebar") is True:
        return True
    t = (text or "").lower()
    keywords = ("filter", "filters", "filter pane", "with filters", "show filters")
    return any(k in t for k in keywords)

SHOW_SIDEBAR = _wants_sidebar(user_query, params)

# Apply CSS to hide sidebar only when NOT requested
if not SHOW_SIDEBAR:
    st.markdown(
        """
        <style>
          [data-testid="stSidebar"], [data-testid="stSidebarNav"] { display: none !important; }
          section[data-testid="stSidebar"] { display: none !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )

# ============================== Run question ===============================
try:
    result, df = _run_question(store, slug, params, user_text=user_query)
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
