# app.py
import os, glob, importlib, inspect
import pandas as pd
import streamlit as st

# If your data store helper exists (as in your project), keep using it:
try:
    from core.data_store import load_store
except Exception:
    load_store = None  # graceful fallback if not present

# --- Page config (collapse & hide sidebar)
st.set_page_config(page_title="Quality Chat", page_icon="💬", layout="wide", initial_sidebar_state="collapsed")
st.markdown(
    """
    <style>
      /* hide the sidebar & its toggle forever */
      [data-testid="stSidebar"], section[data-testid="stSidebar"] {display:none !important;}
      [data-testid="stToolbar"] {right: 0 !important;}
      .block-container {padding-top: 1.2rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Branding (HALO orange pill + Quality dark blue)
BRAND_HTML = """
<h1 style="margin:.2rem 0 1rem 0; font-weight:700; letter-spacing:.2px;">
  <span style="
      background:#ff7a00;
      color:#fff;
      padding:.15rem .5rem .2rem .5rem;
      border-radius:.40rem;
      display:inline-block;">
    HALO
  </span>
  <span style="color:#0b3d91;"> Quality</span>
  <span style="color:#6c757d; font-weight:600;"> — Chat</span>
</h1>
"""
st.markdown(BRAND_HTML, unsafe_allow_html=True)

# --- Quick action chips
st.write("")
c1, c2 = st.columns([1.1, 1.1])
with c1:
    chip_complaints = st.button("complaint analysis — June 2025 (by portfolio)")
with c2:
    chip_fpa = st.button("first pass accuracy analysis")

# --- Query box (autofill from chips)
default_q = ""
if chip_complaints:
    default_q = "complaint analysis — June 2025 (by portfolio)"
elif chip_fpa:
    default_q = "first pass accuracy analysis"

q = st.text_input(
    "Type your question (e.g., 'complaint analysis — June 2025 by portfolio' or 'first pass accuracy analysis')",
    value=default_q,
    placeholder="Ask about complaints or first pass accuracy…",
)

# --- Load the shared store so question modules get data
@st.cache_resource(show_spinner=False)
def get_store():
    if load_store is None:
        return None
    try:
        # match your earlier behaviour (assume 2025 year for complaints month-join)
        return load_store(assume_year_for_complaints=2025)
    except TypeError:
        return load_store()

store = get_store()

# --- Router → question module selection
import semantic_router as _router

def _safe_call_run(mod, store, params, q):
    """
    Call mod.run(...) no matter what signature it uses.
    Supports: run(), run(store), run(store, params), run(store, params, q),
              as well as named kwargs (store/params/q).
    """
    if not hasattr(mod, "run"):
        raise AttributeError("Question module is missing a `run` function.")
    fn = getattr(mod, "run")

    sig = inspect.signature(fn)
    if len(sig.parameters) == 0:
        return fn()

    # Build kwargs for common names
    kwargs = {}
    for name in sig.parameters:
        if name == "store":
            kwargs["store"] = store
        elif name in ("params", "parameters", "args"):
            kwargs["params"] = params
        elif name in ("q", "query", "question"):
            kwargs["q"] = q

    try:
        # First try keyword call (most friendly)
        return fn(**kwargs)
    except TypeError:
        # As a fallback, build a positional arg list in the order declared
        ordered = []
        for name in sig.parameters:
            if name == "store":
                ordered.append(store)
            elif name in ("params", "parameters", "args"):
                ordered.append(params)
            elif name in ("q", "query", "question"):
                ordered.append(q)
            else:
                ordered.append(None)
        return fn(*ordered)

if q.strip():
    try:
        route = _router.match(q)
        slug = route.get("slug")
        params = route.get("params", {}) or {}

        if not slug:
            st.warning("Sorry—couldn't figure out which analysis to run.")
        else:
            mod = importlib.import_module(f"questions.{slug}")
            _safe_call_run(mod, store, params, q)

    except Exception as e:
        st.error("This question failed.")
        st.exception(e)
