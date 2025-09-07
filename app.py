# -*- coding: utf-8 -*-
import os
import inspect
from importlib import import_module

import streamlit as st

st.set_page_config(page_title="HALO Quality — Chat", page_icon="✅", layout="wide")

# ----------------------
# Brand header
# ----------------------
BRAND_CSS = """
<style>
.halo-badge{
  display:inline-block;
  background:#ff7a00; /* orange */
  color:#ffffff;
  padding:8px 14px;
  margin-right:10px;
  font-weight:800;
  border-radius:6px;
  letter-spacing:1px;
}
.quality-word{
  color:#0a3b8f;      /* dark blue */
  font-weight:800;
}
h1.app-title{
  margin:6px 0 22px 0 !important;
}
</style>
"""
st.markdown(BRAND_CSS, unsafe_allow_html=True)
st.markdown('<h1 class="app-title"><span class="halo-badge">HALO</span><span class="quality-word">Quality</span> — Chat</h1>', unsafe_allow_html=True)

# ----------------------
# Chips (exactly two)
# ----------------------
chip_css = """
<style>
.chip {display:inline-block; padding:8px 14px; border:1px solid #ddd; border-radius:999px;
       margin:4px 12px 10px 0; cursor:pointer; background:#fff;}
.chip:hover {border-color:#0a3b8f; color:#0a3b8f;}
</style>
"""
st.markdown(chip_css, unsafe_allow_html=True)

c1, c2 = st.columns([1, 1])
with c1:
    if st.button("complaint analysis — June 2025 (by portfolio)", key="chip_comp"):
        st.session_state["_router_q"] = "complaint analysis — June 2025 (by portfolio)"
with c2:
    if st.button("first pass accuracy analysis", key="chip_fpa"):
        st.session_state["_router_q"] = "first pass accuracy analysis"

# ----------------------
# Input (kept for parity)
# ----------------------
default_q = st.session_state.get("_router_q", "complaint analysis — June 2025 (by portfolio)")
q = st.text_input(
    "Type your question (e.g., 'complaint analysis — June 2025 by portfolio' or 'first pass accuracy analysis')",
    value=default_q,
)

# ----------------------
# Semantic routing
# ----------------------
try:
    from question_engine.semantic_router import match as router_match
except Exception:
    router_match = None

def route_query(text: str):
    if router_match is not None:
        m = router_match(text)
        if isinstance(m, dict) and "slug" in m:
            return m["slug"], m.get("params", {})
    # fallback: simple heuristics
    tl = (text or "").lower()
    if "first pass accuracy" in tl or "first-pass" in tl or "fpa" in tl:
        return "first_pass_accuracy", {}
    return "complaints_june_by_portfolio", {}

slug, params = route_query(q)

# ----------------------
# Safe dynamic import & invocation
# ----------------------
def _import_run(slug_name: str):
    """
    Import questions.<slug_name> and fetch its `run` attribute.
    """
    mod = import_module(f"questions.{slug_name}")
    run_fn = getattr(mod, "run", None)
    return run_fn

def _safe_invoke(run_fn, q_text: str, params_dict: dict):
    """
    Some question modules take different run() signatures across versions:
      - run()
      - run(params)
      - run(store, params)
      - run(store, slug, params, user_text=...)
    We introspect and try sensible permutations so we don't crash with TypeError.
    """
    if run_fn is None:
        st.error("Selected question has no `run()` entrypoint.")
        return

    sig = inspect.signature(run_fn)
    required_positional = [
        p for p in sig.parameters.values()
        if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        and p.default is inspect._empty
    ]
    # Try common call-shapes in order of least assumptions → most assumptions
    attempts = [
        (),                         # run()
        (params_dict,),             # run(params)
        ({}, params_dict),          # run(store={}, params)
        ({}, slug, params_dict),    # run(store={}, slug, params)
        ({}, slug, params_dict, q_text),       # run(store, slug, params, user_text)
    ]
    for args in attempts:
        try:
            return run_fn(*args)
        except TypeError:
            continue
    # Final fallback: just try no-args
    try:
        return run_fn()
    except Exception as e:
        st.error(f"Failed to execute question module: {e}")

# ----------------------
# Execute module
# ----------------------
try:
    run_fn = _import_run(slug)
except Exception as e:
    st.error(f"Could not import question module for slug '{slug}': {e}")
else:
    _safe_invoke(run_fn, q, params)
