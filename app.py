# -*- coding: utf-8 -*-
import os
import streamlit as st

# App-wide page config
st.set_page_config(page_title="HALO Quality — Chat", page_icon="✅", layout="wide")

# ----------------------
# Brand header (HALO = orange bg / white text, Quality = dark blue)
# ----------------------
BRAND_CSS = """
<style>
/* brand wordmark */
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
# Chips (only two)
# ----------------------
chip_css = """
<style>
.chip {display:inline-block; padding:8px 14px; border:1px solid #ddd; border-radius:999px;
       margin:4px 12px 10px 0; cursor:pointer; background: #fff;}
.chip:hover {border-color:#0a3b8f; color:#0a3b8f;}
</style>
"""
st.markdown(chip_css, unsafe_allow_html=True)

colA, colB = st.columns([1,1])
with colA:
    if st.button("complaint analysis — June 2025 (by portfolio)", key="chip_comp"):
        st.session_state["_router_q"] = "complaint analysis — June 2025 (by portfolio)"
with colB:
    if st.button("first pass accuracy analysis", key="chip_fpa"):
        st.session_state["_router_q"] = "first pass accuracy analysis"

# ----------------------
# Free-text input (kept for parity)
# ----------------------
default_q = st.session_state.get("_router_q", "complaint analysis — June 2025 (by portfolio)")
q = st.text_input(
    "Type your question (e.g., 'complaint analysis — June 2025 by portfolio' or 'first pass accuracy analysis')",
    value=default_q,
)

# ----------------------
# Lightweight semantic router
# ----------------------
def route_query(q: str):
    ql = q.lower()
    if "first pass accuracy" in ql or "fpa" in ql or "first-pass" in ql:
        return "first_pass_accuracy"
    # default to complaints question you already have working
    return "complaints_june_by_portfolio"

slug = route_query(q)

# ----------------------
# Run question module
# ----------------------
# We keep the question runner in this file to avoid sidebars and to keep layout simple.
# Each module returns None and directly renders its content.
if slug == "complaints_june_by_portfolio":
    from questions.complaints_june_by_portfolio import run as run_complaints
    run_complaints()
else:
    from questions.first_pass_accuracy import run as run_fpa
    run_fpa()
