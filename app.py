# app.py
from __future__ import annotations
import importlib
import sys
from pathlib import Path
import traceback
import streamlit as st

# ---------- Page ----------
st.set_page_config(page_title="HALO Quality — Chat", layout="wide", initial_sidebar_state="collapsed")
st.markdown("""
<style>
[data-testid="stSidebar"]{display:none!important;}
.halo{background:#ff7a00;color:#fff;font-weight:800;border-radius:.5rem;padding:.25rem .7rem;margin-right:.5rem}
.brand{color:#0d3b82;font-weight:800}
</style>
""", unsafe_allow_html=True)
st.markdown('<span class="halo">HALO</span><span class="brand">Quality</span> — Chat', unsafe_allow_html=True)

# ---------- Shared store ----------
ROOT = Path(__file__).parent
store = {"root": ROOT, "data": ROOT / "data"}

# ---------- Chips / inputs ----------
c1, c2 = st.columns([1,1])
with c1:
    chip_q1 = st.button("complaint analysis — June 2025 (by portfolio)", use_container_width=True)
with c2:
    chip_q2 = st.button("first pass accuracy analysis", use_container_width=True)

default_q = "complaint analysis — June 2025 by portfolio"
if chip_q2:
    default_q = "first pass accuracy analysis"
elif chip_q1:
    default_q = "complaint analysis — June 2025 by portfolio"

q_text = st.text_input(
    "Type your question (e.g., 'complaint analysis — June 2025 by portfolio' or 'first pass accuracy analysis')",
    value=default_q,
)

# ---------- Router ----------
try:
    from semantic_router import route
except Exception as e:
    st.error(f"Could not import semantic router: {e}")
    st.stop()

slug, params = route(q_text or "")
if not slug:
    st.info("Ask me about complaints or first-pass accuracy using the chips above.")
    st.stop()

# ---------- HARD ISOLATION: only import the chosen question ----------
Q1 = "questions.complaints_june_by_portfolio"
Q2 = "questions.first_pass_accuracy"
for m in [Q1, Q2]:
    if m in sys.modules:
        del sys.modules[m]
target = Q1 if slug == "complaints_june_by_portfolio" else Q2

try:
    mod = importlib.import_module(target)
except Exception:
    st.error(f"Could not load `{target}`.")
    st.code("".join(traceback.format_exc()))
    st.stop()

# ---------- Run the question ----------
try:
    _ = mod.run(store, params, q_text)
except Exception as e:
    st.error("This question failed.")
    st.exception(e)
    st.code("".join(traceback.format_exc()))
