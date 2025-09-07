# app.py
from __future__ import annotations
import importlib
from pathlib import Path
import traceback
import streamlit as st

# ---------- Page & style ----------
st.set_page_config(page_title="HALO Quality — Chat", layout="wide", initial_sidebar_state="collapsed")
HIDE_SIDEBAR = """
<style>
[data-testid="stSidebar"] {display:none !important;}
section.main > div {padding-top: 0rem;}
/* Brand */
.halo-badge{background:#ff7a00;color:#fff;font-weight:800;border-radius:.5rem;padding:.25rem .7rem;display:inline-block;margin-right:.5rem}
.brand-q{color:#0d3b82;font-weight:800}
</style>
"""
st.markdown(HIDE_SIDEBAR, unsafe_allow_html=True)
st.markdown('<div class="halo-badge">HALO</div><span class="brand-q">Quality</span> — Chat', unsafe_allow_html=True)

# ---------- Storage ----------
ROOT = Path(__file__).parent
store = {
    "root": ROOT,
    "data": ROOT / "data",
}

# ---------- Quick actions (chips) ----------
c1, c2 = st.columns([1,1])
with c1:
    q1_btn = st.button("complaint analysis — June 2025 (by portfolio)", use_container_width=True)
with c2:
    q2_btn = st.button("first pass accuracy analysis", use_container_width=True)

q_text = st.text_input(
    "Type your question (e.g., 'complaint analysis — June 2025 by portfolio' or 'first pass accuracy analysis')",
    value=("first pass accuracy analysis" if q2_btn else ("complaint analysis — June 2025 by portfolio" if q1_btn else "")),
)

# ---------- Routing ----------
try:
    from semantic_router import route
except Exception as e:
    st.error(f"Could not import semantic router: {e}")
    st.stop()

slug, params = route(q_text or "")

if not slug:
    st.info("Ask me about complaints or first-pass accuracy. Try a chip above.")
    st.stop()

# ---------- Load, run ----------
try:
    mod = importlib.import_module(f"questions.{slug}")
except Exception:
    st.error(f"Could not load question module: `questions.{slug}`")
    st.stop()

try:
    # Every question gets the same call signature
    _ = mod.run(store, params, q_text)
except Exception as e:
    st.error("This question failed.")
    st.exception(e)
    st.code("".join(traceback.format_exc()))
