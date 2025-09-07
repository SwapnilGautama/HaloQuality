# app.py
import os, glob, importlib
import pandas as pd
import streamlit as st

from core.data_store import load_store  # keep the shared store used by existing questions

# ---------- Page config & branding ----------
st.set_page_config(page_title="Quality Chat", page_icon="💬", layout="wide")

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

# ---------- Two chips ----------
st.write("")
c1, c2 = st.columns([1.1, 1.1])
with c1:
    chip_complaints = st.button("complaint analysis — June 2025 (by portfolio)")
with c2:
    chip_fpa = st.button("first pass accuracy analysis")

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

# ---------- Shared store for questions (preserves existing behaviour) ----------
@st.cache_resource(show_spinner=False)
def get_store():
    # Same default your complaints question expects
    try:
        return load_store(assume_year_for_complaints=2025)
    except Exception:
        return load_store()

store = get_store()

# ---------- Semantic router & dispatch ----------
import semantic_router as sem_router

if q.strip():
    try:
        match = sem_router.match(q)
        slug = match.get("slug")
        params = match.get("params", {})
        if not slug:
            st.warning("Sorry—couldn't figure out which analysis to run.")
        else:
            mod = importlib.import_module(f"questions.{slug}")
            # Every question's run() draws directly to Streamlit
            try:
                mod.run(store, params, q)
            except Exception as e:
                st.error("This question failed.")
                st.exception(e)
    except Exception as e:
        st.error("Sorry—couldn't run that question.")
        st.exception(e)
