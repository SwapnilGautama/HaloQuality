# -*- coding: utf-8 -*-
# app.py
from __future__ import annotations

import importlib
from pathlib import Path

import streamlit as st

# --------------------- basic page setup ---------------------
st.set_page_config(page_title="HALO Quality — Chat", layout="wide")

# brand header
st.markdown(
    """
    <style>
      .halo-badge {
        display:inline-block;padding:6px 12px;border-radius:6px;
        background:#ff7a00;color:#ffffff;font-weight:800;letter-spacing:1px;
        font-family: ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,Ubuntu,Cantarell,"Helvetica Neue",Arial;
      }
      .brand-title {
        font-size:34px;font-weight:800;color:#0b3d91;vertical-align:middle;margin-left:10px
      }
      .brand-sub { font-size:34px;font-weight:600;color:#444;vertical-align:middle;margin-left:6px }
      .stTabs [data-baseweb="tab-list"] { gap: 18px; }
    </style>
    <div style="margin:6px 0 14px 0;">
      <span class="halo-badge">HALO</span>
      <span class="brand-title">Quality</span>
      <span class="brand-sub">— Chat</span>
    </div>
    """,
    unsafe_allow_html=True,
)

# --------------------- question chips ----------------------
colc, colf = st.columns([1, 1])
with colc:
    chip1 = st.button("complaint analysis — June 2025 (by portfolio)")
with colf:
    chip2 = st.button("first pass accuracy analysis")

# text box (optional)
q = st.text_input(
    "Type your question (e.g., 'complaint analysis — June 2025 by portfolio' or 'first pass accuracy analysis')",
    value="",
)

# determine which question to run
if chip1:
    q = "complaint analysis — June 2025 by portfolio"
if chip2:
    q = "first pass accuracy analysis"

# --------------------- semantic router ---------------------
# Your semantic router returns (slug, params)
try:
    from semantic_router import route  # your existing router
except Exception as e:
    st.error(f"Could not import semantic router: {e}")
    st.stop()

if not q.strip():
    st.stop()

slug, params = route(q)

# --------------------- question module dispatch -------------
# Always pass store['root'] = project root
ROOT = Path(__file__).resolve().parent
store = {"root": str(ROOT)}

try:
    mod = importlib.import_module(f"questions.{slug}")
except ModuleNotFoundError:
    st.error(f"Question module not found for slug: `{slug}`")
    st.stop()
except Exception as e:
    st.error(f"Failed to import question `{slug}`: {e}")
    st.stop()

# Each question gets (store, params, q)
try:
    _ = mod.run(store, params, q)
except Exception as e:
    st.error("This question failed.")
    st.exception(e)
