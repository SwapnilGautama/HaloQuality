# app.py
from __future__ import annotations
import importlib, traceback
from pathlib import Path
import streamlit as st
import pandas as pd

# ---------- resilient import helpers ----------
def _imp(mod: str, attr: str | None = None):
    try:
        m = importlib.import_module(mod)
    except ModuleNotFoundError:
        m = importlib.import_module(f"core.{mod}")
    return getattr(m, attr) if attr else m

load_store = _imp("data_store", "load_store")
sem_router = _imp("semantic_router")

# ---------- UI ----------
st.set_page_config(page_title="Halo Quality — Chat", layout="wide")
st.title("Halo Quality — Chat")
st.caption("Hi! Ask me about cases, complaints (incl. RCA), or first-pass accuracy.")

# Load data once (assume year=2025 for complaints Report Month)
with st.spinner("Reading Excel / parquet sources"):
    store = load_store(assume_year_for_complaints=2025)
cases = store.get("cases", pd.DataFrame())
complaints = store.get("complaints", pd.DataFrame())

# Data status
with st.sidebar:
    st.header("Data status")
    st.write(f"Cases rows: **{len(cases):,}**")
    st.write(f"Complaints rows: **{len(complaints):,}**")

# Quick chips
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("complaint analysis — June 2025 (by portfolio)", use_container_width=True):
        st.session_state["q"] = "complaint analysis — June 2025 (by portfolio)"
with col2:
    st.button("show rca1 by portfolio for process Member Enquiry last 3 months", use_container_width=True)
with col3:
    st.button("unique cases by process and portfolio Apr 2025 to Jun 2025", use_container_width=True)

# Query box
default_q = st.session_state.get("q", "complaint analysis for June 2025 by portfolio")
q = st.text_input("Type your question (e.g., 'complaint analysis for June 2025 by portfolio')
