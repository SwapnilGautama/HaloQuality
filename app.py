# app.py — very small chat UI that calls semantic_router.route()
from __future__ import annotations
import streamlit as st
import pandas as pd

from core.data_store import load_store
import semantic_router as sr

st.set_page_config(page_title="Halo Quality", layout="wide")

@st.cache_data(show_spinner="Loading data…")
def _load():
    # Re-use your existing loaders (cases, complaints with RCA labels, FPA)
    return load_store(sig_cases=True, sig_complaints=True, sig_fpa=True)

store = _load()

st.title("Halo Quality — Chat (Question-per-file)")

with st.sidebar:
    st.markdown("### Data status")
    st.write(f"Cases rows: **{len(store.get('cases', pd.DataFrame())):,}**")
    st.write(f"Complaints rows: **{len(store.get('complaints', pd.DataFrame())):,}**")
    st.write(f"FPA rows: **{len(store.get('fpa', pd.DataFrame())):,}**")
    st.divider()
    st.caption("Try:")
    st.markdown("- complaints per 1000 by process for portfolio London Jun 2025 to Aug 2025")
    st.markdown('- show rca1 by portfolio for process "Member Enquiry" last 3 months')
    st.markdown("- unique cases by process and portfolio Apr 2025 to Jun 2025")
    st.markdown("- fpa fail rate by team last 3 months")
    st.markdown("- biggest drivers of case fails")

prompt = st.chat_input("Ask a question…")
if not prompt:
    st.stop()

st.chat_message("user").write(prompt)
with st.chat_message("assistant"):
    try:
        sr.route(prompt, store)   # each handler renders charts/tables
    except Exception as e:
        st.error(f"Sorry—this question failed: {e}")
