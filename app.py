# app.py
import streamlit as st
from core.data_store import load_store
import semantic_router as router

st.set_page_config(page_title="Halo Quality", layout="wide")

@st.cache_data(show_spinner="Loading data…")
def _load():
    return load_store(sig_cases=True, sig_complaints=True, sig_fpa=True)

store = _load()

st.title("Halo Quality — Chat")

prompt = st.chat_input("Ask a question…")
if prompt:
    st.chat_message("user").write(prompt)
    with st.chat_message("assistant"):
        try:
            router.route(prompt, store)
        except Exception as e:
            st.error(f"Sorry—this question failed: {e}")
