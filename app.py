# app.py
import sys
from pathlib import Path
import streamlit as st

# Ensure project root is importable (so "questions" and "core" resolve in Streamlit Cloud)
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.data_store import load_store
import semantic_router as router  # NL router

st.set_page_config(page_title="Halo Quality — Chat", layout="wide")

@st.cache_data(show_spinner="Loading data…")
def _load():
    # Load all signatures you’ve wired (cases, complaints, fpa)
    return load_store(sig_cases=True, sig_complaints=True, sig_fpa=True)

store = _load()

st.title("Halo Quality — Chat")

# very light “data status” line (optional)
with st.sidebar:
    st.markdown("### Data status")
    st.write(
        f"Cases rows: {len(store.get('cases', [])):,} | "
        f"Complaints rows: {len(store.get('complaints', [])):,} | "
        f"FPA rows: {len(store.get('fpa', [])):,}"
    )

prompt = st.chat_input("Ask a question…")
if prompt:
    st.chat_message("user").write(prompt)
    with st.chat_message("assistant"):
        try:
            router.route(prompt, store)
        except Exception as e:
            st.error(f"Sorry—this question failed: {e}")
