from __future__ import annotations
import sys
from pathlib import Path
import streamlit as st
import pandas as pd

# local imports
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

from core.data_store import load_processed, latest_month, available_months
from questions import intent_router
from questions import corr_nps

st.set_page_config(page_title="Halo Quality", layout="wide")

# ---------- Load data (build processed if needed) ----------
@st.cache_data(show_spinner=True)
def _boot_store():
    return load_processed(rebuild_if_missing=True)

store = _boot_store()
complaints, cases, survey = store["complaints"], store["cases"], store["survey"]
latest = latest_month(complaints) or latest_month(cases) or latest_month(survey)

# ---------- Header ----------
st.markdown("## Conversational Analytics Assistant")
st.caption("We auto-load the latest files from `data/` (multi-month supported). Try: "
           "*“complaints nps correlation”* or add a month like *“complaints nps correlation 2025-06”* and press **Enter**.")

# data status
c_count = len(complaints)
k_count = len(cases)
s_count = len(survey)
st.caption(f"Data status — Complaints: **{c_count}** • Cases: **{k_count}** • Survey: **{s_count}** • "
           f"Months: {', '.join(available_months(complaints, cases, survey)) or '—'}")

# ---------- Prompt ----------
st.session_state.setdefault("prompt", "")
st.session_state.setdefault("do_run", False)

def _on_enter():
    st.session_state["do_run"] = True

st.text_input("Ask a question",
              key="prompt",
              placeholder="e.g., complaints nps correlation 2025-06",
              label_visibility="collapsed",
              on_change=_on_enter)

# ---------- Run intent ----------
def run_prompt(prompt: str):
    qid = intent_router.route(prompt)
    if not qid:
        st.info("I can answer: **complaints vs NPS correlation**. "
                "Try: *“complaints nps correlation”* (optionally add YYYY-MM).")
        return
    month = intent_router.extract_month(prompt, latest)
    if not month:
        st.warning("I couldn't detect a month and couldn't infer a latest month from the data.")
        return

    if qid == "corr_nps":
        corr_nps.run(store, month)

if st.session_state.get("do_run"):
    st.session_state["do_run"] = False
    run_prompt(st.session_state.get("prompt", ""))
