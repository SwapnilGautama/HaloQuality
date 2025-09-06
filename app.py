# app.py
import streamlit as st
import pandas as pd

from data_store import load_store
from semantic_router import match_query, IntentMatch

from complaints_per_thousand import run as q_complaints_per_thousand
from rca1_portfolio_process import run as q_rca1_portfolio_process
from unique_cases_mom import run as q_unique_cases_mom

QUESTIONS = {
    "complaints_per_thousand": q_complaints_per_thousand,
    "rca1_portfolio_process": q_rca1_portfolio_process,
    "unique_cases_mom": q_unique_cases_mom,
}

st.set_page_config(page_title="Halo Quality — Chat", layout="wide")

@st.cache_data(show_spinner=False)
def _load_store_cached():
    return load_store()

def _data_status(store):
    st.sidebar.subheader("Data status")
    st.sidebar.write(f"Cases rows: **{store.get('cases_rows', 0)}**")
    st.sidebar.write(f"Complaints rows: **{store.get('complaints_rows', 0)}**")
    lm = store.get("last_case_month")
    if lm is not None:
        st.sidebar.write(f"Latest Month — Cases: **{lm.strftime('%b %y')}**")
    lm2 = store.get("last_complaint_month")
    if lm2 is not None:
        st.sidebar.write(f"Complaints: **{lm2.strftime('%b %y')}**")

def _parsed_filters_box(title: str, params: dict):
    with st.expander(title, expanded=False):
        if not params:
            st.caption("No filters parsed.")
        else:
            df = pd.DataFrame([params])
            st.dataframe(df, hide_index=True, use_container_width=True)

def _run_question(slug: str, params: dict, store: dict, user_text: str):
    mod = QUESTIONS.get(slug)
    if not mod:
        st.warning("Sorry—couldn't understand that question.")
        return
    try:
        df = mod(store, params=params, user_text=user_text)
        st.dataframe(df, use_container_width=True)
    except Exception as e:
        st.error("This question failed.")
        st.exception(e)

def main():
    st.title("Halo Quality — Chat")
    st.caption("Hi! Ask me about cases, complaints (incl. RCA), or first-pass accuracy.")

    with st.status("Reading Excel / parquet sources", expanded=True):
        store = _load_store_cached()
    _data_status(store)

    # quick action buttons (same phrasing as your pinned prompts)
    c1, c2, c3 = st.columns([1,1,1])
    with c1:
        if st.button("complaints per 1000 by process for portfolio London Jun 2025 to Aug 2025", use_container_width=True):
            st.session_state["free_text"] = "complaints per 1000 by process for portfolio london jun 2025 to aug 2025"
    with c2:
        if st.button("show rca1 by portfolio for process Member Enquiry last 3 months", use_container_width=True):
            st.session_state["free_text"] = "show rca1 by portfolio for process member enquiry last 3 months"
    with c3:
        if st.button("unique cases by process and portfolio Apr 2025 to Jun 2025", use_container_width=True):
            st.session_state["free_text"] = "unique cases by process and portfolio apr 2025 to jun 2025"

    st.write("")  # spacing

    # free-text input
    q = st.text_input("Type your question (e.g., 'complaints per 1000 by process last 3 months')",
                      value=st.session_state.get("free_text",""))

    match: IntentMatch = match_query(q)
    slug = match.slug
    params = match.params or {}

    if slug == "complaints_per_thousand":
        st.subheader("Complaints per 1,000 cases")
    elif slug == "rca1_portfolio_process":
        st.subheader("RCA1 by Portfolio × Process — last 3 months")
    elif slug == "unique_cases_mom":
        st.subheader("Unique cases (MoM)")

    _parsed_filters_box("Parsed filters", params)
    _run_question(slug, params, store, q)

if __name__ == "__main__":
    main()
