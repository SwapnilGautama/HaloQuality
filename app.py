# app.py
import sys
from pathlib import Path

import streamlit as st
import pandas as pd

# -----------------------------------------------------------------------------
# Robust imports (works whether files are in project root, core/, or package)
# -----------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
for extra in [HERE, HERE / "core", HERE.parent, HERE.parent / "core"]:
    sys.path.insert(0, str(extra))

# data_store import shim
_load_store_err = None
try:
    from data_store import load_store  # root
except ModuleNotFoundError as e1:
    try:
        from core.data_store import load_store  # ./core
    except ModuleNotFoundError as e2:
        try:
            from haloquality.core.data_store import load_store  # package path
        except ModuleNotFoundError as e3:
            _load_store_err = (e1, e2, e3)

# semantic_router import shim
_router_err = None
try:
    from semantic_router import match_query, IntentMatch  # root
except Exception as r1:
    try:
        from core.semantic_router import match_query, IntentMatch  # ./core
    except Exception as r2:
        try:
            from haloquality.core.semantic_router import match_query, IntentMatch  # package
        except Exception as r3:
            _router_err = (r1, r2, r3)

# question handlers (import after path setup)
_q_errs = {}
def _safe_import(label, modname, funcname="run"):
    try:
        m = __import__(modname, fromlist=[funcname])
        return getattr(m, funcname)
    except Exception as e:
        _q_errs[label] = e
        return None

q_complaints_per_thousand = _safe_import(
    "complaints_per_thousand", "complaints_per_thousand", "run"
)
q_rca1_portfolio_process = _safe_import(
    "rca1_portfolio_process", "rca1_portfolio_process", "run"
)
q_unique_cases_mom = _safe_import(
    "unique_cases_mom", "unique_cases_mom", "run"
)

QUESTIONS = {
    "complaints_per_thousand": q_complaints_per_thousand,
    "rca1_portfolio_process": q_rca1_portfolio_process,
    "unique_cases_mom": q_unique_cases_mom,
}

# -----------------------------------------------------------------------------
# Streamlit page
# -----------------------------------------------------------------------------
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
    if mod is None:
        err = _q_errs.get(slug)
        st.error("That question module failed to import.")
        if err:
            with st.expander("Import error details"):
                st.exception(err)
        return
    try:
        df = mod(store, params=params, user_text=user_text)
        st.dataframe(df, use_container_width=True)
    except Exception as e:
        st.error("This question failed.")
        st.exception(e)

def _import_gate():
    """Surface helpful errors in the UI if imports failed."""
    if _load_store_err:
        st.error("Could not import `data_store.load_store`.")
        with st.expander("Import attempts & errors"):
            for i, err in enumerate(_load_store_err, 1):
                st.write(f"Attempt {i}:")
                st.exception(err)
        st.stop()
    if _router_err:
        st.error("Could not import `semantic_router`.")
        with st.expander("Import attempts & errors"):
            for i, err in enumerate(_router_err, 1):
                st.write(f"Attempt {i}:")
                st.exception(err)
        st.stop()

def main():
    _import_gate()

    st.title("Halo Quality — Chat")
    st.caption("Hi! Ask me about cases, complaints (incl. RCA), or first-pass accuracy.")

    with st.status("Reading Excel / parquet sources", expanded=True):
        store = _load_store_cached()
    _data_status(store)

    # Quick action buttons (your pinned phrases)
    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        if st.button(
            "complaints per 1000 by process for portfolio London Jun 2025 to Aug 2025",
            use_container_width=True,
        ):
            st.session_state["free_text"] = (
                "complaints per 1000 by process for portfolio london jun 2025 to aug 2025"
            )
    with c2:
        if st.button(
            "show rca1 by portfolio for process Member Enquiry last 3 months",
            use_container_width=True,
        ):
            st.session_state["free_text"] = (
                "show rca1 by portfolio for process member enquiry last 3 months"
            )
    with c3:
        if st.button(
            "unique cases by process and portfolio Apr 2025 to Jun 2025",
            use_container_width=True,
        ):
            st.session_state["free_text"] = (
                "unique cases by process and portfolio apr 2025 to jun 2025"
            )

    st.write("")  # spacing

    q = st.text_input(
        "Type your question (e.g., 'complaints per 1000 by process last 3 months')",
        value=st.session_state.get("free_text", ""),
    )

    try:
        match = match_query(q)
    except Exception as e:
        st.error("Query routing failed.")
        st.exception(e)
        st.stop()

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
