# app.py
from __future__ import annotations

import importlib
import inspect
import re
from dataclasses import asdict
from typing import Any, Dict

import pandas as pd
import streamlit as st

from core.data_store import load_store
from semantic_router import match_query, IntentMatch

# ---------- page & cache ----------

st.set_page_config(page_title="Halo Quality — Chat", layout="wide")

@st.cache_data(show_spinner=False)
def _load_store_cached():
    return load_store()

store = _load_store_cached()

# ---------- helpers ----------

_CANON = {
    # optional canonicalization hook if you ever rename files
    "complaints_per_thousand": "complaints_per_thousand",
    "rca1_portfolio_process": "rca1_portfolio_process",
    "unique_cases_mom": "unique_cases_mom",
}

def _import_question_module(slug: str):
    """
    Import questions.<slug> with a tiny canonical fallback.
    """
    slug = _CANON.get(slug, slug)
    return importlib.import_module(f"questions.{slug}")

def _run_question(slug: str, args: Dict[str, Any], user_text: str | None):
    """
    Import and execute questions.<slug>.run(store, **filtered_kwargs)
    Only forwards kwargs the function actually accepts.
    """
    mod = _import_question_module(slug)
    if not hasattr(mod, "run"):
        st.error(f"Question module '{slug}' has no run()")
        return

    sig = inspect.signature(mod.run)
    safe = {}
    for k, v in args.items():
        if k in sig.parameters:
            safe[k] = v

    # pass user_text only if accepted
    if "user_text" in sig.parameters:
        safe["user_text"] = user_text

    # always pass the store as positional first argument
    return mod.run(store, **safe)

def _status_pill(label: str, value: Any):
    col1, col2 = st.columns([1, 1.5])
    with col1:
        st.caption(label)
    with col2:
        st.write(value)

def _sidebar_status():
    st.sidebar.subheader("Data status")
    try:
        cases = store["cases"]
        complaints = store["complaints"]
        fpa = store.get("fpa", pd.DataFrame())

        st.sidebar.write("Cases rows:", len(cases))
        st.sidebar.write("Complaints rows:", len(complaints))
        st.sidebar.write("FPA rows:", len(fpa))

        # last months (rough)
        if not cases.empty:
            last_case = pd.to_datetime(cases.get("month_dt", cases.iloc[:, 0]), errors="coerce").max()
        else:
            last_case = None
        if not complaints.empty:
            last_cmp = pd.to_datetime(complaints.get("month_dt", complaints.iloc[:, 0]), errors="coerce").max()
        else:
            last_cmp = None

        if last_case is not None or last_cmp is not None:
            st.sidebar.caption(
                f"Latest Month — Cases: {last_case.strftime('%b %y') if last_case is not None else '—'} | "
                f"Complaints: {last_cmp.strftime('%b %y') if last_cmp is not None else '—'}"
            )
    except Exception as e:
        st.sidebar.error("Failed to read data store.")
        st.sidebar.exception(e)

# ---------- UI ----------

_sidebar_status()

st.title("Halo Quality — Chat")
st.caption("Hi! Ask me about cases, complaints (incl. RCA), or first-pass accuracy.")

# Quick-ask chips
c1, c2, c3 = st.columns(3)
with c1:
    if st.button("complaints per 1000 by process for portfolio London Jun 2025 to Aug 2025", use_container_width=True):
        st.session_state["free_text"] = "complaints per 1000 by process for portfolio London Jun 2025 to Aug 2025"
with c2:
    if st.button("show rca1 by portfolio for process Member Enquiry last 3 months", use_container_width=True):
        st.session_state["free_text"] = "show rca1 by portfolio for process Member Enquiry last 3 months"
with c3:
    if st.button("unique cases by process and portfolio Apr 2025 to Jun 2025", use_container_width=True):
        st.session_state["free_text"] = "unique cases by process and portfolio Apr 2025 to Jun 2025"

# input
default_text = st.session_state.get("free_text", "")
query = st.text_input(
    "Type your question (e.g., 'complaints per 1000 by process last 3 months')",
    value=default_text,
    placeholder="Ask a question…",
)

def _pretty_window(args: Dict[str, Any]) -> str:
    start = args.get("start_month")
    end = args.get("end_month")
    rel = args.get("relative_months")
    if start and end:
        s = pd.to_datetime(start)
        e = pd.to_datetime(end)
        return f"{s.strftime('%b %Y')} → {e.strftime('%b %Y')}"
    if rel:
        return f"last {rel} months"
    return "time window: auto"

def _debug_args(args: Dict[str, Any]):
    with st.expander("Parsed filters", expanded=False):
        st.json(args)

# ---------- router & execution ----------

if query:
    match: IntentMatch | None = match_query(query)
else:
    match = None

if match is None:
    st.info("Sorry—couldn’t understand that question.")
else:
    slug = match.slug
    args = dict(match.args)  # a dict with keys like start_month, end_month, relative_months, portfolio, process

    st.subheader(match.title)
    st.caption(_pretty_window(args))

    # Optional debug view
    _debug_args(args)

    try:
        _run_question(slug, args, user_text=query)
    except ModuleNotFoundError as e:
        st.error(f"Can’t load module for '{slug}'.")
        st.exception(e)
    except TypeError as e:
        # if a question’s signature changes unexpectedly, you’ll see it here;
        # the safe-kwargs filter above should prevent most of these.
        st.error("This question failed (signature mismatch).")
        st.exception(e)
    except Exception as e:
        st.error("This question failed unexpectedly.")
        st.exception(e)
