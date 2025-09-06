# app.py — Halo Quality (simple NL → question router)
from __future__ import annotations

import importlib
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

import pandas as pd
import streamlit as st

from core.data_store import load_store
from semantic_router import match_query, IntentMatch


# ------------------------
# Page & session defaults
# ------------------------
st.set_page_config(
    page_title="Halo Quality — Chat",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Session keys we use
for k, v in {
    "free_text": "",
    "last_intent": None,
}.items():
    st.session_state.setdefault(k, v)


# ------------------------
# Data store (cached in core)
# ------------------------
try:
    store: Dict[str, Any] = load_store()
except Exception as e:
    st.error("Failed to load data store.")
    st.exception(e)
    st.stop()


# ------------------------
# Small helpers for UI
# ------------------------
def _fmt_int(n: Optional[int]) -> str:
    if n is None:
        return "—"
    try:
        return f"{int(n):,}"
    except Exception:
        return str(n)


def _month_of(df: Optional[pd.DataFrame], col: str) -> Optional[str]:
    """
    Return latest month label (e.g., 'Aug 25') if available.
    Looks for a datetime-like column and formats to MMM YY.
    """
    if df is None or df.empty or col not in df.columns:
        return None
    try:
        dt = pd.to_datetime(df[col], errors="coerce")
        if dt.notna().any():
            m = dt.max().to_period("M").to_timestamp()
            return m.strftime("%b %y")
    except Exception:
        pass
    return None


def _show_data_status():
    cases = store.get("cases")
    complaints = store.get("complaints")
    fpa = store.get("fpa")

    cases_rows = len(cases) if isinstance(cases, pd.DataFrame) else None
    comp_rows = len(complaints) if isinstance(complaints, pd.DataFrame) else None
    fpa_rows = len(fpa) if isinstance(fpa, pd.DataFrame) else None

    # Attempt to detect month columns we prepared in core/data_store.py
    latest_cases = _month_of(cases, "month_dt") or _month_of(cases, "Create Date")
    latest_comps = _month_of(complaints, "month_dt") or _month_of(
        complaints, "Date"
    )
    latest_fpa = _month_of(fpa, "month_dt") or _month_of(fpa, "Activity Date")

    with st.sidebar:
        st.markdown("### Data status")
        st.write(f"Cases rows: **{_fmt_int(cases_rows)}**")
        st.write(f"Complaints rows: **{_fmt_int(comp_rows)}**")
        st.write(f"FPA rows: **{_fmt_int(fpa_rows)}**")
        if latest_cases or latest_comps or latest_fpa:
            st.markdown(
                f"""
Latest Month — Cases: **{latest_cases or '—'}** |
Complaints: **{latest_comps or '—'}** | FPA: **{latest_fpa or '—'}**
"""
            )

        st.divider()
        st.caption("Tip: Ask things like:")
        st.markdown("- complaints per **1000** by process for **portfolio London** **Jun 2025 to Aug 2025**")
        st.markdown("- show **rca1** by portfolio for process **Member Enquiry** **last 3 months**")
        st.markdown("- **unique cases** by process and portfolio **Apr 2025 to Jun 2025**")


_show_data_status()


# ------------------------
# Header & examples
# ------------------------
st.markdown("# Halo Quality — Chat")
st.caption("Hi! Ask me about cases, complaints (incl. RCA), or first-pass accuracy.")

ex1, ex2, ex3 = st.columns(3)
with ex1:
    if st.button("complaints per 1000 by process for portfolio London Jun 2025 to Aug 2025", use_container_width=True):
        st.session_state["free_text"] = "complaints per 1000 by process for portfolio London Jun 2025 to Aug 2025"
        st.rerun()
with ex2:
    if st.button("show rca1 by portfolio for process Member Enquiry last 3 months", use_container_width=True):
        st.session_state["free_text"] = "show rca1 by portfolio for process Member Enquiry last 3 months"
        st.rerun()
with ex3:
    if st.button("unique cases by process and portfolio Apr 2025 to Jun 2025", use_container_width=True):
        st.session_state["free_text"] = "unique cases by process and portfolio Apr 2025 to Jun 2025"
        st.rerun()

st.write("")  # small gap


# ------------------------
# Ask box
# ------------------------
query = st.text_input(
    "Type your question (e.g., 'complaints per 1000 by process last 3 months')",
    key="free_text",
)


# ------------------------
# Dispatcher
# ------------------------
def _run_question(slug: str, params: Dict[str, Any], user_text: str) -> None:
    """
    Import and run questions.<slug>.run(store, params, user_text=...)
    """
    try:
        mod = importlib.import_module(f"questions.{slug}")
    except Exception as e:
        st.error(f"Sorry—this question failed: {e}")
        st.exception(e)
        return

    if not hasattr(mod, "run"):
        st.error("Sorry—this question failed: run() not found in module.")
        return

    try:
        with st.spinner("Working..."):
            # All questions follow the simple contract: run(store, params, user_text=None)
            mod.run(store, params or {}, user_text=user_text)
    except Exception as e:
        st.error(f"Sorry—this question failed: {e}")
        st.exception(e)


# ------------------------
# Handle the query
# ------------------------
if query.strip():
    intent: Optional[IntentMatch] = None
    try:
        intent = match_query(query)
    except Exception as e:
        st.error("Sorry—couldn't understand that question.")
        st.exception(e)

    if not intent:
        st.error("Sorry—couldn't understand that question.")
    else:
        st.markdown(f"## {intent.title}")
        _run_question(intent.slug, intent.params, user_text=query)
else:
    st.info("Ask me something using the box above, or click one of the example buttons.")
