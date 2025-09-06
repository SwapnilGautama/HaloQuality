# app.py — HaloQuality chat UI + question runner (back-compat safe)

from __future__ import annotations

import importlib
import inspect
import traceback
from typing import Dict, Tuple

import pandas as pd
import streamlit as st

from core.data_store import load_store
from semantic_router import match_intent


st.set_page_config(page_title="Halo Quality — Chat", layout="wide")

# ------------ helpers --------------

def _import_question(slug: str):
    """
    Import questions.<slug> as a module.
    """
    try:
        return importlib.import_module(f"questions.{slug}")
    except ModuleNotFoundError as e:
        # Try a canonicalized version (defensive)
        canon = slug.strip().lower().replace(" ", "_")
        return importlib.import_module(f"questions.{canon}")

def _run_question(slug: str, params: Dict, store, user_text: str):
    """
    Run a question module with stable interface:
      Preferred:   run(store, params, user_text="")
      Back-compat: run(store, **filtered_kwargs)
    """
    mod = _import_question(slug)
    run_fn = getattr(mod, "run", None)
    if run_fn is None or not callable(run_fn):
        raise RuntimeError(f"Question '{slug}' has no callable run().")

    sig = inspect.signature(run_fn)
    param_names = set(sig.parameters.keys())

    # Preferred modern signature
    if "params" in param_names:
        kwargs = {}
        if "user_text" in param_names:
            kwargs["user_text"] = user_text
        return run_fn(store, params=params, **kwargs)

    # Back-compat: filter unknown keys, never pass stray kwargs
    safe = {k: v for k, v in params.items() if k in param_names}
    if "user_text" in param_names:
        safe["user_text"] = user_text
    return run_fn(store, **safe)

def _data_status(store) -> None:
    st.sidebar.subheader("Data status")
    cases = store["cases"]
    complaints = store["complaints"]
    fpa = store["fpa"]

    st.sidebar.write(f"Cases rows: **{len(cases):,}**")
    st.sidebar.write(f"Complaints rows: **{len(complaints):,}**")
    st.sidebar.write(f"FPA rows: **{len(fpa):,}**")

    # latest month quick glance
    def _latest_month(df: pd.DataFrame, month_col: str) -> str:
        if month_col not in df.columns or df.empty:
            return "—"
        try:
            mx = pd.to_datetime(df[month_col]).dropna().max()
            return mx.strftime("%b %y") if pd.notna(mx) else "—"
        except Exception:
            return "—"

    st.sidebar.write(
        f"Latest Month — Cases: **{_latest_month(cases,'month_dt')}** | "
        f"Complaints: **{_latest_month(complaints,'month_dt')}** | "
        f"FPA: **{_latest_month(fpa,'month_dt')}**"
    )

# ------------ UI -------------

st.title("Halo Quality — Chat")

# Cached store
try:
    store = load_store()
except Exception as e:
    st.error("Failed to load data store.")
    with st.expander("Traceback"):
        st.code("".join(traceback.format_exc()))
    st.stop()

_data_status(store)

# Quick-prompt buttons (unchanged wording)
c1, c2, c3 = st.columns(3)
with c1:
    if st.button("complaints per 1000 by process for portfolio London Jun 2025 to Aug 2025"):
        st.session_state["_prompt"] = "complaints per 1000 by process for portfolio London Jun 2025 to Aug 2025"
with c2:
    if st.button("show rca1 by portfolio for process Member Enquiry last 3 months"):
        st.session_state["_prompt"] = "show rca1 by portfolio for process Member Enquiry last 3 months"
with c3:
    if st.button("unique cases by process and portfolio Apr 2025 to Jun 2025"):
        st.session_state["_prompt"] = "unique cases by process and portfolio Apr 2025 to Jun 2025"

# Main input
prompt = st.session_state.get("_prompt", "")
user_text = st.chat_input("Type your question (e.g., 'complaints per 1000 by process last 3 months')") or prompt
st.session_state["_prompt"] = ""  # consume

if user_text:
    with st.chat_message("user"):
        st.write(user_text)

    try:
        slug, params, title = match_intent(user_text)
    except Exception:
        with st.chat_message("assistant"):
            st.error("Sorry—couldn't understand that question.")
            with st.expander("Traceback"):
                st.code("".join(traceback.format_exc()))
        st.stop()

    with st.chat_message("assistant"):
        st.subheader(title)
        try:
            _run_question(slug, params, store, user_text=user_text)
        except Exception:
            st.error("Sorry—this question failed:")
            with st.expander("Traceback"):
                st.code("".join(traceback.format_exc()))
