# app.py — Halo Quality Chat

from __future__ import annotations

import importlib
import inspect
from dataclasses import dataclass
from typing import Any, Dict, Optional

import pandas as pd
import streamlit as st

# Data and routing
from core.data_store import load_store
from semantic_router import match_query, IntentMatch


# -----------------------------
# Helpers
# -----------------------------

def _fmt_month_label(dt: Optional[pd.Timestamp]) -> str:
    if dt is None or pd.isna(dt):
        return "—"
    # MMM YY format
    return pd.Timestamp(dt).strftime("%b %y")


def _data_status(store: Dict[str, Any]) -> None:
    """Sidebar data status that works whether or not summary keys exist."""
    def _rows(key_count: str, key_df: str) -> int:
        if isinstance(store.get(key_count), int):
            return int(store[key_count])
        df = store.get(key_df)
        try:
            return int(len(df)) if df is not None else 0
        except Exception:
            return 0

    cases_rows = _rows("cases_rows", "cases")
    complaints_rows = _rows("complaints_rows", "complaints")
    fpa_rows = _rows("fpa_rows", "fpa")

    latest_cases = store.get("latest_cases_month_label")
    latest_complaints = store.get("latest_complaints_month_label")
    latest_fpa = store.get("latest_fpa_month_label")

    st.sidebar.subheader("Data status")
    st.sidebar.write(f"Cases rows: **{cases_rows:,}**")
    st.sidebar.write(f"Complaints rows: **{complaints_rows:,}**")
    st.sidebar.write(f"FPA rows: **{fpa_rows:,}**")

    if any([latest_cases, latest_complaints, latest_fpa]):
        st.sidebar.markdown(
            f"Latest Month — Cases: **{latest_cases or '—'}** | "
            f"Complaints: **{latest_complaints or '—'}** | "
            f"FPA: **{latest_fpa or '—'}**"
        )

    st.sidebar.markdown("---")
    st.sidebar.caption("Tip: Ask things like:")
    st.sidebar.markdown(
        "- complaints **per 1000** by process for **portfolio London Jun 2025 to Aug 2025**\n"
        "- show **rca1** by portfolio for process **Member Enquiry** last **3 months**\n"
        "- **unique cases** by process and portfolio **Apr 2025 to Jun 2025**"
    )


def _import_question_module(slug: str):
    """Import questions.<slug> with a clear error if missing."""
    try:
        return importlib.import_module(f"questions.{slug}")
    except ModuleNotFoundError as e:
        raise RuntimeError(f"Question module not found: questions.{slug}") from e


def _call_question_run(mod, store: Dict[str, Any], params: Dict[str, Any], user_text: str):
    """
    Call `mod.run(...)` in a way that works across slightly different signatures:
      - run(store, params, user_text)
      - run(store, params)
      - run(store, **params)
      - run(store, user_text=...)
      - run(**kwargs) where kwargs includes store/params fields
    """
    func = getattr(mod, "run", None)
    if not callable(func):
        raise RuntimeError(f"`run` function not found in {mod.__name__}")

    sig = inspect.signature(func)
    names = list(sig.parameters.keys())

    # Build the most-compatible kwargs first
    kwargs: Dict[str, Any] = {}
    if "store" in names:
        kwargs["store"] = store
    if "user_text" in names:
        kwargs["user_text"] = user_text

    # Prefer passing a single `params` dict if the function expects it;
    # otherwise expand the dict into keyword args.
    if "params" in names:
        kwargs["params"] = params or {}
        try:
            return func(**kwargs)
        except TypeError:
            # Some modules may still want expanded kwargs; try that path.
            kwargs.pop("params", None)
            kwargs.update(params or {})
            return func(**kwargs)
    else:
        kwargs.update(params or {})
        try:
            return func(**kwargs)
        except TypeError:
            # As a final fallback, try giving params as a single dict
            # if the function happens to accept **kwargs but expects `params`.
            kwargs.pop("store", None)
            kwargs.pop("user_text", None)
            return func(store, params, user_text)  # last-resort
            

def _parsed_filters_block(params: Dict[str, Any]) -> None:
    """Small expander to show how the query was interpreted."""
    if not params:
        return
    with st.expander("Parsed filters", expanded=False):
        # Show a compact, stable order if present
        priority = [
            "relative_months",
            "start_month",
            "end_month",
            "portfolio",
            "process",
            "process_filter",
            "range_label",
        ]
        shown = set()
        lines = []

        for k in priority:
            if k in params:
                lines.append(f"- **{k}**: {params[k]}")
                shown.add(k)
        # Any remaining keys
        for k, v in params.items():
            if k not in shown:
                lines.append(f"- **{k}**: {v}")

        st.markdown("\n".join(lines))


# -----------------------------
# UI & main flow
# -----------------------------

def _pills():
    # Example suggestions as "chips"
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        if st.button("complaints per 1000 by process for portfolio London Jun 2025 to Aug 2025", use_container_width=True):
            st.session_state.free_text = "complaints per 1000 by process for portfolio London Jun 2025 to Aug 2025"
    with col2:
        if st.button("show rca1 by portfolio for process Member Enquiry last 3 months", use_container_width=True):
            st.session_state.free_text = "show rca1 by portfolio for process Member Enquiry last 3 months"
    with col3:
        if st.button("unique cases by process and portfolio Apr 2025 to Jun 2025", use_container_width=True):
            st.session_state.free_text = "unique cases by process and portfolio Apr 2025 to Jun 2025"


def main():
    st.set_page_config(page_title="Halo Quality — Chat", layout="wide")
    st.markdown("# Halo Quality — Chat")
    st.caption("Hi! Ask me about cases, complaints (incl. RCA), or first-pass accuracy.")

    # Load data
    with st.spinner("Loading data store…"):
        store = load_store()

    # Sidebar status (safe)
    _data_status(store)

    # Suggestion “chips”
    _pills()

    # Query input
    default_text = st.session_state.get("free_text", "")
    user_text = st.text_input(
        "Type your question (e.g., 'complaints per 1000 by process last 3 months')",
        value=default_text,
        label_visibility="collapsed",
        placeholder="Type your question (e.g., 'complaints per 1000 by process last 3 months')",
    )
    # Clear the chip value after rendering the input
    st.session_state.free_text = ""

    if not user_text.strip():
        return

    # Route
    match: Optional[IntentMatch] = match_query(user_text)

    if not match or not match.slug:
        st.error("Sorry—couldn't understand that question.")
        return

    # Header for the selected question
    st.markdown(f"## {match.title or 'Question'}")

    # Show parsed filters
    _parsed_filters_block(match.params or {})

    # Run the question module
    try:
        mod = _import_question_module(match.slug)
        _call_question_run(mod, store, match.params or {}, user_text=user_text)
    except RuntimeError as e:
        st.error(str(e))
    except TypeError as e:
        # Signature mismatch safety net
        st.error("This question failed: (signature mismatch).")
        with st.expander("Traceback"):
            st.exception(e)
    except Exception as e:
        st.error("This question failed.")
        with st.expander("Traceback"):
            st.exception(e)


if __name__ == "__main__":
    main()
