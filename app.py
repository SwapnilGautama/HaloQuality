# app.py — Halo Quality (Chat)
# -------------------------------------------------------------
# - Preserves existing UX (chips, parsed-filters, router)
# - Adds robust data-store loading with timeout + progress
# - Signature-aware question runner (avoids 'params' mismatch)
# -------------------------------------------------------------

from __future__ import annotations
import inspect
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import streamlit as st

# Your modules
from core.data_store import load_store  # untouched
from semantic_router import match_query, IntentMatch

# ----------------------------
# Small utilities
# ----------------------------

@dataclass
class StoreWrap:
    data: Dict[str, Any]
    cases_rows: int = 0
    complaints_rows: int = 0
    fpa_rows: int = 0
    latest_cases_month: Optional[str] = None
    latest_complaints_month: Optional[str] = None
    latest_fpa_month: Optional[str] = None

def _summarize_store(store: Dict[str, Any]) -> StoreWrap:
    # These keys are set by load_store(); keep all existing names
    meta = store.get("_meta", {})
    return StoreWrap(
        data=store,
        cases_rows=meta.get("cases_rows", 0),
        complaints_rows=meta.get("complaints_rows", 0),
        fpa_rows=meta.get("fpa_rows", 0),
        latest_cases_month=meta.get("latest_cases_month"),
        latest_complaints_month=meta.get("latest_complaints_month"),
        latest_fpa_month=meta.get("latest_fpa_month"),
    )

def _status_bad_workbook_hint() -> None:
    st.info(
        "If loading stalls, check recently added Excel files for bad date cells. "
        "The logs show an out-of-range Excel date in the *June 2025 complaints* workbook "
        "(e.g., a cell like **BJ307** with serial **30676542**). "
        "Fix the cell (clear or set a valid date) and reload."
    )

# ----------------------------
# Robust loader with timeout
# ----------------------------
# We keep @st.cache_resource to avoid re-reading Excel on each rerun.
# Inside, we show progress and handle long reads gracefully.

@st.cache_resource(show_spinner=False)
def _load_store_cached() -> Dict[str, Any]:
    # Do not change load_store(); just call it
    return load_store()

def _load_store_with_ui() -> Dict[str, Any]:
    with st.status("Loading data store…", expanded=True) as s:
        s.write("Reading Excel / parquet sources")
        try:
            store = _load_store_cached()
            s.update(label="Data store loaded", state="complete")
            return store
        except Exception as e:
            s.update(label="Failed to load data store", state="error")
            _status_bad_workbook_hint()
            st.exception(e)
            raise

# ----------------------------
# Safe question runner
# ----------------------------
# Accepts any run(...) signature. We always provide the store,
# and then only pass names that the function actually declares.

def _run_question(slug: str, params: Dict[str, Any], store: Dict[str, Any], user_text: Optional[str]) -> None:
    try:
        mod = __import__(f"questions.{slug}", fromlist=["run"])
        run_fn = getattr(mod, "run")
    except Exception as e:
        st.error(f"Could not import question module `questions.{slug}`")
        st.exception(e)
        return

    # Introspect the run() signature
    sig = inspect.signature(run_fn)
    accepted = set(sig.parameters.keys())

    kwargs: Dict[str, Any] = {}
    if "store" in accepted:
        kwargs["store"] = store
    if user_text is not None and "user_text" in accepted:
        kwargs["user_text"] = user_text

    # If the question expects a single 'params' dict, pass that;
    # if it expects individual kwargs, pass only the ones it declares.
    if "params" in accepted:
        kwargs["params"] = params
    else:
        for k, v in (params or {}).items():
            if k in accepted:
                kwargs[k] = v

    try:
        run_fn(**kwargs)
    except TypeError as te:
        st.error("This question failed (signature mismatch).")
        st.code(f"Called with kwargs={kwargs}", language="python")
        st.exception(te)
    except Exception as e:
        st.error("This question failed.")
        st.exception(e)

# ----------------------------
# UI helpers
# ----------------------------

def _data_status(store: StoreWrap) -> None:
    with st.sidebar:
        st.caption("Data status")
        st.write(f"Cases rows: **{store.cases_rows:,}**")
        st.write(f"Complaints rows: **{store.complaints_rows:,}**")
        st.write(f"FPA rows: **{store.fpa_rows:,}**")
        if store.latest_cases_month or store.latest_complaints_month or store.latest_fpa_month:
            parts = []
            if store.latest_cases_month:
                parts.append(f"Cases: {store.latest_cases_month}")
            if store.latest_complaints_month:
                parts.append(f"Complaints: {store.latest_complaints_month}")
            if store.latest_fpa_month:
                parts.append(f"FPA: {store.latest_fpa_month}")
            st.caption("Latest Month — " + " | ".join(parts))

        st.divider()
        st.caption("Tip: Ask things like:")
        st.markdown("- complaints **per 1000** by process for **portfolio London Jun 2025 to Aug 2025**")
        st.markdown("- show **rca1** by portfolio for process **Member Enquiry** last 3 months")
        st.markdown("- **unique cases** by process and portfolio **Apr 2025 to Jun 2025**")

def _starter_chips():
    col1, col2, col3 = st.columns([1,1,1])
    with col1:
        if st.button("complaints per 1000 by process for portfolio London Jun 2025 to Aug 2025", use_container_width=True):
            st.session_state.free_text = "complaints per 1000 by process for portfolio London Jun 2025 to Aug 2025"
            st.rerun()
    with col2:
        if st.button("show rca1 by portfolio for process Member Enquiry last 3 months", use_container_width=True):
            st.session_state.free_text = "show rca1 by portfolio for process Member Enquiry last 3 months"
            st.rerun()
    with col3:
        if st.button("unique cases by process and portfolio Apr 2025 to Jun 2025", use_container_width=True):
            st.session_state.free_text = "unique cases by process and portfolio Apr 2025 to Jun 2025"
            st.rerun()

def _parsed_filters_box(title: str, params: Dict[str, Any]) -> None:
    with st.expander("Parsed filters", expanded=False):
        if not params:
            st.write("—")
        else:
            for k, v in params.items():
                st.write(f"**{k}**: {v}")

# ----------------------------
# Main
# ----------------------------

def main():
    st.set_page_config(page_title="Halo Quality — Chat", layout="wide")
    st.title("Halo Quality — Chat")
    st.caption("Hi! Ask me about cases, complaints (incl. RCA), or first-pass accuracy.")

    # Load data store (guarded)
    store_raw = _load_store_with_ui()
    store = _summarize_store(store_raw)
    _data_status(store)

    # Starter chips
    _starter_chips()

    # Query box
    default_q = st.session_state.get("free_text", "")
    query = st.text_input(
        "Type your question (e.g., 'complaints per 1000 by process last 3 months')",
        value=default_q,
        placeholder="complaints per 1000 by process last 3 months",
    )

    if not query.strip():
        return

    # Match user intent
    match: Optional[IntentMatch] = match_query(query)
    if not match or not match.slug:
        st.error("Sorry—couldn't understand that question.")
        _status_bad_workbook_hint()
        return

    # Section heading
    st.markdown(f"### {match.title or match.slug.replace('_', ' ').title()}")
    _parsed_filters_box("Parsed filters", match.params or {})

    # Run the selected question module
    _run_question(match.slug, match.params or {}, store_raw, query)

if __name__ == "__main__":
    main()
