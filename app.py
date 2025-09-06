# app.py — HaloQuality Chat (robust caller)
# ---------------------------------------------------------
# Preserves your current UI/flow and adds a resilient
# question runner that works with all question.run(...)
# signatures you’ve used (with/without `params`, kwargs only, etc.)

from __future__ import annotations

import importlib
import inspect
from typing import Any, Dict, Optional, Tuple

import streamlit as st

# local packages
from core.data_store import load_store
from semantic_router import match_query, IntentMatch


# -------------------------------
# Helpers: importing & safe caller
# -------------------------------

@st.cache_resource(show_spinner=False)
def _import_question_module(slug: str):
    """
    Import a question module from questions/<slug>.py.
    First try canonical alias if semantic router provided one;
    otherwise import direct.
    """
    # Support either "slug" or "questions.slug"
    if slug.startswith("questions."):
        module_name = slug
    else:
        module_name = f"questions.{slug}"
    return importlib.import_module(module_name)


def _safe_call_run(mod, store, params: Dict[str, Any], user_text: Optional[str]) -> Any:
    """
    Call question.run(...) regardless of its signature.

    Supports:
        run(store, params, user_text=None)
        run(store, **params)
        run(store, params)
        run(store, **params, user_text=...)
        run(params, store)   (rare, but handle)
        run(**params)        (when module pulls from global store)
    We try a small ordered set of strategies until one works.
    """
    # Make sure params is always a dict
    params = params or {}

    # Build candidate call patterns (ordered)
    candidates = []

    sig = None
    try:
        sig = inspect.signature(mod.run)  # type: ignore[attr-defined]
    except Exception:
        # If we can't introspect, just try broad patterns
        pass

    # Most common first
    candidates.append(lambda: mod.run(store, params, user_text))               # run(store, params, user_text)
    candidates.append(lambda: mod.run(store, params))                           # run(store, params)
    candidates.append(lambda: mod.run(store=store, params=params, user_text=user_text))
    candidates.append(lambda: mod.run(store=store, params=params))
    candidates.append(lambda: mod.run(store, **params))                         # run(store, **params)
    candidates.append(lambda: mod.run(store=store, **params))
    candidates.append(lambda: mod.run(**params, store=store))
    candidates.append(lambda: mod.run(**params))                                # run(**params)

    # If the function exposes 'user_text' but not 'params', prioritize kwargs path
    if sig is not None:
        ps = sig.parameters
        has_store = "store" in ps
        has_params = "params" in ps
        has_user_text = "user_text" in ps

        ordered = []
        if has_store and has_params and has_user_text:
            ordered = [
                lambda: mod.run(store, params, user_text),
                lambda: mod.run(store=store, params=params, user_text=user_text),
            ]
        elif has_store and has_params:
            ordered = [
                lambda: mod.run(store, params),
                lambda: mod.run(store=store, params=params),
            ]
            if has_user_text:
                ordered.insert(0, lambda: mod.run(store, params, user_text))
        elif has_store and not has_params:
            # Try kwargs shapes with/without user_text
            if has_user_text:
                ordered = [
                    lambda: mod.run(store, **params, user_text=user_text),
                    lambda: mod.run(store=store, **params, user_text=user_text),
                ]
            ordered += [
                lambda: mod.run(store, **params),
                lambda: mod.run(store=store, **params),
            ]
        else:
            # No explicit 'store' in signature
            if has_user_text:
                ordered = [
                    lambda: mod.run(**params, user_text=user_text),
                ]
            ordered += [
                lambda: mod.run(**params),
            ]

        # Put our introspection-based attempts first
        candidates = ordered + candidates

    last_err = None
    for attempt in candidates:
        try:
            return attempt()
        except TypeError as e:
            # Signature mismatch; try next pattern
            last_err = e
        except Exception:
            # Let question modules surface their own errors (rendered in UI)
            raise

    # If nothing matched, re-raise the most recent signature error
    if last_err:
        raise last_err

    # Fallback (should never reach)
    return None


def _run_question(match: IntentMatch, store, user_text: Optional[str]):
    """
    Given a semantic match, import and run the corresponding question module.
    """
    slug = match.slug
    params = match.params or {}

    try:
        mod = _import_question_module(slug)
    except ModuleNotFoundError as e:
        st.error(f"Sorry—couldn't load question module '{slug}'.")
        st.exception(e)
        return

    # A little UX: show parsed filters for transparency
    with st.expander("Parsed filters", expanded=False):
        if not params:
            st.write("None")
        else:
            # pretty print the normalized params
            nice = {k: (str(v) if v is not None else v) for k, v in params.items()}
            st.write(nice)

    # And run it, letting the module render into the page
    try:
        _safe_call_run(mod, store, params, user_text)
    except Exception as e:
        st.error("This question failed (signature mismatch or runtime error).")
        st.exception(e)


# -------------------------------
# Page UI
# -------------------------------

def _data_status(store) -> None:
    # Left rail: quick data stats
    st.sidebar.subheader("Data status")
    st.sidebar.write(f"Cases rows: **{store['cases_rows']:,}**")
    st.sidebar.write(f"Complaints rows: **{store['complaints_rows']:,}**")
    st.sidebar.write(f"FPA rows: **{store['fpa_rows']:,}**")

    latest_cases = store.get("latest_cases_month_label", "—")
    latest_complaints = store.get("latest_complaints_month_label", "—")
    latest_fpa = store.get("latest_fpa_month_label", "—")

    st.sidebar.markdown(
        f"Latest Month — Cases: **{latest_cases}** | "
        f"Complaints: **{latest_complaints}** | "
        f"FPA: **{latest_fpa}**"
    )

    st.sidebar.markdown("---")
    st.sidebar.caption("Tip: Ask things like:")
    st.sidebar.markdown(
        "- complaints **per 1000** by process for **portfolio London Jun 2025 to Aug 2025**\n"
        "- show **rca1** by portfolio for process **Member Enquiry** last **3 months**\n"
        "- **unique cases** by process and portfolio **Apr 2025 to Jun 2025**"
    )


def _suggestion_chips():
    cols = st.columns(3)
    with cols[0]:
        if st.button("complaints per 1000 by process for portfolio London Jun 2025 to Aug 2025"):
            st.session_state["free_text"] = "complaints per 1000 by process for portfolio London Jun 2025 to Aug 2025"
    with cols[1]:
        if st.button("show rca1 by portfolio for process Member Enquiry last 3 months"):
            st.session_state["free_text"] = "show rca1 by portfolio for process Member Enquiry last 3 months"
    with cols[2]:
        if st.button("unique cases by process and portfolio Apr 2025 to Jun 2025"):
            st.session_state["free_text"] = "unique cases by process and portfolio Apr 2025 to Jun 2025"


def main():
    st.set_page_config(page_title="Halo Quality — Chat", layout="wide")
    st.title("Halo Quality — Chat")
    st.caption("Hi! Ask me about cases, complaints (incl. RCA), or first-pass accuracy.")

    # Load data store
    try:
        store = load_store()
    except Exception as e:
        st.error("Failed to load data store.")
        st.exception(e)
        return

    # Sidebar status
    _data_status(store)

    # Suggestion buttons
    _suggestion_chips()

    # Free text input
    free_text = st.session_state.get("free_text", "")
    free_text = st.text_input(
        "Type your question (e.g., 'complaints per 1000 by process last 3 months')",
        value=free_text,
        key="free_text",
        placeholder="e.g., complaints per 1000 by process last 3 months",
    ).strip()

    if not free_text:
        st.stop()

    # Route the query
    match = match_query(free_text)

    if match is None:
        st.error("Sorry—couldn't understand that question.")
        st.info("Try something like: 'complaints per 1000 by process for portfolio London Jun 2025 to Aug 2025'.")
        st.stop()

    # Section title mirrors the matched intent title
    st.header(match.title or "Results")

    # Run matched question
    _run_question(match, store, user_text=free_text)


if __name__ == "__main__":
    main()
