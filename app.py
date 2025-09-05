# app.py — Halo Quality Chat (safe argument passing)

from __future__ import annotations
import sys, importlib, traceback, inspect
from pathlib import Path
from typing import Dict, Any

import streamlit as st

# --- Ensure repo root is importable ---
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# --- Load your data store (unchanged) ---
from core.data_store import load_store

# --- Semantic router returns (slug, args, title) ---
from semantic_router import route_intent


def _list_question_modules():
    from pkgutil import iter_modules
    import questions  # noqa
    pkg_path = Path(questions.__file__).resolve().parent
    return sorted([m.name for m in iter_modules([str(pkg_path)])])


def _import_question_module(slug: str):
    try:
        return importlib.import_module(f"questions.{slug}")
    except Exception:
        available = _list_question_modules()
        raise ImportError(
            f"Cannot import 'questions.{slug}'. "
            f"Available: {', '.join(available) or '(none)'}\n"
            f"sys.path={sys.path}"
        )


def _run_question(slug: str, args: Dict[str, Any], store, user_text: str):
    """
    Import questions.<slug> and call run(store, ...), adapting arguments to the
    module's declared signature so older/newer question modules all work.
    """
    mod = _import_question_module(slug)
    if not hasattr(mod, "run"):
        raise AttributeError(f"Module 'questions.{slug}' has no function run(...).")

    # Raw parsed arguments from the semantic router
    raw_args = dict(args or {})

    # Some older code injected 'query' – normalize by removing it and mapping later
    raw_args.pop("query", None)

    # Inspect the target signature and only pass what it accepts
    sig = inspect.signature(mod.run)
    accepted = set(sig.parameters.keys())

    safe_args: Dict[str, Any] = {k: v for k, v in raw_args.items() if k in accepted}

    # Map user text into an appropriate parameter if present
    if "user_question" in accepted and "user_question" not in safe_args:
        safe_args["user_question"] = user_text
    elif "query" in accepted and "query" not in safe_args:
        safe_args["query"] = user_text

    # ADAPTERS for older modules:
    # Many legacy question modules expect a single dict "params" (or sometimes "filters").
    # If the callee declares it and we haven't provided it, pass the full parsed args.
    if "params" in accepted and "params" not in safe_args:
        safe_args["params"] = dict(raw_args)  # full filter set as a dict

    if "filters" in accepted and "filters" not in safe_args:
        safe_args["filters"] = dict(raw_args)  # alias for modules that prefer 'filters'

    # Optional: if a module wants 'options' or 'config', give it the same dict
    if "options" in accepted and "options" not in safe_args:
        safe_args["options"] = dict(raw_args)
    if "config" in accepted and "config" not in safe_args:
        safe_args["config"] = dict(raw_args)

    return mod.run(store, **safe_args)


# ---------- UI ----------
st.set_page_config(page_title="Halo Quality — Chat", layout="wide")

st.sidebar.header("Data status")
with st.sidebar:
    try:
        store = load_store()
        if hasattr(store, "cases") and hasattr(store.cases, "shape"):
            st.write(f"Cases rows: {store.cases.shape[0]:,}")
        if hasattr(store, "complaints") and hasattr(store.complaints, "shape"):
            st.write(f"Complaints rows: {store.complaints.shape[0]:,}")
        if hasattr(store, "fpa") and hasattr(store.fpa, "shape"):
            st.write(f"FPA rows: {store.fpa.shape[0]:,}")
    except Exception as e:
        st.error("Failed to load data store.")
        st.exception(e)
        st.stop()

st.title("Halo Quality — Chat")

# Example chips
examples = [
    "complaints per 1000 by process for portfolio London Jun 2025 to Aug 2025",
    "show rca1 by portfolio for process Member Enquiry last 3 months",
    "unique cases by process and portfolio Apr 2025 to Jun 2025",
]
cols = st.columns(len(examples))
for i, phrase in enumerate(examples):
    if cols[i].button(phrase):
        st.session_state["__prefill__"] = phrase

user_text = st.chat_input("Ask a question…")
if not user_text and st.session_state.get("__prefill__"):
    user_text = st.session_state.pop("__prefill__")

if user_text:
    with st.chat_message("user"):
        st.write(user_text)

    try:
        slug, args, title = route_intent(user_text)
    except Exception as e:
        with st.chat_message("assistant"):
            st.error(f"Failed to route the question: {e}")
        st.stop()

    with st.chat_message("assistant"):
        if title:
            st.subheader(title)
        try:
            _run_question(slug, args, store, user_text)
        except Exception as e:
            st.error(f"Sorry—this question failed: {e}")
            with st.expander("Traceback"):
                st.code("".join(traceback.format_exc()))
