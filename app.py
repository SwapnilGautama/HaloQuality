# app.py
from __future__ import annotations

import importlib
import inspect
from typing import Any, Dict, Tuple, Optional

import pandas as pd
import streamlit as st

# ---- Data store -------------------------------------------------------------
# Uses your existing loader; no behavior change.
from core.data_store import load_store  # cached inside the module

# ---- Semantic matcher -------------------------------------------------------
# Works with either:
#  - IntentMatch dataclass providing .slug, .title, .params
#  - legacy tuple (slug, params, title) or (slug, params)
try:
    from semantic_router import match_query, IntentMatch  # type: ignore
except Exception:  # very defensive; if the module changes we still run
    IntentMatch = object  # sentinel


# ----------------------- helpers: safe match access --------------------------
def _unpack_match(match: Any) -> Tuple[str, Dict[str, Any], str]:
    """
    Return (slug, params, title) for either an IntentMatch dataclass,
    a dict-like object, or the legacy tuple style.
    Never raises; falls back to sensible defaults.
    """
    slug = ""
    params: Dict[str, Any] = {}
    title = ""

    if match is None:
        return slug, params, title

    # IntentMatch dataclass
    if hasattr(match, "slug"):
        slug = getattr(match, "slug", "") or ""
        title = getattr(match, "title", "") or ""
        p = getattr(match, "params", None)
        if isinstance(p, dict):
            params = p
        return slug, params, title

    # dict-like
    if isinstance(match, dict):
        slug = str(match.get("slug", "")) or ""
        title = str(match.get("title", "")) or ""
        p = match.get("params", {})
        if isinstance(p, dict):
            params = p
        return slug, params, title

    # tuple legacy: (slug, params, title?) or (slug, params)
    if isinstance(match, (tuple, list)) and len(match) >= 2:
        slug = str(match[0]) or ""
        if isinstance(match[1], dict):
            params = match[1]
        if len(match) >= 3:
            title = str(match[2]) or ""
        return slug, params, title

    # Fallback
    return slug, params, title


# ----------------------- helpers: safe kwargs for run() ----------------------
def _build_kwargs(fn, store: Dict[str, Any], user_text: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Only pass what the target function actually accepts.
    Eliminates 'unexpected keyword argument' / signature mismatches.
    """
    sig = inspect.signature(fn)
    allowed = set(sig.parameters.keys())
    kw: Dict[str, Any] = {}

    if "store" in allowed:
        kw["store"] = store
    if "user_text" in allowed:
        kw["user_text"] = user_text

    # If the function has a single catch-all **kwargs parameter, we can pass params wholesale.
    has_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())

    if "params" in allowed:
        # Function expects a single 'params' dict
        kw["params"] = params or {}
    elif has_var_kw:
        # Function can take anything; pass params wholesale
        kw.update(params or {})
    else:
        # Match individual named parameters if they exist in params
        for name in allowed:
            if name in ("store", "user_text"):  # already handled
                continue
            if params and (name in params):
                kw[name] = params[name]

    return kw


# ----------------------- UI helpers ------------------------------------------
def _data_status(store: Dict[str, Any]) -> None:
    st.sidebar.header("Data status")

    def _rows(df_or_rows, key_hint: str) -> int:
        try:
            if isinstance(df_or_rows, pd.DataFrame):
                return int(df_or_rows.shape[0])
            if isinstance(df_or_rows, (int, float)) and not pd.isna(df_or_rows):
                return int(df_or_rows)
        except Exception:
            pass
        # Also check common key forms like 'cases_rows'
        return int(store.get(f"{key_hint}_rows", 0) or 0)

    cases_rows = _rows(store.get("cases"), "cases")
    complaints_rows = _rows(store.get("complaints"), "complaints")
    fpa_rows = _rows(store.get("fpa"), "fpa")

    st.sidebar.write(f"**Cases rows:** {cases_rows:,}")
    st.sidebar.write(f"**Complaints rows:** {complaints_rows:,}")
    st.sidebar.write(f"**FPA rows:** {fpa_rows:,}")

    latest_cases = store.get("latest_cases_label")
    latest_complaints = store.get("latest_complaints_label")
    latest_fpa = store.get("latest_fpa_label")
    if any([latest_cases, latest_complaints, latest_fpa]):
        st.sidebar.write(
            f"\nLatest Month — Cases: {latest_cases or '—'} | "
            f"Complaints: {latest_complaints or '—'} | "
            f"FPA: {latest_fpa or '—'}"
        )


def _parsed_filters_box(title: str, params: Optional[Dict[str, Any]]) -> None:
    safe_params = params or {}
    if not isinstance(safe_params, dict) or not safe_params:
        return
    with st.expander(title, expanded=False):
        for k, v in safe_params.items():
            st.write(f"**{k}**: {v}")


def _nice_title(fallback: str, slug: str, explicit_title: str) -> str:
    if explicit_title:
        return explicit_title
    if fallback:
        return fallback
    # a tiny prettifier
    return slug.replace("_", " ").title() if slug else "Result"


def _run_question(slug: str, params: Dict[str, Any], store: Dict[str, Any], user_text: str) -> None:
    if not slug:
        st.error("Sorry—couldn't understand that question.")
        return

    try:
        mod = importlib.import_module(f"questions.{slug}")
    except Exception as e:
        st.error(f"I couldn't find a question module for `{slug}`.")
        st.exception(e)
        return

    if not hasattr(mod, "run"):
        st.error(f"`questions/{slug}.py` is missing a `run()` function.")
        return

    run_fn = getattr(mod, "run")
    try:
        kwargs = _build_kwargs(run_fn, store, user_text, params)
        run_fn(**kwargs)
    except Exception as e:
        st.error("Sorry—this question failed.")
        st.exception(e)


# ----------------------- Main -------------------------------------------------
def main() -> None:
    st.set_page_config(page_title="Halo Quality — Chat", layout="wide")
    st.title("Halo Quality — Chat")
    st.caption("Hi! Ask me about cases, complaints (incl. RCA), or first-pass accuracy.")

    # Load data store (cached inside core.data_store)
    with st.status("Reading Excel / parquet sources", expanded=False):
        store = load_store()

    # Sidebar status (robust to missing keys)
    try:
        _data_status(store)
    except Exception:
        # Never block the app if status fails
        pass

    # Quick prompts row
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("complaints per 1000 by process for portfolio London Jun 2025 to Aug 2025"):
            st.session_state["free_text"] = "complaints per 1000 by process for portfolio London Jun 2025 to Aug 2025"
    with c2:
        if st.button("show rca1 by portfolio for process Member Enquiry last 3 months"):
            st.session_state["free_text"] = "show rca1 by portfolio for process Member Enquiry last 3 months"
    with c3:
        if st.button("unique cases by process and portfolio Apr 2025 to Jun 2025"):
            st.session_state["free_text"] = "unique cases by process and portfolio Apr 2025 to Jun 2025"

    # Free-text input
    query = st.text_input(
        "Type your question (e.g., 'complaints per 1000 by process last 3 months')",
        value=st.session_state.get("free_text", ""),
    )

    if not query:
        return

    # Match the query
    try:
        match = match_query(query)
    except Exception as e:
        st.error("Sorry—couldn't understand that question.")
        st.exception(e)
        return

    slug, params, title = _unpack_match(match)

    # Title + parsed filters (never crashes if params are absent)
    st.header(_nice_title("Result", slug, title))
    _parsed_filters_box("Parsed filters", params)

    # Execute the question module safely
    _run_question(slug, params, store, user_text=query)


if __name__ == "__main__":
    main()
