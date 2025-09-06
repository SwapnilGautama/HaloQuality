# app.py
from __future__ import annotations

import importlib
import traceback
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd
import streamlit as st

# ---------- Safe imports (works whether files are in root or core/) ----------
def _try_import(mod_name: str, attr: Optional[str] = None):
    try:
        mod = importlib.import_module(mod_name)
        return getattr(mod, attr) if attr else mod
    except ModuleNotFoundError:
        # try core.<module>
        if not mod_name.startswith("core."):
            mod = importlib.import_module(f"core.{mod_name}")
            return getattr(mod, attr) if attr else mod
        raise

load_store = _try_import("data_store", "load_store")
semantic_router = None
try:
    semantic_router = _try_import("semantic_router")
except Exception:
    semantic_router = None  # we'll fall back to heuristics

# ---------- Page config ----------
st.set_page_config(page_title="Halo Quality — Chat", layout="wide")

THIS_DIR = Path(__file__).parent


# ---------- UI helpers ----------
def _pill(label: str):
    st.write(
        f"""
        <div style="background:#f6f9fe;border:1px solid #e3ecff;border-radius:8px;padding:10px 12px;margin:6px 0;">
            {label}
        </div>
        """,
        unsafe_allow_html=True,
    )


def _data_status(store: Dict[str, Any]):
    with st.sidebar:
        st.subheader("Data status")
        st.write(f"Cases rows: **{store.get('cases_rows', 0):,}**")
        st.write(f"Complaints rows: **{store.get('complaints_rows', 0):,}**")
        st.write(f"FPA rows: **{store.get('fpa_rows', 0):,}**")


def _parsed_filters_box(title: str, params: Dict[str, Any]):
    with st.expander("Parsed filters"):
        if params:
            st.json(params)
        else:
            st.caption("none")


def _find_question(query: str) -> Tuple[str, Dict[str, Any]]:
    """
    Try semantic_router first; if unavailable, fall back to simple heuristics.
    Returns (slug, params).
    """
    # semantic router present?
    if semantic_router and hasattr(semantic_router, "match"):
        try:
            m = semantic_router.match(query)
            if m and isinstance(m, dict) and "slug" in m:
                return m["slug"], (m.get("params") or {})
        except Exception:
            pass

    # very small heuristic fallback
    q = query.lower().strip()
    params: Dict[str, Any] = {}

    # month range like "jun 2025 to aug 2025"
    import re
    rng = re.search(r"([a-z]{3}\s*\d{4})\s*to\s*([a-z]{3}\s*\d{4})", q)
    if rng:
        params["start"] = rng.group(1).title()
        params["end"] = rng.group(2).title()

    # portfolio
    p = re.search(r"\bportfolio\s+([a-z\s]+)", q)
    if p:
        params["portfolio"] = p.group(1).strip().title()
    else:
        # If user typed "for London" etc.
        p2 = re.search(r"\bfor\s+([a-z\s]+?)\s+(jun|jul|aug|sep|oct|nov|dec|\d{4}|to|last|month)", q)
        if p2:
            params["portfolio"] = p2.group(1).strip().title()

    if "complaints per 1000" in q or "complaints per thousand" in q:
        return "complaints_per_thousand", params

    if "complaint analysis" in q or "complaints dashboard" in q or "june analysis" in q:
        # dedicated June analysis card (assumes 2025 if Month only is present)
        return "complaints_june_by_portfolio", params

    # default to complaints_per_thousand
    return "complaints_per_thousand", params


def _run_question(slug: str, params: Dict[str, Any], store: Dict[str, Any], user_text: str = ""):
    """
    Load questions/<slug>.py -> run(store, params, user_text)
    """
    mod = None
    err = None
    for mod_name in (f"questions.{slug}", f"core.questions.{slug}"):
        try:
            mod = importlib.import_module(mod_name)
            break
        except ModuleNotFoundError as e:
            err = e
    if not mod:
        raise err or ModuleNotFoundError(slug)

    # allow either run(store, params, user_text) or run(store, params)
    if hasattr(mod, "run"):
        try:
            return mod.run(store, params=params, user_text=user_text)
        except TypeError:
            return mod.run(store, params)


# ---------- Main ----------
def main():
    st.title("Halo Quality — Chat")

    # one-time loaded, normalized in data_store.py
    # We also pass "assume_year_for_complaints" so Month-only complaints work.
    store = load_store(assume_year_for_complaints=2025)
    _data_status(store)

    # quick actions
    col1, col2, col3 = st.columns([1.35, 1.35, 1.55])
    with col1:
        if st.button("complaints per 1000 by process for portfolio\nLondon Jun 2025 to Aug 2025"):
            slug = "complaints_per_thousand"
            params = {"portfolio": "London", "start": "Jun 2025", "end": "Aug 2025"}
            _parsed_filters_box("Parsed filters", params)
            st.header("Complaints per 1,000 cases")
            try:
                _run_question(slug, params, store, user_text="")
            except Exception:
                st.error("This question failed.")
                with st.expander("Traceback"):
                    st.code(traceback.format_exc())

    with col2:
        if st.button("show rca1 by portfolio for process Member\nEnquiry last 3 months"):
            slug = "rca1_portfolio_process"
            params = {"process": "Member Enquiry", "range": "last 3 months"}
            _parsed_filters_box("Parsed filters", params)
            st.header("RCA1 by Portfolio × Process — last 3 months")
            try:
                _run_question(slug, params, store, user_text="")
            except Exception:
                st.error("This question failed.")
                with st.expander("Traceback"):
                    st.code(traceback.format_exc())

    with col3:
        if st.button("unique cases by process and portfolio Apr 2025\nto Jun 2025"):
            slug = "unique_cases_mom"
            params = {"start": "Apr 2025", "end": "Jun 2025"}
            _parsed_filters_box("Parsed filters", params)
            st.header("Unique cases (MoM)")
            try:
                _run_question(slug, params, store, user_text="")
            except Exception:
                st.error("This question failed.")
                with st.expander("Traceback"):
                    st.code(traceback.format_exc())

    # free-text
    st.caption("Type your question (e.g., 'complaints per 1000 by process last 3 months')")
    query = st.text_input("", value="complaints per 1000 by process for portfolio London Jun 2025 to Aug 2025")
    if query:
        slug, params = _find_question(query)
        _parsed_filters_box("Parsed filters", params or {})
        # Titles for the two complaint questions
        if slug == "complaints_per_thousand":
            title = "Complaints per 1,000 cases"
        elif slug == "complaints_june_by_portfolio":
            title = "Complaints per 1,000 cases — June 2025 (by portfolio)"
        else:
            title = slug.replace("_", " ").title()
        st.header(title)
        try:
            _run_question(slug, params or {}, store, user_text=query)
        except Exception:
            st.error("This question failed.")
            with st.expander("Traceback"):
                st.code(traceback.format_exc())


if __name__ == "__main__":
    main()
