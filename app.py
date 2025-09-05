# app.py
from __future__ import annotations

import importlib
import io
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, Tuple, Optional

import pandas as pd
import streamlit as st

# --- Make sure our project root is on sys.path so "import questions.<slug>" always works
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Data store (cached inside the module)
from core.data_store import load_store
# Tiny semantic matcher (returns (slug, args, title))
from semantic_router import match as match_intent


# ------------------------- UI helpers -------------------------

def _section_title(emoji: str, title: str):
    st.markdown(
        f"<div style='display:flex;align-items:center;gap:.5rem;font-size:1.4rem;'>"
        f"<span>{emoji}</span><strong>{title}</strong></div>",
        unsafe_allow_html=True,
    )


def _fmt_month(x) -> str:
    try:
        x = pd.to_datetime(x)
        if pd.isna(x):
            return "NaT"
        return x.strftime("%b %y")
    except Exception:
        return "NaT"


def _show_data_status(store: Dict[str, pd.DataFrame]):
    st.sidebar.header("Data status")

    def _rows(df: Optional[pd.DataFrame]) -> int:
        return 0 if df is None else len(df)

    cases = store.get("cases")
    complaints = store.get("complaints")
    fpa = store.get("fpa")

    st.sidebar.write(f"Cases rows: **{_rows(cases):,}**")
    st.sidebar.write(f"Complaints rows: **{_rows(complaints):,}**")
    st.sidebar.write(f"FPA rows: **{_rows(fpa):,}**")

    def _latest_month(df: Optional[pd.DataFrame]) -> str:
        if df is None or df.empty:
            return "NaT"
        col = "month_dt" if "month_dt" in df.columns else None
        if not col:
            return "NaT"
        try:
            return _fmt_month(df[col].max())
        except Exception:
            return "NaT"

    st.sidebar.write(
        f"Latest Month — Cases: **{_latest_month(cases)}** | Complaints: **{_latest_month(complaints)}** | FPA: **{_latest_month(fpa)}**"
    )

    st.sidebar.markdown("---")
    st.sidebar.caption("Tip: Ask things like:")
    for tip in [
        "complaints per 1000 by process for portfolio London Jun 2025 to Aug 2025",
        "show rca1 by portfolio for process Member Enquiry last 3 months",
        "unique cases by process and portfolio Apr 2025 to Jun 2025",
        "show the biggest drivers of case fails",
    ]:
        st.sidebar.markdown(f"- {tip}")


# ------------------------- import & run question -------------------------

# Legacy/alias mapping so older slugs keep working
SLUG_ALIASES = {
    "fpa_fail_rate": "fpa_fail_drivers",  # legacy -> new
}

def _canonicalize(slug: str) -> str:
    s = slug.strip().lower().replace("-", "_").replace(" ", "_")
    return SLUG_ALIASES.get(s, s)


def _import_question_module(slug: str):
    """
    Import questions.<slug> robustly.
    Tries canonical form first, then the raw slug as a fallback.
    """
    canonical = _canonicalize(slug)
    try:
        return importlib.import_module(f"questions.{canonical}")
    except Exception:
        # fallback to raw slug if someone calls with non-canonical name
        return importlib.import_module(f"questions.{slug}")


def _run_question(slug: str, args: Dict[str, Any], store, user_text: str):
    """
    Always pass user_text to question modules for robust parsing.
    """
    mod = _import_question_module(slug)
    safe_args = dict(args or {})
    safe_args.setdefault("user_text", user_text or safe_args.get("query", "") or "")
    return mod.run(store, **safe_args)


def _render_output(obj: Any):
    """
    Minimal renderer that handles:
      - str messages
      - pandas DataFrames
      - dict with {"title": ..., "data": <DataFrame>, ...}
    """
    if obj is None:
        st.info("No result.")
        return

    if isinstance(obj, str):
        # Heuristic: show neutral info vs warnings
        if obj.lower().startswith(("no ", "missing", "rca", "error", "sorry")):
            st.warning(obj)
        else:
            st.info(obj)
        return

    if isinstance(obj, pd.DataFrame):
        st.dataframe(obj, use_container_width=True)
        return

    if isinstance(obj, dict):
        title = obj.get("title")
        if title:
            _section_title("🟡", title)
        df = obj.get("data")
        if isinstance(df, pd.DataFrame):
            st.dataframe(df, use_container_width=True)
        else:
            st.write(obj)
        return

    # Fallback
    st.write(obj)


# ------------------------- Page -------------------------

st.set_page_config(page_title="Halo Quality — Chat", layout="wide")
st.title("Halo Quality — Chat")

# Load the store (cached in core.data_store)
try:
    store = load_store()
except Exception as e:
    st.error("Failed to load data store.")
    with st.expander("Traceback", expanded=False):
        st.code("".join(traceback.format_exc()))
    st.stop()

_show_data_status(store)

# Quick suggestion chips
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("complaints per 1000 by process for portfolio London Jun 2025 to Aug 2025"):
        st.session_state["__q"] = "complaints per 1000 by process for portfolio London Jun 2025 to Aug 2025"
with col2:
    if st.button("show rca1 by portfolio for process Member Enquiry last 3 months"):
        st.session_state["__q"] = "show rca1 by portfolio for process Member Enquiry last 3 months"
with col3:
    if st.button("unique cases by process and portfolio Apr 2025 to Jun 2025"):
        st.session_state["__q"] = "unique cases by process and portfolio Apr 2025 to Jun 2025"

st.markdown("")

# Query box
raw_query = st.chat_input("Type your question (e.g., 'complaints per 1000 by process last 3 months')") or st.session_state.pop("__q", "")

if raw_query:
    with st.chat_message("user"):
        st.write(raw_query)

    with st.chat_message("assistant"):
        try:
            slug, args, title = match_intent(raw_query)
            if title:
                _section_title("📦", title)
            result = _run_question(slug, args, store, user_text=raw_query)
            _render_output(result)
        except Exception as e:
            st.error(f"Sorry—this question failed: {e!s}")
            with st.expander("Traceback", expanded=False):
                st.code("".join(traceback.format_exc()))
else:
    st.info("Hi! Ask me about cases, complaints (incl. RCA), or first-pass accuracy.")
