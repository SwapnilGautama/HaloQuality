# app.py — Halo Quality (Streamlit) — Chat + NL router → question modules
# ----------------------------------------------------------------------
# What this file does:
# - Ensures the repo root is on sys.path so `import questions.<slug>` works on Streamlit Cloud
# - Loads your data store (core.data_store.load_store)
# - Lightweight chat UI that routes a user utterance to a question module
# - Robust dynamic import with aliases/fallbacks so legacy slugs still resolve
# ----------------------------------------------------------------------

from __future__ import annotations

import sys
import importlib
from pathlib import Path
import traceback
from typing import Dict, Any, Tuple

import pandas as pd
import streamlit as st

# --- put project root on sys.path (VERY IMPORTANT for Streamlit Cloud) ---
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# also expose package dirs if needed
if str(ROOT / "core") not in sys.path:
    sys.path.insert(0, str(ROOT / "core"))
if str(ROOT / "questions") not in sys.path:
    sys.path.insert(0, str(ROOT / "questions"))
if str(ROOT / "question_engine") not in sys.path:
    sys.path.insert(0, str(ROOT / "question_engine"))

# Now these imports are reliable
from core.data_store import load_store
from semantic_router import route_intent  # provided below

# Optional: import questions package so its __init__.py can export slugs
try:
    import questions  # noqa: F401
except Exception:
    # not fatal; dynamic import (below) still works
    pass


st.set_page_config(page_title="Halo Quality — Chat", layout="wide")


# ---------- caching ----------
@st.cache_data(show_spinner=False)
def _get_store() -> Dict[str, pd.DataFrame]:
    """
    Call your data loader. Compatible with both
    `load_store()` and `load_store(sig_cases, sig_complaints, sig_fpa)`.
    """
    try:
        return load_store()  # preferred: single convenience loader
    except TypeError:
        # older signature support
        return load_store(sig_cases=True, sig_complaints=True, sig_fpa=True)


# ---------- helper: safe dynamic import with aliases ----------
# Map legacy / alternate slugs to the module that actually implements it.
SLUG_ALIASES: Dict[str, str] = {
    # complaints / cases
    "complaints_per_1000": "complaints_per_thousand",
    "complaints_per_thousand": "complaints_per_thousand",
    "complaint_volume": "complaint_volume_rate",
    "complaint_volume_rate": "complaint_volume_rate",
    "unique_cases_mom": "unique_cases_mom",
    "mom_overview": "mom_overview",
    "corr_nps": "corr_nps",
    "rca1_portfolio_process": "rca1_portfolio_process",
    # FPA
    "fpa_fail_rate": "fpa_fail_drivers",         # legacy name → drivers module
    "fpa_fail_drivers": "fpa_fail_drivers",
}


def _import_question_module(slug: str):
    """import questions.<slug>, with alias fallback."""
    # normalize through alias map
    canonical = SLUG_ALIASES.get(slug, slug)
    try:
        return importlib.import_module(f"questions.{canonical}")
    except ModuleNotFoundError:
        # last attempt: try the raw slug (in case it truly exists)
        return importlib.import_module(f"questions.{slug}")


def _run_question(slug: str, store: Dict[str, pd.DataFrame], args: Dict[str, Any]) -> None:
    """
    Import the question module and execute its `run(store, **kwargs)` (or `main`) entry.
    Any exception is surfaced nicely in the UI.
    """
    try:
        mod = _import_question_module(slug)
        run = getattr(mod, "run", None) or getattr(mod, "main", None)
        if run is None:
            raise AttributeError(f"Module 'questions.{slug}' does not define run() or main().")
        # pass original query too, if the question wants to parse it further
        if "query" not in args:
            args["query"] = st.session_state.get("last_user_query", "")
        run(store, **args)
    except Exception as ex:
        st.error(
            f"Sorry—this question failed: {ex}\n\n"
            f"```\n{traceback.format_exc()}\n```"
        )


# ---------- UI ----------
def _sidebar_status(store: Dict[str, pd.DataFrame]) -> None:
    st.sidebar.markdown("### Data status")
    cases = store.get("cases")
    complaints = store.get("complaints")
    fpa = store.get("fpa")

    def _rows(df: pd.DataFrame | None) -> str:
        return f"{len(df):,}" if isinstance(df, pd.DataFrame) else "0"

    st.sidebar.write(
        f"Cases rows: **{_rows(cases)}** | "
        f"Complaints rows: **{_rows(complaints)}** | "
        f"FPA rows: **{_rows(fpa)}**"
    )

    # quick latest month readout if present
    def _latest_month(df: pd.DataFrame, month_col_candidates=("month", "Month", "create_month")):
        for col in month_col_candidates:
            if isinstance(df, pd.DataFrame) and col in df.columns:
                try:
                    return pd.Series(df[col]).dropna().astype(str).sort_values().iloc[-1]
                except Exception:
                    pass
        return "NaT"

    st.sidebar.caption(
        f"Latest Month — Cases: **{_latest_month(cases)}** | "
        f"Complaints: **{_latest_month(complaints)}** | "
        f"FPA: **{_latest_month(fpa)}**"
    )

    st.sidebar.markdown("---")
    st.sidebar.caption("Tip: Ask things like:")
    st.sidebar.markdown(
        """
- complaints per **1000** by process for **portfolio London** **Jun 2025 to Aug 2025**
- show **rca1** by portfolio for process **Member Enquiry**
- unique **cases** by process and portfolio **Apr 2025 to Jun 2025**
- show the **biggest drivers** of **case fails**
        """
    )


def main() -> None:
    store = _get_store()

    # header
    st.title("Halo Quality — Chat")

    # sidebar status
    _sidebar_status(store)

    # greeting
    with st.chat_message("assistant"):
        st.write("Hi! Ask me about **cases**, **complaints (incl. RCA)**, or **first-pass accuracy**.")

    # quick suggestion chips (they just inject a user query)
    suggestions = [
        "complaints per 1000 by process for portfolio London Jun 2025 to Aug 2025",
        "show rca1 by portfolio for process Member Enquiry",
        "unique cases by process last 3 months",
        "show the biggest drivers of case fails",
    ]
    st.write("")  # spacing
    cols = st.columns(len(suggestions))
    for c, s in zip(cols, suggestions):
        if c.button(s, use_container_width=True):
            st.session_state["last_user_query"] = s
            slug, args, title = route_intent(s)
            with st.chat_message("user"):
                st.write(s)
            with st.chat_message("assistant"):
                st.subheader(title or s)
                _run_question(slug, store, args)
            st.stop()

    # chat input
    user_q = st.chat_input("Type your question (e.g., 'complaints per 1000 by process last 3 months')")
    if not user_q:
        return

    st.session_state["last_user_query"] = user_q
    with st.chat_message("user"):
        st.write(user_q)

    # route NL → (slug, args)
    slug, args, title = route_intent(user_q)

    # run the question
    with st.chat_message("assistant"):
        st.subheader(title or user_q)
        _run_question(slug, store, args)


if __name__ == "__main__":
    main()
