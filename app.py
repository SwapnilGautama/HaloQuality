# app.py
import importlib
import traceback
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import streamlit as st

from core.data_store import load_store
from semantic_router import match_query, IntentMatch


# ---------- Page & cache ----------

st.set_page_config(page_title="Halo Quality — Chat", layout="wide")

@st.cache_data(show_spinner=False)
def _load_store_cached():
    return load_store()

store = _load_store_cached()

# ---------- Small helpers ----------

def _import_question(slug: str):
    """Import questions.<slug> (module must be under questions/ and end with .py)."""
    try:
        return importlib.import_module(f"questions.{slug}")
    except Exception as e:
        raise ImportError(
            f"Could not import module 'questions.{slug}'. "
            f"Make sure questions/{slug}.py exists and has a 'run(store, params, user_text=\"\")' function."
        ) from e


def _run_question(slug: str, params: Dict[str, Any], user_text: str):
    """Run a question module's run(store, params, user_text)."""
    mod = _import_question(slug)
    if not hasattr(mod, "run"):
        raise AttributeError(
            f"'questions.{slug}' has no function 'run(store, params, user_text=\"\")'."
        )
    # Call with a single params dict (uniform contract).
    return mod.run(store, params, user_text=user_text)


def _pill(label: str, value: Any):
    st.markdown(
        f"""
        <div style="display:inline-block;border:1px solid #e5e7eb;border-radius:9999px;padding:4px 10px;margin-right:8px;background:#f8fafc;">
            <span style="color:#6b7280">{label}</span>
            <strong style="margin-left:6px">{value}</strong>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------- Left status column ----------

with st.sidebar:
    st.subheader("Data status")
    try:
        cases = store["cases"]
        complaints = store["complaints"]
        fpa = store["fpa"]
        _pill("Cases rows", f"{len(cases):,}")
        _pill("Complaints rows", f"{len(complaints):,}")
        _pill("FPA rows", f"{len(fpa):,}")

        # Latest months shown with the normalized month column each loader created
        def _last_month(df: pd.DataFrame, col: str) -> Optional[str]:
            if col in df.columns and not df.empty:
                try:
                    return df[col].max().strftime("%b %y")
                except Exception:
                    return None
            return None

        last_case = _last_month(cases, "month_dt")
        last_comp = _last_month(complaints, "month_dt")
        last_fpa = _last_month(fpa, "month_dt")

        if last_case or last_comp or last_fpa:
            st.caption(
                "Latest Month — "
                + " | ".join(
                    filter(
                        None,
                        [
                            f"Cases: {last_case}" if last_case else None,
                            f"Complaints: {last_comp}" if last_comp else None,
                            f"FPA: {last_fpa}" if last_fpa else None,
                        ],
                    )
                )
            )
    except Exception as e:
        st.error("Failed to load data store.")
        st.exception(e)

# ---------- Header & quick suggestions ----------

st.title("Halo Quality — Chat")
st.caption("Hi! Ask me about cases, complaints (incl. RCA), or first-pass accuracy.")

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

# ---------- Input ----------

q = st.text_input(
    "Type your question (e.g., 'complaints per 1000 by process last 3 months')",
    value=st.session_state.pop("__q", "") if "__q" in st.session_state else "",
)

# ---------- Run ----------

def _show_exception(e: Exception):
    st.error(f"Sorry—this question failed: {e}")
    with st.expander("Traceback"):
        st.code("".join(traceback.format_exc()))

if q.strip():
    # 1) route
    intent: Optional[IntentMatch] = None
    try:
        intent = match_query(q)
    except Exception as e:
        _show_exception(e)

    if intent is None:
        st.error("Sorry—couldn't understand that question.")
    else:
        # 2) title + params pills
        st.subheader(intent.title)
        if intent.params:
            pills = []
            for k, v in intent.params.items():
                if v is None or v == "":
                    continue
                _pill(k, v)

        # 3) execute
        try:
            _run_question(intent.slug, intent.params, user_text=q)
        except Exception as e:
            _show_exception(e)
