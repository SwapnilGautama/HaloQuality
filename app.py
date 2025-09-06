# app.py
from __future__ import annotations

import importlib
from dataclasses import asdict
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import streamlit as st

# Local modules
from core.data_store import load_store
from semantic_router import match_query, IntentMatch


# ──────────────────────────────────────────────────────────────────────────────
# Streamlit setup
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Halo Quality — Chat", layout="wide", page_icon="🛠️")
st.markdown(
    """
    <style>
      .pill-btn > button { border-radius: 12px !important; }
      .muted { color: rgba(49,51,63,.6); font-size: 0.9rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ──────────────────────────────────────────────────────────────────────────────
# Data loading (cached)
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading data store…", ttl=60 * 10)
def _load_store_cached() -> Dict[str, Any]:
    # load_store() comes from core/data_store.py (already in your repo)
    return load_store()


store = _load_store_cached()


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
def _month_from_any(s: pd.Series) -> Optional[pd.Series]:
    """Coerce to datetime and return month start; None if cannot."""
    if s is None:
        return None
    try:
        return pd.to_datetime(s, errors="coerce").dt.to_period("M").dt.to_timestamp()
    except Exception:
        return None


def _first_col(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    if df is None:
        return None
    lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
        for lc, orig in lower.items():
            if cand.lower() in lc:
                return orig
    return None


def _latest_month_label(df: Optional[pd.DataFrame], date_candidates: list[str]) -> Optional[str]:
    if df is None or df.empty:
        return None
    month_col = None
    if "month_dt" in df.columns:
        month_col = "month_dt"
    else:
        col = _first_col(df, date_candidates)
        if col:
            df = df.copy()
            df["__m"] = _month_from_any(df[col])
            month_col = "__m"
    if not month_col:
        return None
    m = df[month_col].max()
    if pd.isna(m):
        return None
    return pd.Timestamp(m).strftime("%b %y")


def _count_rows(df: Optional[pd.DataFrame]) -> Optional[int]:
    if df is None:
        return None
    try:
        return int(len(df))
    except Exception:
        return None


def _run_question(slug: str, params: Dict[str, Any], store: Dict[str, Any], user_text: str) -> Any:
    """
    Import questions.<slug> and call its run().

    IMPORTANT: We pass BOTH names (params & args) so legacy modules that
    expect `args` keep working alongside newer modules that expect `params`.
    """
    mod = importlib.import_module(f"questions.{slug}")
    safe_kwargs = {
        "params": params or {},
        "args": params or {},
        "user_text": user_text or "",
    }
    return mod.run(store, **safe_kwargs)


def _chip(label: str, key: str) -> bool:
    # Small helper for pill-like buttons
    with st.container():
        return st.button(label, key=key, use_container_width=True)


def _pretty_filters(p: Dict[str, Any]) -> Dict[str, Any]:
    """Format filters for the 'Parsed filters' expander."""
    pretty = {}
    if not p:
        return pretty
    for k, v in p.items():
        if v is None:
            continue
        if "month" in k and isinstance(v, (str, pd.Timestamp)):
            try:
                ts = pd.to_datetime(v, errors="coerce")
                if not pd.isna(ts):
                    pretty[k] = ts.strftime("%Y-%m-%d")
                    continue
            except Exception:
                pass
        pretty[k] = v
    return pretty


# ──────────────────────────────────────────────────────────────────────────────
# Sidebar — Data status
# ──────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.subheader("Data status")
    cases = store.get("cases")
    complaints = store.get("complaints")
    fpa = store.get("fpa")

    # Counts
    st.write(
        f"**Cases rows:** {_count_rows(cases) if _count_rows(cases) is not None else '—'}"
    )
    st.write(
        f"**Complaints rows:** {_count_rows(complaints) if _count_rows(complaints) is not None else '—'}"
    )
    st.write(f"**FPA rows:** {_count_rows(fpa) if _count_rows(fpa) is not None else '—'}")

    # Latest months
    cases_latest = _latest_month_label(
        cases, ["Create Date", "Created Date", "Report Date", "Report_Date"]
    )
    comp_latest = _latest_month_label(
        complaints,
        [
            "Date Complaint Received - DD/MM/YY",
            "Date Complaint Received",
            "Complaint Date",
            "Created Date",
        ],
    )
    fpa_latest = _latest_month_label(fpa, ["Activity Date", "Date", "Completed Date"])

    left_bits = []
    if cases_latest:
        left_bits.append(f"Cases: {cases_latest}")
    if comp_latest:
        left_bits.append(f"Complaints: {comp_latest}")
    if fpa_latest:
        left_bits.append(f"FPA: {fpa_latest}")
    if left_bits:
        st.write(
            f"<span class='muted'>Latest Month — "
            + " | ".join(left_bits)
            + "</span>",
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.caption("Tip: Ask things like:")
    st.markdown(
        """
        - complaints per **1000** by process for **portfolio London** **Jun 2025** to **Aug 2025**  
        - show **rca1** by portfolio for process **Member Enquiry** **last 3 months**  
        - **unique cases** by process and portfolio **Apr 2025** to **Jun 2025**
        """
    )


# ──────────────────────────────────────────────────────────────────────────────
# Header
# ──────────────────────────────────────────────────────────────────────────────
st.title("Halo Quality — Chat")
st.caption("Hi! Ask me about cases, complaints (incl. RCA), or first-pass accuracy.")


# ──────────────────────────────────────────────────────────────────────────────
# Quick-ask chips (your three canonical examples)
# ──────────────────────────────────────────────────────────────────────────────
col1, col2, col3 = st.columns(3)
with col1:
    if _chip(
        "complaints per 1000 by process for portfolio London Jun 2025 to Aug 2025",
        key="pill_q1",
    ):
        st.session_state["free_text"] = "complaints per 1000 by process for portfolio London Jun 2025 to Aug 2025"
        st.rerun()
with col2:
    if _chip(
        "show rca1 by portfolio for process Member Enquiry last 3 months",
        key="pill_q2",
    ):
        st.session_state["free_text"] = "show rca1 by portfolio for process Member Enquiry last 3 months"
        st.rerun()
with col3:
    if _chip(
        "unique cases by process and portfolio Apr 2025 to Jun 2025",
        key="pill_q3",
    ):
        st.session_state["free_text"] = "unique cases by process and portfolio Apr 2025 to Jun 2025"
        st.rerun()


# ──────────────────────────────────────────────────────────────────────────────
# Free-text query box
# ──────────────────────────────────────────────────────────────────────────────
ft = st.text_input(
    "Type your question (e.g., 'complaints per 1000 by process last 3 months')",
    key="free_text",
    placeholder="complaints per 1000 by process last 3 months",
)

query = (ft or "").strip()

# Nothing asked yet – stop here.
if not query:
    st.stop()


# ──────────────────────────────────────────────────────────────────────────────
# Route the query → question module
# ──────────────────────────────────────────────────────────────────────────────
match: Optional[IntentMatch] = match_query(query)
if not match:
    st.error("Sorry—couldn't understand that question.")
    st.stop()

# Title block
st.subheader(match.title or "Question")
params = dict(match.args or {})

# Parsed filters expander
with st.expander("Parsed filters", expanded=False):
    pretty = _pretty_filters(params)
    if pretty:
        st.write(" | ".join(f"**{k}**: {v}" for k, v in pretty.items()))
    else:
        st.caption("No explicit filters were parsed.")

# ──────────────────────────────────────────────────────────────────────────────
# Execute the question module
# ──────────────────────────────────────────────────────────────────────────────
try:
    _run_question(match.slug, params, store, user_text=query)
except TypeError as te:
    # Common developer-time errors: unexpected kw, missing positional arg etc.
    st.error("This question failed (signature mismatch).")
    st.exception(te)
except Exception as ex:
    st.error("This question failed.")
    st.exception(ex)
