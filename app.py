import os
import sys
from datetime import date
from typing import Dict, Any, Optional, Tuple

import pandas as pd
import streamlit as st

# Make local modules importable when running on Streamlit Cloud
APP_DIR = os.path.dirname(os.path.abspath(__file__))
if APP_DIR not in sys.path:
    sys.path.append(APP_DIR)

# ---- our modules
from core.data_store import load_store  # cached loader already in your codebase
from semantic_router import match_query, IntentMatch  # <— IMPORTANT: import router


# ---------- Cache & small utils ----------

@st.cache_data(show_spinner=False)
def _load_store_cached() -> Dict[str, Any]:
    """Loads the prepared store (cases, complaints, fpa, etc.)."""
    return load_store()


def _get_global_month_bounds(store: Dict[str, Any]) -> Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]:
    """Return (min_month_key, max_month_key) from whatever data exists."""
    keys = []
    for k in ("cases", "complaints", "fpa"):
        if k in store and not store[k].empty and "month_key" in store[k].columns:
            keys.append(store[k]["month_key"].min())
            keys.append(store[k]["month_key"].max())
    if not keys:
        return None, None
    return min(keys), max(keys)


def _shift_months(ts: pd.Timestamp, n: int) -> pd.Timestamp:
    """Shift months and snap to first day of month."""
    return (ts.to_period("M") + n).to_timestamp()


def _resolve_dates(store: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Resolve date parameters from router into concrete month keys.

    Input (may contain):
      - relative_months: int  → use last N months ending at latest month in data
      - start_ym: 'YYYY-MM'   → explicit
      - end_ym:   'YYYY-MM'   → explicit

    Output (merged back):
      - start_month_key: pd.Timestamp (month start)
      - end_month_key:   pd.Timestamp (month start)
    """
    out = dict(params or {})
    mn, mx = _get_global_month_bounds(store)

    if out.get("relative_months"):
        n = max(int(out["relative_months"]), 1)
        if mx is None:
            return out
        end_key = mx.to_period("M").to_timestamp()
        start_key = _shift_months(end_key, -(n - 1))
        out["start_month_key"] = start_key
        out["end_month_key"] = end_key
        return out

    # explicit yyyy-mm range
    if out.get("start_ym"):
        try:
            out["start_month_key"] = pd.to_datetime(out["start_ym"]).to_period("M").to_timestamp()
        except Exception:
            pass
    if out.get("end_ym"):
        try:
            out["end_month_key"] = pd.to_datetime(out["end_ym"]).to_period("M").to_timestamp()
        except Exception:
            pass

    # if only one end provided, fill with latest
    if out.get("start_month_key") is not None and out.get("end_month_key") is None and mx is not None:
        out["end_month_key"] = mx.to_period("M").to_timestamp()
    if out.get("end_month_key") is not None and out.get("start_month_key") is None and mn is not None:
        # default to show single month if only end given
        out["start_month_key"] = out["end_month_key"]

    return out


def _chip(text: str, key: str) -> None:
    if st.button(text, key=key):
        st.session_state.free_text = text


def _parsed_filters_box(title: str, params: Dict[str, Any]) -> None:
    if not params:
        return
    with st.expander(title, expanded=False):
        for k, v in params.items():
            if isinstance(v, pd.Timestamp):
                st.write(f"- **{k}**: {v.strftime('%Y-%m')}")
            else:
                st.write(f"- **{k}**: {v}")


def _run_question(slug: str, store: Dict[str, Any], params: Dict[str, Any], user_text: str = "") -> None:
    """
    Dynamically import the question module by slug and run it.
    Each module must expose run(store, params, user_text=None).
    """
    try:
        if slug == "complaints_per_thousand":
            from questions import complaints_per_thousand as mod
        elif slug == "rca1_portfolio_process":
            from questions import rca1_portfolio_process as mod
        elif slug == "unique_cases_mom":
            from questions import unique_cases_mom as mod
        # OPTIONAL: add more slugs here if you’ve created them
        # elif slug == "complaints_reasons":
        #     from questions import complaints_reasons as mod
        else:
            st.warning(f"Sorry—no handler found for '{slug}'.")
            return

        mod.run(store, params, user_text=user_text)
    except Exception as e:
        st.exception(e)


# ---------- UI ----------

def _data_status(store: Dict[str, Any]) -> None:
    st.sidebar.header("Data status")
    def _count(df_key: str) -> int:
        if df_key in store and isinstance(store[df_key], pd.DataFrame):
            return len(store[df_key])
        return 0

    st.sidebar.write(f"Cases rows: **{_count('cases')}**")
    st.sidebar.write(f"Complaints rows: **{_count('complaints')}**")
    st.sidebar.write(f"FPA rows: **{_count('fpa')}**")


def main():
    st.set_page_config(page_title="Halo Quality — Chat", layout="wide")
    st.title("Halo Quality — Chat")
    st.caption("Hi! Ask me about cases, complaints (incl. RCA), or first-pass accuracy.")

    # Load store
    with st.spinner("Loading data store…"):
        store = _load_store_cached()

    _data_status(store)

    # Quick chips row
    st.write("")
    st.toggle("Reading Excel / parquet sources", value=True, key="reading_sources", disabled=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        _chip("complaints per 1000 by process for portfolio London Jun 2025 to Aug 2025", "chip_cpt")
    with col2:
        _chip("show rca1 by portfolio for process Member Enquiry last 3 months", "chip_rca")
    with col3:
        _chip("unique cases by process and portfolio Apr 2025 to Jun 2025", "chip_unique")

    # Free text box
    query = st.text_input(
        "Type your question (e.g., 'complaints per 1000 by process last 3 months')",
        key="free_text",
        value=st.session_state.get("free_text", "")
    )

    if not query:
        st.stop()

    # ---- Route user text
    match: Optional[IntentMatch] = match_query(query)

    if match is None:
        st.error("Sorry—couldn’t understand that question.")
        st.stop()

    # Resolve dates into concrete keys
    params = _resolve_dates(store, match.params or {})

    # Pretty header for the section
    st.header(match.title or match.slug.replace("_", " ").title())

    # Show parsed filters for transparency
    _parsed_filters_box("Parsed filters", params)

    # Run the question module
    _run_question(match.slug, store, params, user_text=query)


if __name__ == "__main__":
    main()
