# app.py
from __future__ import annotations

import re
import importlib
from pathlib import Path
import traceback

import pandas as pd
import streamlit as st

# ---------------------------
# Resilient imports (root -> core fallback)
# ---------------------------
try:
    from data_store import load_store
except ModuleNotFoundError:
    from core.data_store import load_store

try:
    from semantic_router import match_query, IntentMatch
except ModuleNotFoundError:
    from core.semantic_router import match_query, IntentMatch


# ---------------------------
# Helpers
# ---------------------------
MONTH_RX = re.compile(
    r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+(\d{4})",
    re.I,
)

def _safe_month(dt: pd.Timestamp) -> pd.Timestamp:
    return pd.Timestamp(year=int(dt.year), month=int(dt.month), day=1)

def parse_months_from_text(text: str, default_start: pd.Timestamp, default_end: pd.Timestamp):
    hits = MONTH_RX.findall(text or "")
    if len(hits) >= 1:
        def to_ts(m, y): return _safe_month(pd.to_datetime(f"1 {m} {y}", dayfirst=True))
        if len(hits) == 1:
            m1, y1 = hits[0]
            start = to_ts(m1, y1)
            end = start
        else:
            m1, y1 = hits[0]; m2, y2 = hits[1]
            start = to_ts(m1, y1)
            end = to_ts(m2, y2)
            if end < start:
                start, end = end, start
        return start, end
    return default_start, default_end

def parse_portfolio_from_text(text: str) -> str | None:
    m = re.search(r"\bportfolio\s+([A-Za-z\- ]+)", text or "", flags=re.I)
    if m:
        return m.group(1).strip()
    for cand in ["London", "Chichester", "NorthWest", "Scotland", "Leatherhead", "Exeter", "BAES-Leatherhead"]:
        if re.search(rf"\b{re.escape(cand)}\b", text or "", flags=re.I):
            return cand
    return None


# ---------------------------
# UI
# ---------------------------
st.set_page_config(page_title="Halo Quality — Chat", layout="wide")
st.title("Halo Quality — Chat")
st.caption("Hi! Ask me about cases, complaints (incl. RCA), or first-pass accuracy.")

# Load data store (cached)
with st.spinner("Reading Excel / parquet sources"):
    store = load_store()

# Sidebar status
with st.sidebar:
    st.subheader("Data status")
    st.write(f"Cases rows: **{len(store['cases']):,}**")
    st.write(f"Complaints rows: **{len(store['complaints']):,}**")
    if "fpa" in store:
        st.write(f"FPA rows: **{len(store['fpa']):,}**")

# Suggested prompts
c1, c2, c3 = st.columns(3)
with c1:
    st.button(
        "complaints per 1000 by process for portfolio\nLondon Jun 2025 to Aug 2025",
        key="btn_cpt",
        use_container_width=True,
        on_click=lambda: st.session_state.setdefault(
            "free_text",
            "complaints per 1000 by process for portfolio London Jun 2025 to Aug 2025",
        ),
    )
with c2:
    st.button(
        "show rca1 by portfolio for process Member Enquiry last 3 months",
        key="btn_rca",
        use_container_width=True,
        on_click=lambda: st.session_state.setdefault(
            "free_text", "show rca1 by portfolio for process Member Enquiry last 3 months"
        ),
    )
with c3:
    st.button(
        "unique cases by process and portfolio Apr 2025 to Jun 2025",
        key="btn_uc",
        use_container_width=True,
        on_click=lambda: st.session_state.setdefault(
            "free_text", "unique cases by process and portfolio Apr 2025 to Jun 2025"
        ),
    )

# Free text box
text = st.text_input(
    "Type your question (e.g., 'complaints per 1000 by process last 3 months')",
    key="free_text",
    placeholder="complaints per 1000 by process for portfolio London Jun 2025 to Aug 2025",
)

# Route to question
match: IntentMatch = match_query(text)

# Default month window from cases (if available)
if store["cases"].empty:
    end_default = _safe_month(pd.Timestamp.today())
else:
    end_default = _safe_month(store["cases"]["date"].max())
start_default = _safe_month((end_default - pd.offsets.MonthBegin(2)))

# Extract filters from text
portfolio = match.params.get("portfolio") or parse_portfolio_from_text(text) or "London"
start_m, end_m = parse_months_from_text(text, start_default, end_default)

params = {
    "portfolio": portfolio,
    "start_month": str(start_m.date()),
    "end_month": str(end_m.date()),
}

with st.expander("Parsed filters", expanded=False):
    st.write(f"start_month: {params['start_month']} | end_month: {params['end_month']}")
    st.write(f"portfolio: {params['portfolio']}")

st.subheader(match.title)

# Run the selected question module from questions/* (root package)
def _run_question(slug: str, params: dict, user_text: str):
    try:
        mod = importlib.import_module(f"questions.{slug}")
        with st.spinner("Working..."):
            return mod.run(store, params=params, user_text=user_text)
    except Exception as ex:
        st.error("This question failed.")
        with st.expander("Traceback"):
            st.code("".join(traceback.format_exception(type(ex), ex, ex.__traceback__)))

_run_question(match.slug, params, text)
