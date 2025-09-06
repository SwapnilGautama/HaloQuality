# app.py
from __future__ import annotations

import importlib
from pathlib import Path
import re
import pandas as pd
import streamlit as st

# --- Load data store (your data_store.py is inside core/)
try:
    from core.data_store import load_store   # <— correct location per your repo
except Exception:  # fallback if environment differs
    from data_store import load_store

PAGE_TITLE = "Halo Quality — Chat"
THIS_DIR = Path(__file__).parent

# ---------------------------
# very small router
# ---------------------------
def _choose_slug_from_query(q: str) -> str:
    ql = q.lower()
    if "complaints per 1000" in ql or "complaints per 1,000" in ql or "per 1000" in ql:
        return "complaints_per_thousand"
    if "rca1" in ql:
        return "rca1_portfolio_process"        # your existing module
    if "unique cases" in ql or "mom" in ql:
        return "unique_cases_mom"              # your existing module
    # default to complaints per 1000 (keeps UI predictable)
    return "complaints_per_thousand"

MONTH_RX = r"(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\s*[\-\/ ]?\s*(\d{2,4})"

def _extract_params(q: str) -> dict:
    """
    Extracts bare-minimum params we support:
      - portfolio <word>
      - <MMM YYYY> to <MMM YYYY>
      - 'last 3 months'
    """
    params: dict[str, str] = {}

    ql = q.lower().strip()

    # portfolio
    m = re.search(r"\bportfolio\s+([a-zA-Z ]+)", ql)
    if m:
        # stop at ' to ' or a month token
        tail = m.group(1)
        # trim at the first token that looks like a month/year
        toks = re.split(r"\bto\b", tail, maxsplit=1)
        chunk = toks[0]
        mm = re.search(MONTH_RX, chunk, flags=re.I)
        if mm:
            chunk = chunk[:mm.start()].strip()
        params["portfolio"] = chunk.strip().split()[0]  # single word works for London
        if not params["portfolio"]:
            params.pop("portfolio", None)

    # months range
    mm_all = list(re.finditer(MONTH_RX, ql, flags=re.I))
    if len(mm_all) >= 2:
        def _fmt(mo, yr):
            yr = int(yr)
            if yr < 100:
                yr += 2000
            return f"{mo.title()} {yr}"
        start_m = _fmt(mm_all[0].group(1), mm_all[0].group(2))
        end_m   = _fmt(mm_all[1].group(1), mm_all[1].group(2))
        params["start_month"] = start_m
        params["end_month"] = end_m
    elif "last 3 month" in ql:
        params["relative_months"] = 3

    return params

def _run_question(slug: str, params: dict, store, user_text: str | None = None):
    try:
        mod = importlib.import_module(f"questions.{slug}")
    except Exception as e:
        st.error(f"That question module failed to import.\n\n{e}")
        return

    try:
        title, df, note = mod.run(store, params=params, user_text=user_text)
    except Exception as e:
        st.error("This question failed.")
        with st.expander("Traceback"):
            import traceback, io
            buf = io.StringIO()
            traceback.print_exc(file=buf)
            st.code(buf.getvalue())
        return

    st.subheader(title)
    with st.expander("Parsed filters", expanded=False):
        st.write(params)

    if note:
        st.info(note)

    if isinstance(df, pd.DataFrame) and not df.empty:
        st.dataframe(df, use_container_width=True)
    else:
        st.info("No rows returned for the current filters.")

# ------------- UI -------------
st.set_page_config(page_title=PAGE_TITLE, layout="wide")
st.title(PAGE_TITLE)

# Load once and cache in session
if "store" not in st.session_state:
    st.session_state.store = load_store()

store = st.session_state.store

# Sidebar status
with st.sidebar:
    st.subheader("Data status")
    try:
        st.write(f"Cases rows: **{len(store['cases']):,}**")
    except Exception:
        st.write("Cases rows: **0**")
    try:
        st.write(f"Complaints rows: **{len(store['complaints']):,}**")
    except Exception:
        st.write("Complaints rows: **0**")

# Quick chips
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("complaints per 1000 by process for portfolio London Jun 2025 to Aug 2025"):
        query = "complaints per 1000 by process for portfolio London Jun 2025 to Aug 2025"
        params = _extract_params(query)
        _run_question(_choose_slug_from_query(query), params, store, query)
with col2:
    if st.button("show rca1 by portfolio for process Member Enquiry last 3 months"):
        query = "show rca1 by portfolio for process Member Enquiry last 3 months"
        params = _extract_params(query)
        _run_question(_choose_slug_from_query(query), params, store, query)
with col3:
    if st.button("unique cases by process and portfolio Apr 2025 to Jun 2025"):
        query = "unique cases by process and portfolio Apr 2025 to Jun 2025"
        params = _extract_params(query)
        _run_question(_choose_slug_from_query(query), params, store, query)

# Free-text
query = st.text_input(
    "Type your question (e.g., 'complaints per 1000 by process last 3 months')",
    value=""
)

if query:
    slug = _choose_slug_from_query(query)
    params = _extract_params(query)
    _run_question(slug, params, store, query)
