# app.py  — minimal chat UI + auto-load using your core loaders
from __future__ import annotations

import streamlit as st
import pandas as pd

# --------- Data loading from your existing core package -----------------------
from core.data_store import load_store  # uses your loaders + join logic

# Guarded import — if rapidfuzz (used by the NL parser) hasn't installed yet,
# the app will still load and show a helpful banner.
try:
    from question_engine import run_nl  # -> question_engine/__init__.py
    NL_READY = True
except Exception as e:  # pragma: no cover
    run_nl = None  # type: ignore
    NL_READY = False
    NL_IMPORT_ERROR = e

st.set_page_config(page_title="Halo Quality", layout="wide")

# ---------- Cached store (fast reloads) ---------------------------------------
@st.cache_data(show_spinner="Loading data…")
def _load() -> dict:
    # Your existing function returns:
    #   cases, complaints, fpa, joined_summary (optional), rca (optional)
    return load_store(
        sig_cases=True,
        sig_complaints=True,
        sig_fpa=True,
    )

store = _load()

# ---------- Header / status ---------------------------------------------------
st.title("Halo Quality — Chat")

def _latest_month(df: pd.DataFrame, date_col_guess: str) -> str:
    if df is None or df.empty:
        return "NaT"
    col = "month"
    if col not in df.columns:
        # try to infer
        col = date_col_guess if date_col_guess in df.columns else None
        if col is None:
            return "NaT"
        m = pd.to_datetime(df[col]).dt.strftime("%Y-%m")
    else:
        m = df[col]
    return str(sorted(m.dropna().unique())[-1]) if len(m.dropna().unique()) else "NaT"

with st.sidebar:
    c_rows = len(store.get("cases", pd.DataFrame()))
    comp_rows = len(store.get("complaints", pd.DataFrame()))
    fpa_rows = len(store.get("fpa", pd.DataFrame()))
    st.markdown("### Data status")
    st.write(f"Cases rows: **{c_rows:,}**")
    st.write(f"Complaints rows: **{comp_rows:,}**")
    st.write(f"FPA rows: **{fpa_rows:,}**")
    try:
        st.caption(
            f"Latest Month — Cases: { _latest_month(store.get('cases', pd.DataFrame()), 'Create Date') }"
            f" | Complaints: { _latest_month(store.get('complaints', pd.DataFrame()), 'Report_Date') }"
            f" | FPA: { _latest_month(store.get('fpa', pd.DataFrame()), 'Review_Date') }"
        )
    except Exception:
        pass

# ---------- NL UI -------------------------------------------------------------
if not NL_READY or run_nl is None:
    st.warning(
        "Natural-language Q&A will be available after the dependency finishes installing "
        f"or the module loads successfully. (Import detail: `{NL_IMPORT_ERROR}`)"
        if not NL_READY else
        "Natural-language module not ready."
    )
else:
    run_nl(store)
