# app.py
import re
from types import SimpleNamespace

import streamlit as st
import pandas as pd
import plotly.express as px

# --- our modules
from core.data_store import load_store          # your existing loader (cached)
from question_engine import run_nl               # we export run_nl in __init__.py

st.set_page_config(page_title="Halo Quality — Chat", layout="wide")

# -------------------------------
# helpers
# -------------------------------
def _fmt_month(p):
    """Render Periods or timestamps nicely."""
    if isinstance(p, pd.Period):
        return p.strftime("%b %y")
    if isinstance(p, pd.Timestamp):
        return p.strftime("%b %y")
    return str(p)

def _left_status(store):
    st.markdown("### Data status")
    cases_rows = len(store["cases"]) if store.get("cases") is not None else 0
    comp_rows  = len(store["complaints"]) if store.get("complaints") is not None else 0
    fpa_rows   = len(store["fpa"]) if store.get("fpa") is not None else 0

    months = store.get("months", {})
    lm_cases = months.get("cases_latest")
    lm_comp  = months.get("complaints_latest")
    lm_fpa   = months.get("fpa_latest")

    st.write(f"**Cases rows:** {cases_rows:,}")
    st.write(f"**Complaints rows:** {comp_rows:,}")
    st.write(f"**FPA rows:** {fpa_rows:,}")
    st.write(
        f"**Latest Month** — Cases: {_fmt_month(lm_cases)} | "
        f"Complaints: {_fmt_month(lm_comp)} | FPA: {_fmt_month(lm_fpa)}"
    )

    st.markdown("---")
    st.caption("Tip: Ask things like:")
    st.markdown(
        """
- complaints per **1000** by **process** for **portfolio London** **Jun 2025 to Aug 2025**
- show **rca1** by **portfolio** for process **Member Enquiry**
- unique **cases** by process and portfolio **Apr 2025 to Jun 2025**
- show the **biggest drivers** of **case fails**
        """
    )

def _chip(text):
    st.chat_message("user").write(text)

def _render_payload(payload: dict):
    """Draw whatever the resolver returned."""
    kind = payload.get("kind")

    if kind == "text":
        st.chat_message("assistant").write(payload.get("text", ""))
        return

    if kind == "table":
        msg = st.chat_message("assistant")
        df = payload.get("df")
        caption = payload.get("caption", "")
        if df is None or df.empty:
            msg.warning("No data after applying filters.")
        else:
            if caption:
                msg.caption(caption)
            msg.dataframe(df, use_container_width=True)
        return

    if kind == "figure":
        msg = st.chat_message("assistant")
        fig = payload.get("fig")
        caption = payload.get("caption", "")
        if caption:
            msg.caption(caption)
        if fig is None:
            msg.warning("No chart was produced.")
        else:
            msg.plotly_chart(fig, use_container_width=True)
        # optional extra table if provided
        df = payload.get("df")
        if isinstance(df, pd.DataFrame) and not df.empty:
            st.chat_message("assistant").dataframe(df, use_container_width=True)
        return

    # Multi payload (list of parts)
    if kind == "multi":
        parts = payload.get("parts", [])
        for part in parts:
            _render_payload(part)
        return

    # Fallback
    st.chat_message("assistant").write("I ran but had nothing to show — try rephrasing?")

# -------------------------------
# app UI
# -------------------------------
# Cache key signals for the loader. Bump these to force refresh.
sig_cases = "v5"
sig_complaints = "v5"
sig_fpa = "v5"

@st.cache_data(show_spinner=False)
def _load_once(sig_cases, sig_complaints, sig_fpa):
    return load_store(sig_cases, sig_complaints, sig_fpa)

store = _load_once(sig_cases, sig_complaints, sig_fpa)

# Sidebar status
with st.sidebar:
    _left_status(store)

st.title("Halo Quality — Chat")

# seed assistant
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        dict(role="assistant",
             content="Hi! Ask me about cases, complaints (incl. RCA), or first-pass accuracy.")
    ]

# render history
for m in st.session_state.chat_history:
    st.chat_message(m["role"]).write(m["content"])

# quick chips (they post as user messages)
col = st.container()
_ = col  # for symmetry if we expand later
_chip("complaints per 1000 by process for portfolio London Jun 2025 to Aug 2025")
_chip("show rca1 by portfolio for process Member Enquiry")

# chat input
user_q = st.chat_input("Type your question (e.g., 'complaints per 1000 by process last 3 months')")

if user_q:
    # echo user
    st.session_state.chat_history.append(dict(role="user", content=user_q))
    st.chat_message("user").write(user_q)

    # run NL
    try:
        with st.spinner("Let’s look at that…"):
            payload = run_nl(user_q, store)
    except Exception as e:
        st.chat_message("assistant").error(f"Sorry, I hit an error: {e}")
    else:
        # keep a plain-text breadcrumb for history
        st.session_state.chat_history.append(dict(role="assistant", content=""))
        # draw result now
        _render_payload(payload)
