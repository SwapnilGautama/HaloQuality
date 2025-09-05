# chat.py
from __future__ import annotations
import io
from pathlib import Path
from typing import Dict, Tuple, List

import pandas as pd
import streamlit as st

# ---- Data loaders & joiners ----
from core.loader_cases import load_cases
from core.loader_complaints import load_complaints
from core.loader_fpa import load_fpa
from core.fpa_labeller import label_fpa_comments
from core.join_cases_complaints import build_cases_complaints_join

# ---- NL question engine ----
from question_engine import run_nl  # requires question_engine/__init__.py to expose run_nl

# Optional: Plotly is used by the NL engine for charts
# If you prefer Altair or Plotly setup tweaks, adjust question_engine/aggregate.py


# ------------------------- Page & Styling -------------------------
st.set_page_config(page_title="Halo Quality — Conversational Analytics", layout="wide")

HIDE_SIDEBAR_CSS = """
<style>
    [data-testid="stSidebar"] { width: 300px !important; }
    .small-muted { color:#6b7280; font-size:0.85rem; }
    .tight-table table { font-size: 0.9rem; }
</style>
"""
st.markdown(HIDE_SIDEBAR_CSS, unsafe_allow_html=True)


# ------------------------- Helpers -------------------------
def _folder_signature(folder: str | Path, exts: Tuple[str, ...] = (".xlsx", ".xls", ".csv")) -> Tuple[Tuple[str, float], ...]:
    """
    Create a deterministic signature for a folder so Streamlit caching is invalidated
    when files are added/updated. Returns a tuple of (relative_path, mtime) entries.
    """
    p = Path(folder)
    if not p.exists():
        return tuple()
    items: List[Tuple[str, float]] = []
    for f in sorted(p.rglob("*")):
        if f.is_file() and f.suffix.lower() in exts:
            try:
                rel = str(f.relative_to(p))
            except Exception:
                rel = str(f.name)
            items.append((rel, f.stat().st_mtime))
    return tuple(items)


@st.cache_data(show_spinner=False)
def load_store(_sig_cases, _sig_complaints, _sig_fpa) -> Dict[str, pd.DataFrame]:
    """
    Load all datasets and build derived tables. Cached by folder signatures.
    """
    # Core data
    cases = load_cases("data/cases")
    complaints = load_complaints("data/complaints")

    # FPA + label failure comments
    fpa = load_fpa("data/first_pass_accuracy")
    if not fpa.empty:
        fpa = label_fpa_comments(fpa, "data/fpa_patterns.yml")

    # Build complaints × cases join (for Complaints_per_1000 etc.)
    joined_summary, rca = build_cases_complaints_join(cases, complaints)

    return {
        "cases": cases,
        "complaints": complaints,
        "fpa": fpa,
        "complaints_join": joined_summary,   # Month, Portfolio_std, ProcessName, Unique_Cases, Complaints, Complaints_per_1000
        "complaints_rca": rca,               # Month, Portfolio_std, ProcessName, RCA1, Complaints, Share
    }


def _download_df_button(df: pd.DataFrame, label: str, key: str):
    if df is None or df.empty:
        return
    buff = io.BytesIO()
    df.to_csv(buff, index=False)
    st.download_button(
        label=label,
        data=buff.getvalue(),
        file_name=f"{key}.csv",
        mime="text/csv",
        key=f"dl_{key}"
    )


def _render_payload(payload: Dict):
    # Insights
    for tip in payload.get("insights", []):
        st.markdown(f"• {tip}")

    # Visuals first (if any)
    figs = payload.get("figs", {})
    if figs:
        cols = st.columns(len(figs))
        for i, (name, fig) in enumerate(figs.items()):
            with cols[i]:
                st.plotly_chart(fig, use_container_width=True)

    # Tables (each with a download)
    tables = payload.get("tables", {})
    if tables:
        for name, df in tables.items():
            st.markdown(f"**{name.replace('_',' ').title()}**")
            st.dataframe(df, use_container_width=True, height=min(520, 80 + 28 * min(len(df), 12)))
            _download_df_button(df, "Download CSV", key=name)


# ------------------------- Sidebar (lightweight status) -------------------------
with st.sidebar:
    st.markdown("### Data status")
    sig_cases = _folder_signature("data/cases")
    sig_complaints = _folder_signature("data/complaints")
    sig_fpa = _folder_signature("data/first_pass_accuracy")

    store = load_store(sig_cases, sig_complaints, sig_fpa)

    # small counts
    cases_rows = len(store["cases"])
    comp_rows = len(store["complaints"])
    fpa_rows = len(store["fpa"])

    st.markdown(
        f"<div class='small-muted'>Cases rows: <b>{cases_rows:,}</b><br>"
        f"Complaints rows: <b>{comp_rows:,}</b><br>"
        f"FPA rows: <b>{fpa_rows:,}</b></div>",
        unsafe_allow_html=True
    )

    # show latest months detected
    def _latest_month(df: pd.DataFrame) -> str:
        if df is None or df.empty or "Month" not in df.columns:
            return "—"
        try:
            return str(sorted(df["Month"].dropna().unique())[-1])
        except Exception:
            return "—"

    st.markdown(
        f"<div class='small-muted'>Latest Month — "
        f"Cases: <b>{_latest_month(store['cases'])}</b> | "
        f"Complaints: <b>{_latest_month(store['complaints'])}</b> | "
        f"FPA: <b>{_latest_month(store['fpa'])}</b></div>",
        unsafe_allow_html=True
    )

    st.divider()
    st.caption("Tip: Ask things like:")
    st.markdown(
        "- `complaints per 1000 by process for portfolio London Jun 2025 to Aug 2025`  \n"
        "- `show rca1 by portfolio for process \"Member Enquiry\" last 3 months`  \n"
        "- `unique cases by process and portfolio Apr 2025 to Jun 2025`  \n"
        "- `show the biggest drivers of case fails`"
    )


# ------------------------- Main Chat UI -------------------------
st.title("Halo Quality — Chat")

# Initialize message history
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hi! Ask me about cases, complaints (incl. RCA), or first-pass accuracy."}
    ]

# Render history
for m in st.session_state["messages"]:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# Chat input
prompt = st.chat_input("Type your question (e.g., 'complaints per 1000 by process last 3 months')")

if prompt:
    # Show user message
    st.session_state["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Run NL engine immediately (no background tasks)
    with st.chat_message("assistant"):
        try:
            payload = run_nl(prompt, store)
            # Persist a short textual summary to history
            _summary = "; ".join(payload.get("insights", [])[:1]) or "Let’s look at that…"
            st.session_state["messages"].append({"role": "assistant", "content": _summary})
            # Render full payload (charts + tables + downloads)
            _render_payload(payload)
        except Exception as e:
            st.error(f"Sorry, I couldn't process that: {e}")
