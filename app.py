# app.py
import os, glob, importlib
import pandas as pd
import streamlit as st

# --- Page config
st.set_page_config(page_title="Quality Chat", page_icon="💬", layout="wide")

# --- Branding (Halo = orange pill, white text; Quality = dark blue)
BRAND_HTML = """
<h1 style="margin: .2rem 0 1rem 0; font-weight:700; letter-spacing:.2px;">
  <span style="
      background:#ff7a00;
      color:#fff;
      padding:.15rem .5rem .2rem .5rem;
      border-radius:.40rem;
      display:inline-block;">
    HALO
  </span>
  <span style="color:#0b3d91;"> Quality</span>
  <span style="color:#6c757d; font-weight:600;"> — Chat</span>
</h1>
"""
st.markdown(BRAND_HTML, unsafe_allow_html=True)

# --- Utility: tiny data status for the sidebar (best-effort, optional)
@st.cache_data(show_spinner=False)
def _safe_len(path_patterns, read_excel=False):
    total = 0
    try:
        files = []
        for pat in path_patterns:
            files += glob.glob(pat)
        for f in files:
            if f.lower().endswith((".xls", ".xlsx")) or read_excel:
                df = pd.read_excel(f)
            else:
                df = pd.read_csv(f)
            total += len(df)
    except Exception:
        pass
    return total

@st.cache_data(show_spinner=False)
def sidebar_counts():
    cases_rows = _safe_len(["data/cases/*.csv", "data/cases/*.xlsx"])
    complaints_rows = _safe_len(["data/complaints/*.csv", "data/complaints/*.xlsx"])
    fpa_rows = _safe_len(["data/first_pass_accuracy/*.xlsx"], read_excel=True)
    # Also look for the dev-upload fallback if present
    if fpa_rows == 0 and os.path.exists("/mnt/data/FirstPassAccuracy_Aug'25.xlsx"):
        try:
            fpa_rows = len(pd.read_excel("/mnt/data/FirstPassAccuracy_Aug'25.xlsx"))
        except Exception:
            pass
    return cases_rows, complaints_rows, fpa_rows

with st.sidebar:
    st.subheader("Data status")
    cr, cmpr, fpar = sidebar_counts()
    st.caption(f"Cases rows: **{cr:,}**")
    st.caption(f"Complaints rows: **{cmpr:,}**")
    st.caption(f"FPA rows: **{fpar:,}**")

# --- Chips (only two, as requested)
st.write("")
c1, c2 = st.columns([1.1, 1.1])
with c1:
    chip_complaints = st.button("complaint analysis — June 2025 (by portfolio)")
with c2:
    chip_fpa = st.button("first pass accuracy analysis")

# --- Query box (single line) with chip autopopulate
default_q = ""
if chip_complaints:
    default_q = "complaint analysis — June 2025 (by portfolio)"
elif chip_fpa:
    default_q = "first pass accuracy analysis"

q = st.text_input(
    "Type your question (e.g., 'complaint analysis — June 2025 by portfolio' or 'first pass accuracy analysis')",
    value=default_q,
    placeholder="Ask about complaints or first pass accuracy…",
)

# --- Router → question module selection
import semantic_router as _router

store = {}  # simple passthrough store (kept for compatibility)
if q.strip():
    try:
        route = _router.match(q)
        slug = route.get("slug")
        params = route.get("params", {})

        if not slug:
            st.warning("Sorry—couldn't figure out which analysis to run.")
        else:
            # dynamic import and run
            mod = importlib.import_module(f"questions.{slug}")
            # every question module implements run(store, params, user_text)
            st.write("")  # spacing
            try:
                _ = mod.run(store, params, q)
            except Exception as e:
                st.error("This question failed.")
                st.exception(e)

    except Exception as e:
        st.error("Sorry—couldn't run that question.")
        st.exception(e)
