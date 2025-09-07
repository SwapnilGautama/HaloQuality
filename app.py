# -*- coding: utf-8 -*-
# app.py — main entrypoint

from __future__ import annotations
import importlib
from pathlib import Path
from typing import Dict, Optional, Tuple

import streamlit as st

# local
import semantic_router as sem_router

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"

# --------------------------------------------------------------------------------------
# Styling (brand + permanently hide any sidebar)
# --------------------------------------------------------------------------------------
_BRAND_CSS = """
<style>
/* Hide Streamlit sidebar & hamburger entirely */
section[data-testid="stSidebar"], div[data-testid="stToolbar"] { display: none !important; }
button[kind="header"] { display: none !important; }

/* Page padding a touch wider now that the left pane is gone */
.block-container { padding-top: 1.5rem; max-width: 1280px; }

/* Brand */
.halo-badge{
  display:inline-block;background:#FF7A00;color:#fff;
  font-weight:800;border-radius:8px;padding:.25rem .6rem;margin-right:.5rem;
  font-family: ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto;
}
.halo-word{color:#0E3B82;font-weight:800;}
.halo-chat{color:#4B5563;font-weight:600;}
h1,h2{color:#0E3B82;}
h3,h4,h5,h6{color:#1F2937;}
/* Chip buttons (suggested questions) */
button.suggest-chip {
  border-radius: 9999px !important;
  border: 1px solid #E5E7EB !important;
  background: #fff !important;
  color: #111827 !important;
  padding: .35rem .8rem !important;
  margin-right:.5rem !important;
}
</style>
"""

def _brand_header() -> None:
    st.markdown(_BRAND_CSS, unsafe_allow_html=True)
    st.markdown(
        '<div>'
        '<span class="halo-badge">HALO</span>'
        '<span class="halo-word" style="font-size:2rem">Quality</span>'
        '<span class="halo-chat" style="font-size:2rem"> — Chat</span>'
        '</div>',
        unsafe_allow_html=True,
    )

# --------------------------------------------------------------------------------------
# Question runner
# --------------------------------------------------------------------------------------
def _run_question(slug: str, params: Dict, q: str) -> Optional[Tuple[str, Optional["pd.DataFrame"]]]:
    """
    Dispatch into questions/<slug>.py and call run(store, params, q).
    Every question module must expose run(store, params, q).
    """
    try:
        mod = importlib.import_module(f"questions.{slug}")
    except Exception as e:
        st.error(f"Could not import question module '{slug}'.\n{e}")
        return None

    try:
        return mod.run({"root": ROOT, "data_dir": DATA_DIR}, params, q)
    except TypeError as e:
        # Helpful diagnostic when the run() signature is wrong
        st.error(
            "This question failed.\n\n"
            "TypeError: " + str(e) + "\n\n"
            "Expected signature: run(store: dict, params: dict, q: str)"
        )
        return None
    except Exception as e:
        st.error(f"This question failed.\n\n{e}")
        return None

# --------------------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------------------
def main():
    _brand_header()

    # Suggested question chips
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("complaint analysis — June 2025 (by portfolio)", key="chip1", help="Open complaints slide", type="secondary"):
            st.session_state["q"] = "complaint analysis — June 2025 (by portfolio)"
    with col2:
        if st.button("first pass accuracy analysis", key="chip2", help="Open FPA analysis", type="secondary"):
            st.session_state["q"] = "first pass accuracy analysis"

    # Free-form input (pre-fill last used)
    q_default = st.session_state.get("q", "complaint analysis — June 2025 (by portfolio)")
    q = st.text_input(
        "Type your question (e.g., 'complaint analysis — June 2025 by portfolio' or 'first pass accuracy analysis')",
        value=q_default,
        placeholder="Type here…",
    )
    st.session_state["q"] = q

    # Route to a question
    match = sem_router.match(q)
    if not match:
        st.info("I couldn't recognise that question. Try one of the chips above.")
        return

    slug = match["slug"]
    params = match.get("params", {})

    _run_question(slug, params, q)


if __name__ == "__main__":
    main()
