# app.py
from __future__ import annotations

import importlib
import importlib.util
import re
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------
# Resilient imports
# ---------------------------------------------------------------------

def _try_import(module_name: str):
    """Try a normal import; return module or None."""
    try:
        return importlib.import_module(module_name)
    except Exception:
        return None

def _try_import_from_file(candidate: Path, dotted_fallback: str = ""):
    """Import a module from an explicit file path."""
    try:
        if not candidate.exists():
            return None
        spec = importlib.util.spec_from_file_location(dotted_fallback or candidate.stem, candidate)
        if not spec or not spec.loader:
            return None
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # type: ignore[attr-defined]
        return mod
    except Exception:
        return None


# --- data_store: support both project-root and /core placement
DATA_STORE_MOD = _try_import("data_store")
if DATA_STORE_MOD is None:
    DATA_STORE_MOD = _try_import("core.data_store")
if DATA_STORE_MOD is None:
    # last resort: direct file lookups
    here = Path(__file__).parent
    DATA_STORE_MOD = _try_import_from_file(here / "data_store.py", "data_store") or \
                     _try_import_from_file(here / "core" / "data_store.py", "core.data_store")

if DATA_STORE_MOD is None or not hasattr(DATA_STORE_MOD, "load_store"):
    raise ModuleNotFoundError(
        "Could not import load_store from data_store or core.data_store. "
        "Ensure data loader exists at 'core/data_store.py' (preferred) or 'data_store.py'."
    )

load_store = getattr(DATA_STORE_MOD, "load_store")


# ---------------------------------------------------------------------
# Helpers: dates, parsing, matching
# ---------------------------------------------------------------------

MONTH_RE = r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+(\d{4})"
MONTH_LOOK = {
    "jan": 1, "january": 1,
    "feb": 2, "february": 2,
    "mar": 3, "march": 3,
    "apr": 4, "april": 4,
    "may": 5,
    "jun": 6, "june": 6,
    "jul": 7, "july": 7,
    "aug": 8, "august": 8,
    "sep": 9, "sept": 9, "september": 9,
    "oct": 10, "october": 10,
    "nov": 11, "november": 11,
    "dec": 12, "december": 12,
}

KNOWN_PORTFOLIOS = [
    # Add/adjust as needed for your environment
    "London", "Leatherhead", "Exeter", "Scotland", "Chichester", "Northwest", "BAES Leatherhead",
    "SUMU01", "JPCU01", "SCHRD04", "HARR99", "HARROds",  # safe to list more; matching is case-insensitive
]

def _safe_month(ts: pd.Timestamp) -> pd.Timestamp:
    """Normalize to first-of-month (naive)."""
    ts = pd.to_datetime(ts)
    return pd.Timestamp(year=ts.year, month=ts.month, day=1)

def _month_from_words(m: str, y: str) -> Optional[pd.Timestamp]:
    try:
        mnum = MONTH_LOOK.get(m.lower()[:3])
        if not mnum:
            return None
        return pd.Timestamp(year=int(y), month=int(mnum), day=1)
    except Exception:
        return None

def parse_months_from_text(
    text: str,
    default_start: pd.Timestamp,
    default_end: pd.Timestamp,
) -> Tuple[pd.Timestamp, pd.Timestamp]:
    """
    Parse 'Apr 2025 to Jun 2025', 'Jun 2025', or 'last 3 months' from text.
    Fallback to provided defaults.
    """
    text_l = text.lower().strip()

    # last N months
    lm = re.search(r"last\s+(\d+)\s+month", text_l)
    if lm:
        n = max(1, int(lm.group(1)))
        end_m = default_end
        start_m = _safe_month((end_m - pd.offsets.MonthBegin(n-1)))
        return start_m, end_m

    # Two explicit months
    m = re.findall(MONTH_RE, text, flags=re.IGNORECASE)
    if len(m) >= 2:
        s = _month_from_words(m[0][0], m[0][1])
        e = _month_from_words(m[1][0], m[1][1])
        if s is not None and e is not None:
            return s, e

    # One explicit month
    if len(m) == 1:
        s = _month_from_words(m[0][0], m[0][1])
        if s is not None:
            return s, s

    return default_start, default_end

def parse_portfolio_from_text(text: str, default: str = "London") -> str:
    t = text.lower()
    for p in KNOWN_PORTFOLIOS:
        if p.lower() in t:
            return p
    return default


# ---------------------------------------------------------------------
# Question matching & dynamic import from questions/
# ---------------------------------------------------------------------

def guess_slug_from_text(text: str) -> str:
    t = text.lower()
    if "rca1" in t or "first pass" in t or "fpa" in t:
        return "rca1_portfolio_process"
    if "unique case" in t or "mom" in t:
        return "unique_cases_mom"
    # default to complaints per 1,000
    return "complaints_per_thousand"

def import_question(slug: str):
    """
    Import question module from questions/<slug>.py robustly.
    """
    mod = _try_import(f"questions.{slug}")
    if mod:
        return mod
    here = Path(__file__).parent
    return _try_import_from_file(here / "questions" / f"{slug}.py", f"questions.{slug}")

def _call_question_run(mod, store: Dict[str, pd.DataFrame], params: Dict[str, Any], user_text: str):
    """
    Call the question's run() with flexible signatures without breaking older code.

    Tries these forms in order:
      run(store, params, user_text=...)
      run(store, params)
      run(store, **params)
      run(store)
    """
    if not hasattr(mod, "run"):
        raise AttributeError(f"Question module {mod.__name__} has no 'run' function")

    try:
        # newest contract
        return mod.run(store, params, user_text=user_text)
    except TypeError:
        pass

    try:
        # older contract
        return mod.run(store, params)
    except TypeError:
        pass

    try:
        # kwargs contract
        return mod.run(store, **params)
    except TypeError:
        pass

    # minimal
    return mod.run(store)


# ---------------------------------------------------------------------
# UI glue
# ---------------------------------------------------------------------

st.set_page_config(page_title="Halo Quality — Chat", layout="wide")

def _data_status(store: Dict[str, pd.DataFrame]):
    cases_rows = int(store.get("cases", pd.DataFrame()).shape[0]) if "cases" in store else 0
    comp_rows  = int(store.get("complaints", pd.DataFrame()).shape[0]) if "complaints" in store else 0
    fpa_rows   = int(store.get("fpa", pd.DataFrame()).shape[0]) if "fpa" in store else 0

    with st.sidebar:
        st.header("Data status")
        st.write(f"Cases rows: **{cases_rows}**")
        st.write(f"Complaints rows: **{comp_rows}**")
        if fpa_rows:
            st.write(f"FPA rows: **{fpa_rows}**")

def _chips():
    cols = st.columns([1,1,1])
    with cols[0]:
        if st.button("complaints per 1000 by process for portfolio\nLondon Jun 2025 to Aug 2025"):
            st.session_state.free_text = "complaints per 1000 by process for portfolio London Jun 2025 to Aug 2025"
    with cols[1]:
        if st.button("show rca1 by portfolio for process Member Enquiry\nlast 3 months"):
            st.session_state.free_text = "show rca1 by portfolio for process Member Enquiry last 3 months"
    with cols[2]:
        if st.button("unique cases by process and portfolio Apr 2025\nto Jun 2025"):
            st.session_state.free_text = "unique cases by process and portfolio Apr 2025 to Jun 2025"

def _free_text_box(placeholder: str):
    val = st.session_state.get("free_text", "")
    txt = st.text_input("Type your question (e.g., 'complaints per 1000 by process last 3 months')", value=val, placeholder=placeholder)
    st.session_state.free_text = txt
    return txt


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    st.title("Halo Quality — Chat")

    # Load store (robust call – supports different loader signatures)
    try:
        store = load_store()
    except TypeError:
        # some versions accept a flag or streamlit context; try best-effort
        try:
            store = load_store(read_excel=True)
        except Exception:
            store = load_store(st)

    _data_status(store)

    _chips()
    text = _free_text_box("complaints per 1000 by process last 3 months")

    # ---- Determine slug/question to run
    slug = guess_slug_from_text(text)
    mod = import_question(slug)

    if mod is None:
        st.error(f"Could not import question module: questions/{slug}.py")
        return

    # ---- Default months from the data if available
    # use 'cases' when possible for the natural right-edge month
    if "cases" in store and not store["cases"].empty and "date" in store["cases"].columns:
        end_default = _safe_month(pd.to_datetime(store["cases"]["date"]).max())
    else:
        end_default = _safe_month(pd.Timestamp.today())
    start_default = _safe_month(end_default - pd.offsets.MonthBegin(2))  # ~ last 3 months by default

    # ---- Build params safely (this is what was crashing before)
    # We do NOT assume a semantic-router 'match' object exists;
    # everything is derived from the user's text with sensible defaults.
    portfolio = parse_portfolio_from_text(text, default="London")
    start_m, end_m = parse_months_from_text(text, start_default, end_default)

    params: Dict[str, Any] = {
        "portfolio": portfolio,
        "start_month": str(start_m.date()),
        "end_month": str(end_m.date()),
    }

    with st.expander("Parsed filters", expanded=False):
        st.write(f"start_month: {params['start_month']} | end_month: {params['end_month']}")
        st.write(f"portfolio: {params['portfolio']}")

    # ---- Section header per question
    pretty_titles = {
        "complaints_per_thousand": "Complaints per 1,000 cases",
        "rca1_portfolio_process": "RCA1 by Portfolio × Process — last 3 months",
        "unique_cases_mom": "Unique cases (MoM)",
    }
    st.header(pretty_titles.get(slug, "Results"))

    # ---- Run the question and render whatever it returns
    try:
        result = _call_question_run(mod, store, params, user_text=text)

        # Allow questions to fully control rendering; but if they return something, display it politely
        if result is None:
            return

        if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], pd.DataFrame):
            title, frame = result
            st.subheader(str(title))
            st.dataframe(frame, use_container_width=True)
        elif isinstance(result, pd.DataFrame):
            st.dataframe(result, use_container_width=True)
        elif isinstance(result, (list, tuple)):
            for item in result:
                if isinstance(item, pd.DataFrame):
                    st.dataframe(item, use_container_width=True)
                else:
                    st.write(item)
        else:
            st.write(result)

    except Exception as e:
        st.error("This question failed.")
        with st.expander("Traceback", expanded=False):
            st.exception(e)


if __name__ == "__main__":
    main()
