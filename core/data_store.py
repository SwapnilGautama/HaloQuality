from __future__ import annotations

from pathlib import Path
import pandas as pd

# Streamlit cache when available (falls back to normal memoization if app imports this outside Streamlit)
try:
    import streamlit as st
    cache_fn = st.cache_data(show_spinner=False)
except Exception:  # pragma: no cover
    from functools import lru_cache as cache_fn  # type: ignore

# Local loaders (each returns a pandas DataFrame)
from .loader_cases import load_cases
from .loader_complaints import load_complaints
from .loader_fpa import load_fpa
from .join_cases_complaints import build_cases_complaints_join


DATA_DIR = Path("data")  # root data dir (as in your repo layout)


def _safe_len(df: pd.DataFrame | None) -> int:
    return 0 if df is None else int(len(df))


@cache_fn
def load_store(sig_cases: str = "", sig_complaints: str = "", sig_fpa: str = "") -> dict:
    """
    Central entrypoint the app calls.
    - Loads Cases, Complaints, and FPA (each can be multi-month and auto-merged).
    - Builds the joined view: complaints × cases (per your join rules).
    - Returns a dictionary with the raw and derived tables, and some quick stats.

    The signature strings (sig_*) only exist to give Streamlit cache a cheap invalidation
    key when you change data on disk; the function ignores their contents otherwise.
    """

    # ---- Load raw tables ----
    cases = load_cases(DATA_DIR / "cases")
    complaints = load_complaints(DATA_DIR / "complaints")
    fpa = load_fpa(DATA_DIR / "first_pass_accuracy")

    # ---- Build joined tables (complaints × cases) ----
    joined_summary, rca = build_cases_complaints_join(cases, complaints)

    # ---- Quick stats the UI shows in the left panel ----
    stats = {
        "cases_rows": _safe_len(cases),
        "complaints_rows": _safe_len(complaints),
        "fpa_rows": _safe_len(fpa),
        "latest_cases_month": str(cases["month"].max()) if "month" in cases.columns and len(cases) else "NaT",
        "latest_complaints_month": str(complaints["month"].max()) if "month" in complaints.columns and len(complaints) else "NaT",
        "latest_fpa_month": str(fpa["month"].max()) if "month" in fpa.columns and len(fpa) else "NaT",
    }

    # Package everything for the app
    return {
        "cases": cases,
        "complaints": complaints,
        "fpa": fpa,
        "joined_summary": joined_summary,
        "rca": rca,
        "stats": stats,
    }
