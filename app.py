# app.py
import sys
from pathlib import Path
import importlib
import importlib.util
from types import ModuleType
from typing import Callable, Dict, Optional, Tuple

import streamlit as st
import pandas as pd

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
CANDIDATE_DIRS = [
    HERE,
    HERE / "questions",
    HERE / "core",
    HERE / "core" / "questions",
    HERE.parent,
    HERE.parent / "questions",
    HERE.parent / "core",
    HERE.parent / "core" / "questions",
]

for p in CANDIDATE_DIRS:
    sys.path.insert(0, str(p))

# -----------------------------------------------------------------------------
# Utilities: robust import (module-or-file), then get a function (default: run)
# -----------------------------------------------------------------------------
def _try_import_module(modname: str) -> Tuple[Optional[ModuleType], Optional[Exception]]:
    try:
        return importlib.import_module(modname), None
    except Exception as e:
        return None, e

def _try_import_file(filepath: Path, modname_hint: str) -> Tuple[Optional[ModuleType], Optional[Exception]]:
    try:
        if not filepath.exists():
            return None, FileNotFoundError(str(filepath))
        spec = importlib.util.spec_from_file_location(modname_hint, str(filepath))
        module = importlib.util.module_from_spec(spec)
        assert spec and spec.loader  # type: ignore
        spec.loader.exec_module(module)  # type: ignore
        sys.modules[modname_hint] = module
        return module, None
    except Exception as e:
        return None, e

def load_callable(
    base: str,
    func: str = "run",
) -> Tuple[Optional[Callable], Dict[str, Exception], Optional[str]]:
    """
    Try many module names & file locations, then return callable.
    Returns (callable, errors_by_candidate, where_loaded)
    """
    errors: Dict[str, Exception] = {}
    loaded_from: Optional[str] = None

    module_candidates = [
        base,
        f"questions.{base}",
        f"core.{base}",
        f"core.questions.{base}",
        f"haloquality.{base}",
        f"haloquality.questions.{base}",
    ]

    for cand in module_candidates:
        m, err = _try_import_module(cand)
        if m:
            fn = getattr(m, func, None)
            if callable(fn):
                return fn, errors, f"module:{cand}"
            errors[cand] = AttributeError(f"Function '{func}' not found in {cand}")
        else:
            errors[cand] = err  # type: ignore

    # Try direct file paths if module import failed
    file_candidates = []
    for d in CANDIDATE_DIRS:
        file_candidates.append(d / f"{base}.py")
    # Deduplicate while preserving order
    seen = set()
    unique_files = []
    for f in file_candidates:
        if f not in seen:
            unique_files.append(f)
            seen.add(f)

    for f in unique_files:
        m, err = _try_import_file(f, f"__dyn__{base}__{abs(hash(str(f)))}")
        if m:
            fn = getattr(m, func, None)
            if callable(fn):
                return fn, errors, f"file:{f}"
            errors[str(f)] = AttributeError(f"Function '{func}' not found in file {f}")
        else:
            errors[str(f)] = err  # type: ignore

    return None, errors, None

# -----------------------------------------------------------------------------
# Import data_store & semantic_router with strong fallbacks
# -----------------------------------------------------------------------------
def load_load_store():
    fn, errs, where = load_callable("data_store", func="load_store")
    if fn:
        return fn, errs, where
    # additional package paths
    for extra in ("core.data_store", "haloquality.core.data_store"):
        m, err = _try_import_module(extra)
        if m:
            fn2 = getattr(m, "load_store", None)
            if callable(fn2):
                return fn2, errs, f"module:{extra}"
    return None, errs, None

def load_router():
    # expect match_query, IntentMatch in a module named semantic_router
    try_candidates = [
        "semantic_router",
        "core.semantic_router",
        "haloquality.core.semantic_router",
        "questions.semantic_router",  # just in case
    ]
    errs: Dict[str, Exception] = {}
    for cand in try_candidates:
        m, err = _try_import_module(cand)
        if m:
            mq = getattr(m, "match_query", None)
            IM = getattr(m, "IntentMatch", None)
            if callable(mq) and IM is not None:
                return mq, IM, errs, f"module:{cand}"
            errs[cand] = AttributeError(f"'match_query'/'IntentMatch' missing in {cand}")
        else:
            errs[cand] = err  # type: ignore

    # try file path loads too
    m, err = _try_import_file(HERE / "semantic_router.py", "__dyn__semantic_router")
    if m:
        mq = getattr(m, "match_query", None)
        IM = getattr(m, "IntentMatch", None)
        if callable(mq) and IM is not None:
            return mq, IM, errs, f"file:{HERE/'semantic_router.py'}"
        errs["semantic_router.py"] = AttributeError("'match_query'/'IntentMatch' missing in file")
    return None, None, errs, None

load_store, ds_errs, ds_where = load_load_store()
match_query, IntentMatch, router_errs, router_where = load_router()

# Question handlers
complaints_run, c_errs, c_where = load_callable("complaints_per_thousand", "run")
rca_run, r_errs, r_where = load_callable("rca1_portfolio_process", "run")
ucm_run, u_errs, u_where = load_callable("unique_cases_mom", "run")

# -----------------------------------------------------------------------------
# Streamlit page
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Halo Quality — Chat", layout="wide")

def _import_gate():
    ok = True
    if not load_store:
        ok = False
        st.error("Could not import `data_store.load_store`.")
        with st.expander("data_store import attempts & errors"):
            for k, v in ds_errs.items():
                st.write(k)
                st.exception(v)
    if not match_query:
        ok = False
        st.error("Could not import `semantic_router`.")
        with st.expander("semantic_router import attempts & errors"):
            for k, v in router_errs.items():
                st.write(k)
                st.exception(v)
    if not complaints_run or not rca_run or not ucm_run:
        ok = False
        st.error("One or more question modules failed to import.")
        with st.expander("question import attempts & errors"):
            st.markdown("**complaints_per_thousand**")
            for k, v in c_errs.items():
                st.write(k)
                st.exception(v)
            st.markdown("---\n**rca1_portfolio_process**")
            for k, v in r_errs.items():
                st.write(k)
                st.exception(v)
            st.markdown("---\n**unique_cases_mom**")
            for k, v in u_errs.items():
                st.write(k)
                st.exception(v)
    return ok

@st.cache_data(show_spinner=False)
def _load_store_cached():
    return load_store()

def _data_status(store):
    st.sidebar.subheader("Data status")
    st.sidebar.write(f"Cases rows: **{store.get('cases_rows', 0)}**")
    st.sidebar.write(f"Complaints rows: **{store.get('complaints_rows', 0)}**")

def _parsed_filters_box(title: str, params: dict):
    with st.expander(title, expanded=False):
        if not params:
            st.caption("No filters parsed.")
        else:
            df = pd.DataFrame([params])
            st.dataframe(df, hide_index=True, use_container_width=True)

def _run_question(slug: str, params: dict, store: dict, user_text: str):
    if slug == "complaints_per_thousand":
        runner = complaints_run
        st.subheader("Complaints per 1,000 cases")
    elif slug == "rca1_portfolio_process":
        runner = rca_run
        st.subheader("RCA1 by Portfolio × Process — last 3 months")
    elif slug == "unique_cases_mom":
        runner = ucm_run
        st.subheader("Unique cases (MoM)")
    else:
        st.warning("Sorry—couldn't understand that question.")
        return

    if runner is None:
        st.error("That question module failed to import.")
        return

    try:
        df = runner(store, params=params, user_text=user_text)
        if isinstance(df, pd.DataFrame):
            st.dataframe(df, use_container_width=True)
        else:
            st.write(df)
    except Exception as e:
        st.error("This question failed.")
        st.exception(e)

def main():
    st.title("Halo Quality — Chat")
    st.caption("Hi! Ask me about cases, complaints (incl. RCA), or first-pass accuracy.")

    if not _import_gate():
        st.stop()

    with st.status("Reading Excel / parquet sources", expanded=True):
        store = _load_store_cached()
    _data_status(store)

    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        if st.button(
            "complaints per 1000 by process for portfolio London Jun 2025 to Aug 2025",
            use_container_width=True,
        ):
            st.session_state["free_text"] = (
                "complaints per 1000 by process for portfolio london jun 2025 to aug 2025"
            )
    with c2:
        if st.button(
            "show rca1 by portfolio for process Member Enquiry last 3 months",
            use_container_width=True,
        ):
            st.session_state["free_text"] = (
                "show rca1 by portfolio for process member enquiry last 3 months"
            )
    with c3:
        if st.button(
            "unique cases by process and portfolio Apr 2025 to Jun 2025",
            use_container_width=True,
        ):
            st.session_state["free_text"] = (
                "unique cases by process and portfolio apr 2025 to jun 2025"
            )

    q = st.text_input(
        "Type your question (e.g., 'complaints per 1000 by process last 3 months')",
        value=st.session_state.get("free_text", ""),
    )

    try:
        match = match_query(q)
    except Exception as e:
        st.error("Query routing failed.")
        st.exception(e)
        st.stop()

    slug = getattr(match, "slug", None)
    params = getattr(match, "params", {}) or {}

    _parsed_filters_box("Parsed filters", params)
    _run_question(slug, params, store, q)

if __name__ == "__main__":
    main()
