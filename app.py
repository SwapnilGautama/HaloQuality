# app.py
from __future__ import annotations

# --- stdlib
import importlib
import importlib.util
import inspect
import sys
from pathlib import Path
import traceback

# --- third party
import streamlit as st
import pandas as pd

# ------------------------------------------------------------
# Resilient module import helper
# ------------------------------------------------------------
THIS_DIR = Path(__file__).parent


def _import_from_candidates(
    candidates: list[str],
    prefer_attr: str | None = None,
    file_glob: str | None = None,
):
    """
    Try to import a module by several dotted names, e.g. ['core.data_store', 'data_store'].
    If all fail, optionally search for a file via file_glob (e.g. 'data_store*.py') and load it.
    If prefer_attr is provided, return that attribute from the loaded module.
    """
    last_exc: BaseException | None = None

    # 1) try dotted imports
    for dotted in candidates:
        try:
            mod = importlib.import_module(dotted)
            return getattr(mod, prefer_attr) if prefer_attr else mod
        except Exception as e:
            last_exc = e

    # 2) try file search (covers cases like "data_store (1).py" in same tree)
    if file_glob:
        for p in list(THIS_DIR.rglob(file_glob)):
            try:
                spec = importlib.util.spec_from_file_location(p.stem, p)
                if spec and spec.loader:
                    mod = importlib.util.module_from_spec(spec)
                    sys.modules[p.stem] = mod
                    spec.loader.exec_module(mod)
                    return getattr(mod, prefer_attr) if prefer_attr else mod
            except Exception as e:
                last_exc = e

    # nothing worked
    if last_exc:
        raise last_exc
    raise ModuleNotFoundError(
        f"Could not import any of {candidates} and no file matched {file_glob!r}"
    )


# ------------------------------------------------------------
# Imports that must work in either layout
#   - data_store:     core/data_store.py  (or data_store.py at repo root)
#   - semantic_router: semantic_router.py at repo root
# ------------------------------------------------------------
load_store = _import_from_candidates(
    ["core.data_store", "data_store"], prefer_attr="load_store", file_glob="data_store*.py"
)
match_query = _import_from_candidates(
    ["semantic_router"], prefer_attr="match_query", file_glob="semantic_router*.py"
)
IntentMatch = _import_from_candidates(
    ["semantic_router"], prefer_attr="IntentMatch", file_glob="semantic_router*.py"
)

# ------------------------------------------------------------
# Streamlit page config
# ------------------------------------------------------------
st.set_page_config(
    page_title="Halo Quality — Chat",
    layout="wide",
    page_icon=":bar_chart:",
)


# ------------------------------------------------------------
# Cached store loader
# ------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def _get_store():
    # Your load_store() should read Excel/Parquet and return a dict of DataFrames + counts
    # e.g.: {"cases": df, "complaints": df, "fpa": df, "cases_rows": int, "complaints_rows": int, "fpa_rows": int}
    return load_store()


# ------------------------------------------------------------
# Utility: load a question module by slug
# ------------------------------------------------------------
def _module_for_slug(slug: str):
    """
    Returns a loaded module for the given intent slug.

    Expected files live under /questions, e.g.
      questions/complaints_per_thousand.py (slug: 'complaints_per_thousand')
      questions/rca1_portfolio_process.py  (slug: 'rca1_portfolio_process')
      questions/unique_cases_mom.py        (slug: 'unique_cases_mom')
    """
    # direct mapping for known slugs; fall back to 'questions.<slug>'
    mapping = {
        "complaints_per_thousand": "questions.complaints_per_thousand",
        "rca1_portfolio_process": "questions.rca1_portfolio_process",
        "unique_cases_mom": "questions.unique_cases_mom",
        # allow a friendly alias some folks type
        "complaints_dashboard": "questions.complaints_per_thousand",
    }
    candidates = []
    if slug in mapping:
        candidates.append(mapping[slug])
    # common fallbacks
    candidates += [f"questions.{slug}", slug, f"core.{slug}"]

    return _import_from_candidates(candidates, prefer_attr=None, file_glob=f"{slug}*.py")


# ------------------------------------------------------------
# Utility: call module.run() regardless of its signature
#   - supports: run(store, params, user_text)
#   - or:       run(store, **params)
#   - includes user_text only if accepted
# ------------------------------------------------------------
def _safe_run(mod, store: dict, params: dict, user_text: str):
    if not hasattr(mod, "run"):
        raise AttributeError(f"Module {mod.__name__} has no 'run' function.")
    sig = inspect.signature(mod.run)
    kwargs = {}
    if "store" in sig.parameters:
        kwargs["store"] = store

    # Prefer a single 'params' dict if the function accepts it, else expand.
    if "params" in sig.parameters:
        kwargs["params"] = params or {}
    else:
        kwargs.update(params or {})

    if "user_text" in sig.parameters:
        kwargs["user_text"] = user_text

    return mod.run(**kwargs)


# ------------------------------------------------------------
# UI helpers
# ------------------------------------------------------------
def _render_result(result, title: str | None = None):
    if title:
        st.subheader(title)

    if result is None:
        st.info("No output.")
        return

    # Flexible result contract:
    # - DataFrame directly
    # - dict with keys: dataframe/df, fig, message, table_title
    if isinstance(result, pd.DataFrame):
        if result.empty:
            st.info("No data for the current filters.")
        else:
            st.dataframe(result, use_container_width=True)
        return

    if isinstance(result, dict):
        if "message" in result and result["message"]:
            st.info(str(result["message"]))

        df = result.get("dataframe") or result.get("df")
        if isinstance(df, pd.DataFrame):
            if df.empty:
                st.info("No data for the current filters.")
            else:
                label = result.get("table_title")
                if label:
                    st.caption(label)
                st.dataframe(df, use_container_width=True)

        fig = result.get("fig")
        if fig is not None:
            st.pyplot(fig)
        return

    # Fallback: just show whatever we got
    st.write(result)


def _data_status(store: dict):
    with st.sidebar:
        st.markdown("### Data status")
        cases_rows = store.get("cases_rows", len(store.get("cases", [])))
        complaints_rows = store.get("complaints_rows", len(store.get("complaints", [])))
        fpa_rows = store.get("fpa_rows", len(store.get("fpa", [])))

        st.write(f"Cases rows: **{cases_rows}**")
        st.write(f"Complaints rows: **{complaints_rows}**")
        st.write(f"FPA rows: **{fpa_rows}**")


def _parsed_filters_box(label: str, params: dict | None):
    with st.expander("Parsed filters", expanded=False):
        if not params:
            st.write("—")
        else:
            flat = {k: (str(v) if not isinstance(v, (str, int, float)) else v) for k, v in params.items()}
            st.write(label)
            st.json(flat)


# ------------------------------------------------------------
# Main app
# ------------------------------------------------------------
def main():
    st.title("Halo Quality — Chat")
    st.caption("Hi! Ask me about cases, complaints (incl. RCA), or first-pass accuracy.")

    # Load store (cached)
    with st.spinner("Reading Excel / parquet sources"):
        store = _get_store()

    # Sidebar status
    _data_status(store)

    # Quick-pick buttons (these feed the free-text box)
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("complaints per 1000 by process for portfolio London Jun 2025 to Aug 2025"):
            st.session_state["free_text"] = "complaints per 1000 by process for portfolio London Jun 2025 to Aug 2025"
    with c2:
        if st.button("show rca1 by portfolio for process Member Enquiry last 3 months"):
            st.session_state["free_text"] = "show rca1 by portfolio for process Member Enquiry last 3 months"
    with c3:
        if st.button("unique cases by process and portfolio Apr 2025 to Jun 2025"):
            st.session_state["free_text"] = "unique cases by process and portfolio Apr 2025 to Jun 2025"

    # Free-text query
    query = st.text_input(
        "Type your question (e.g., 'complaints per 1000 by process last 3 months')",
        value=st.session_state.get("free_text", ""),
        key="query_box",
    ).strip()

    if not query:
        return

    # Route the query
    try:
        match: IntentMatch = match_query(query)
    except Exception as e:
        st.error("Sorry—couldn't understand that question.")
        with st.expander("Traceback"):
            st.exception(e)
        return

    # Draw section title from the slug (nice defaults)
    slug_titles = {
        "complaints_per_thousand": "Complaints per 1,000 cases",
        "rca1_portfolio_process": "RCA1 by Portfolio × Process — last 3 months",
        "unique_cases_mom": "Unique cases (MoM)",
        "complaints_dashboard": "Complaints dashboard",
    }
    section_title = slug_titles.get(match.slug, match.slug.replace("_", " ").title())

    # Show parsed filters
    _parsed_filters_box("Parsed filters", match.params or {})

    # Import and run the matched module
    try:
        mod = _module_for_slug(match.slug)
    except Exception as e:
        st.error("That question module failed to import.")
        with st.expander("Import error details"):
            st.code("\n".join(traceback.format_exception(e)))
        return

    try:
        with st.spinner(f"Running: {section_title}"):
            result = _safe_run(mod, store=store, params=(match.params or {}), user_text=query)
        _render_result(result, title=section_title)
    except Exception as e:
        st.error("This question failed.")
        with st.expander("Traceback"):
            st.code("\n".join(traceback.format_exception(e)))


if __name__ == "__main__":
    main()
