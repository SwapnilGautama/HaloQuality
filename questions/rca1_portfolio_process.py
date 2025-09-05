# questions/rca1_portfolio_process.py
from __future__ import annotations
import pandas as pd
import streamlit as st
from core.rca_labeller import label_complaints_rca

TITLE = "RCA1 by Portfolio × Process"

def _soft_contains(s: pd.Series, needle: str) -> pd.Series:
    return s.fillna("").str.contains((needle or "").strip().lower(), case=False, na=False)

def _alias_cols(df: pd.DataFrame) -> pd.DataFrame:
    """
    Be very forgiving with column shapes across complaint extracts.
    Builds/aliases:
      - Portfolio_std  (from Portfolio / Site / Location)
      - Parent_Case_Type_std (from Parent Case Type / Process Name)
      - ProcessName_std (alias of Parent_Case_Type_std for convenience)
      - Process Name   (raw alias so any downstream code won't KeyError)
    """
    def first_present(cands):
        for c in cands:
            if c in df.columns:
                return c
        return None

    # Portfolio_std
    if "Portfolio_std" not in df.columns:
        pcol = first_present(["Portfolio", "portfolio", "Site", "Location"])
        df["Portfolio_std"] = df[pcol].astype(str).str.strip().str.lower() if pcol else ""

    # Parent_Case_Type_std
    if "Parent_Case_Type_std" not in df.columns:
        pct = first_present([
            "Parent Case Type", "Parent_Case_Type", "Parent case type",
            "Process Name", "Process", "Process_Name", "Processname"
        ])
        df["Parent_Case_Type_std"] = df[pct].astype(str).str.strip().str.lower() if pct else ""

    # Friendly aliases for other code paths that might expect these
    df["ProcessName_std"] = df.get("ProcessName_std", df["Parent_Case_Type_std"])

    if "Process Name" not in df.columns:
        df["Process Name"] = df.get("Parent Case Type", df.get("Parent_Case_Type", df["Parent_Case_Type_std"]))

    return df

# IMPORTANT: make user_text optional so the module works with (store, params) and (store, params, user_text)
def run(store, params, user_text: str | None = None):
    st.subheader(TITLE)

    comp = store.get("complaints", pd.DataFrame()).copy()
    if comp.empty:
        st.info("No complaints data available.")
        return

    # Robust column aliases so we never KeyError on headers
    comp = _alias_cols(comp)

    # Ensure RCA labels exist
    if "RCA1" not in comp.columns:
        comp = label_complaints_rca(comp)

    # Time window
    start_dt = params.get("start_dt")
    end_dt = params.get("end_dt")
    if start_dt is not None:
        comp = comp[comp["month_dt"] >= start_dt]
    if end_dt is not None:
        comp = comp[comp["month_dt"] <= end_dt]

    # Portfolio filter (soft)
    portfolio = params.get("portfolio")
    if portfolio:
        comp = comp[_soft_contains(comp["Portfolio_std"], portfolio)]

    # Process filter (soft) – use Parent_Case_Type_std
    process = params.get("process")
    if process:
        comp = comp[_soft_contains(comp["Parent_Case_Type_std"], process)]

    st.caption(f"Rows after filters: {len(comp):,}")

    if comp.empty:
        st.info("No rows after applying filters.")
        return

    g = (comp
         .dropna(subset=["RCA1"])
         .groupby(["Portfolio_std", "Parent_Case_Type_std", "RCA1"], dropna=False)
         .size()
         .reset_index(name="Count"))

    if g.empty:
        st.info("No RCA labels found after filtering.")
        return

    g = g.sort_values("Count", ascending=False)
    g.rename(columns={
        "Portfolio_std": "Portfolio",
        "Parent_Case_Type_std": "Process (Parent Case Type)"
    }, inplace=True)

    st.dataframe(g, use_container_width=True)
