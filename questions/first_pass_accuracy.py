# questions/fail_reasons_analysis.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple, Optional, List
import os
import re
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# ---- Brand style (match FPA visuals) ----
_DARK_BLUE = "#0b3d91"
_DARK_GREY = "#333333"
_SOFT_GREY = "#DDDDDD"

# =====================================================================
# Utilities (mirrored from first_pass_accuracy for consistent behaviour)
# =====================================================================

def _find_fpa_workbook() -> Optional[Path]:
    """
    Look in the standard locations and pick the newest matching file:
    data/first_pass_accuracy/FirstPassAccuracy*.xlsx (or .xls)
    """
    roots = [Path("data/first_pass_accuracy"), Path("first_pass_accuracy"), Path("data/first_pass_accuracy/")]
    patterns = ["FirstPassAccuracy*.xls*", "*FirstPassAccuracy*.xls*"]
    for root in roots:
        if not root.exists():
            continue
        hits = []
        for pat in patterns:
            hits.extend(sorted(root.glob(pat)))
        if hits:
            return hits[-1]
    return None


def _read_excel_any(path: Path) -> pd.DataFrame:
    try:
        return pd.read_excel(path)
    except Exception:
        return pd.read_excel(path, header=0)


def _pick(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    cols = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in cols:
            return cols[c.lower()]
    return None


def _coerce_month(s: pd.Series) -> pd.Series:
    dt = pd.to_datetime(s, errors="coerce", dayfirst=True)
    return dt.dt.to_period("M")


def _is_pass(x: str) -> bool:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return False
    t = str(x).strip().lower()
    return t.startswith("pass")


# =====================================================================
# Load FPA rows and prepare latest-month fail comments
# =====================================================================

def _load_fpa() -> Tuple[pd.DataFrame, Dict[str, str]]:
    p = _find_fpa_workbook()
    if not p:
        raise FileNotFoundError("Could not find a FirstPassAccuracy workbook (FirstPassAccuracy*.xlsx).")
    df = _read_excel_any(p)

    col_map = {
        "date": _pick(df, ["Activity Date", "ActivityDate", "Date", "Activity date"]),
        "result": _pick(df, ["Review Result", "Review result", "Result"]),
        "portfolio": _pick(df, ["Portfolio", "portfolio"]),
        "scheme": _pick(df, ["Scheme", "Scheme Name", "Plan", "Plan Name"]),
        "comment": _pick(df, ["Case Comment", "Comments", "Reviewer Comment", "Comment"]),
        "rca2": _pick(df, ["RCA2", "Root Cause 2", "RCA 2"]),
    }
    missing = [k for k, v in col_map.items() if k in ("date", "result") and v is None]
    if missing:
        raise KeyError(f"Missing required columns for FPA: {missing}")

    df = df.rename(columns={v: k for k, v in col_map.items() if v})
    df["date"] = pd.to_datetime(df["date"], errors="coerce", dayfirst=True)
    df["result"] = df["result"].astype(str)
    return df, col_map


# =====================================================================
# Label using shared core.reason_labeller with safe fallbacks
# =====================================================================

def _label_with_shared_labeller(texts: List[str], rca2_vals: List[Optional[str]], df_all: pd.DataFrame) -> List[str]:
    """
    Prefer the shared core.reason_labeller; if anything fails, return simple "Other".
    """
    try:
        from core.reason_labeller import (
            label_dataframe, get_or_fit_model
        )
        # try load or (small) fit an ML bundle (best-effort)
        bundle = get_or_fit_model(df_all, text_col="Case Comment", rca2_col="RCA2")
        tmp = pd.DataFrame({"Case Comment": texts, "RCA2": rca2_vals})
        s = label_dataframe(tmp, text_col="Case Comment", rca2_col="RCA2", model_bundle=bundle)
        return s.fillna("Other").astype(str).tolist()
    except Exception:
        # ultra-safe fallback
        return ["Other" for _ in texts]


# =====================================================================
# Plot helpers
# =====================================================================

def _fig_reasons_bar(df: pd.DataFrame, title: str):
    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    bars = ax.bar(df["reason"], df["count"])
    for b in bars:
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.5, f"{int(b.get_height())}",
                ha="center", va="bottom", fontsize=9, color=_DARK_GREY)
    ax.set_title(title, pad=8, color=_DARK_BLUE)
    for sp in ["left", "right", "top"]:
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color(_SOFT_GREY)
    ax.spines["bottom"].set_linewidth(1.25)
    ax.get_yaxis().set_visible(False)
    plt.setp(ax.get_xticklabels(), rotation=90, ha="center", color=_DARK_GREY)
    ax.grid(False)
    return fig


# =====================================================================
# Streamlit entry
# =====================================================================

def run(store: Dict, params: Dict, user_text: str = "") -> Tuple[str, pd.DataFrame]:
    """
    Standalone "Fail reasons analysis" question.
    - Loads the FPA workbook
    - Picks the latest month
    - Labels fail comments via core.reason_labeller
    - Shows Pareto (top 80% + Other) and a details table
    """
    try:
        df_raw, _ = _load_fpa()
    except FileNotFoundError as e:
        st.error(str(e)); return ("", pd.DataFrame())
    except KeyError as e:
        st.error(f"FPA file found, but a required column is missing: {e}")
        return ("", pd.DataFrame())

    # Determine latest month and subset to fails for that month
    df_raw = df_raw.assign(_m=_coerce_month(df_raw["date"]))
    latest = df_raw["_m"].max()
    if pd.isna(latest):
        st.info("No First-Pass Accuracy rows with a valid date were found.")
        return ("", pd.DataFrame())

    fails = df_raw[(df_raw["_m"] == latest) & (~df_raw["result"].apply(_is_pass))]
    if fails.empty:
        st.info(f"No FAIL rows found for {pd.Period(latest).to_timestamp().strftime('%b-%y')}.")
        return ("", pd.DataFrame())

    # Collect texts + optional RCA2; label using shared labeller
    texts = fails["comment"].astype(str).fillna("").tolist() if "comment" in fails.columns else [""] * len(fails)
    rca2_vals = fails["rca2"].astype(str).fillna("").tolist() if "rca2" in fails.columns else [""] * len(fails)

    labels = _label_with_shared_labeller(texts, rca2_vals, df_raw)

    # Pareto: value counts → top 80% + collapse tail into Other
    s = pd.Series(labels).value_counts().rename_axis("reason").reset_index(name="count")
    s = s.sort_values("count", ascending=False).reset_index(drop=True)
    total = int(s["count"].sum()) or 1
    s["percent"] = (s["count"] * 100.0 / total)
    s["cum_percent"] = s["percent"].cumsum()

    head = s[s["cum_percent"] <= 80.0].copy()
    tail = s[s["cum_percent"] > 80.0].copy()

    # Keep "Other" out of the head if present
    if not head.empty and (head["reason"] == "Other").any():
        move = head[head["reason"] == "Other"]
        head = head[head["reason"] != "Other"]
        tail = pd.concat([tail, move], ignore_index=True)

    if not tail.empty:
        other_row = pd.DataFrame([{
            "reason": "Other",
            "count": int(tail["count"].sum()),
            "percent": float(tail["percent"].sum()),
            "cum_percent": 100.0
        }])
        head = pd.concat([head, other_row], ignore_index=True)

    head["percent"] = head["percent"].round(1)
    head["cum_percent"] = head["cum_percent"].round(1)

    # Render
    st.markdown(
        f"<h4 style='color:{_DARK_BLUE};margin:0 0 .5rem 0;'>"
        f"Fail reasons — Pareto (top 80% + Other) — {pd.Period(latest).to_timestamp().strftime('%b-%y')}"
        f"</h4>", unsafe_allow_html=True
    )
    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.pyplot(_fig_reasons_bar(head[["reason", "count"]], "Fail reasons — Pareto"))
    with c2:
        st.dataframe(head, use_container_width=True)

    # Return empty text and the details table for any downstream use
    return ("", head)
