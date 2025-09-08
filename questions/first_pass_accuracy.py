# questions/first_pass_accuracy.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple, Optional, List
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# ---------------------------
# Brand / palette
# ---------------------------
_DARK_BLUE = "#0b3d91"
_DARK_GREY = "#333333"
_SOFT_GREY = "#E0E0E0"   # softer axis baseline

# Pastel palette for the FPA line
_PASTEL_LINE = "#8ECAE6"

# RCA1-like pastel bar palette (soft blues/greens/greys/oranges)
_RCA1_BARS = [
    "#9ECAE1", "#A1D99B", "#BDBDBD", "#FDAE6B", "#C6DBEF", "#FDD0A2",
    "#D9F0A3", "#BCBDDC", "#C7E9C0", "#F2F0F7", "#E5F5E0", "#FEE6CE"
]
# Smooth cumulative line (soft green/teal)
_RCA1_CUM_LINE = "#74C69D"

# ======================
# Utility: file path + mtime (for stable cache keys)
# ======================
def _find_fpa_workbook() -> Optional[Path]:
    roots = [Path("data/first_pass_accuracy"), Path("first_pass_accuracy"), Path("data/first_pass_accuracy/")]
    patterns = ["FirstPassAccuracy*.xls*", "*FirstPassAccuracy*.xls*"]
    for root in roots:
        if not root.exists():
            continue
        hits = []
        for pat in patterns:
            hits.extend(root.glob(pat))
        if hits:
            return sorted(hits)[-1]
    return None

def _file_sig(p: Path) -> Tuple[str, float]:
    """Cache key: (absolute path, last_modified_time)."""
    return (str(p.resolve()), os.path.getmtime(p))

# ======================
# Data loading (cached)
# ======================
@st.cache_data(show_spinner=False)
def _read_excel_cached(path_str: str, mtime: float) -> pd.DataFrame:
    path = Path(path_str)
    # try regular header; fall back once if needed
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

def _coerce_month_fast(s: pd.Series) -> pd.Series:
    dt = pd.to_datetime(s, errors="coerce", dayfirst=True)
    return dt.dt.to_period("M")

def _load_fpa() -> Tuple[pd.DataFrame, Dict[str, str], Tuple[str, float]]:
    p = _find_fpa_workbook()
    if not p:
        raise FileNotFoundError("Could not find a FirstPassAccuracy workbook (FirstPassAccuracy*.xlsx).")
    sig = _file_sig(p)
    df = _read_excel_cached(sig[0], sig[1])
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
    # fast coercions
    df["date"] = pd.to_datetime(df["date"], errors="coerce", dayfirst=True)
    df["_m"] = df["date"].dt.to_period("M")
    # helpful dtypes
    if "portfolio" in df.columns:
        df["portfolio"] = df["portfolio"].astype("category")
    return df, col_map, sig

# ======================
# Pass% and tables (cached)
# ======================
@st.cache_data(show_spinner=False)
def _series_mom_cached(months: List[pd.Period], results: pd.Series) -> pd.DataFrame:
    # results is aligned to months via index on df["_m"]
    g = results.groupby(results.index).agg(
        total="count",
        passed=lambda x: np.sum([str(v).strip().lower().startswith("pass") for v in x]),
    ).reindex(months, fill_value=0)
    pct = (g["passed"] * 100.0 / g["total"].replace(0, np.nan)).fillna(0.0).round(0)
    label = [pd.Period(m).to_timestamp().strftime("%b-%y") for m in months]
    return pd.DataFrame({"month": label, "pass_pct": pct.values})

@st.cache_data(show_spinner=False)
def _table_portfolio_mom_cached(df_port_mo: pd.DataFrame, months: List[pd.Period]) -> pd.DataFrame:
    grp = df_port_mo.groupby(["portfolio", "_m"])["result"].agg(
        total="count", passed=lambda x: np.sum([str(v).strip().lower().startswith("pass") for v in x])
    ).reset_index()
    grp["pass_%"] = (grp["passed"] * 100.0 / grp["total"]).round(0)
    piv = grp.pivot(index="portfolio", columns="_m", values="pass_%").reindex(columns=months)
    piv.columns = [pd.Period(m).to_timestamp().strftime("%b-%y") for m in piv.columns]
    return piv.sort_index().fillna(0).astype(int)

# ======================
# Reason labelling (cached)
# ======================
@st.cache_data(show_spinner=False)
def _label_all_cached(sig: Tuple[str, float], df_fail_minimal: pd.DataFrame) -> pd.DataFrame:
    """
    Return FAIL rows with a 'reason' column.
    Cache key includes file signature so we don't relabel unless the file changes.
    """
    from core.reason_labeller import label_dataframe
    lab_df = pd.DataFrame({
        "Case Comment": df_fail_minimal["comment"].fillna("").astype(str),
        "RCA2": (df_fail_minimal["rca2"].fillna("").astype(str) if "rca2" in df_fail_minimal.columns else "")
    })
    reasons = label_dataframe(lab_df, text_col="Case Comment", rca2_col="RCA2")\
        .fillna("Other").astype(str).to_numpy()
    out = df_fail_minimal.copy()
    out["reason"] = pd.Categorical(reasons)  # categorical helps pivots
    return out

@st.cache_data(show_spinner=False)
def _reasons_latest_cached(fails_all: pd.DataFrame, latest: pd.Period) -> pd.DataFrame:
    sub = fails_all[fails_all["_m"] == latest]
    vc = sub["reason"].value_counts().rename_axis("reason").reset_index(name="count")
    if vc.empty:
        return vc.assign(percent=[], cum_percent=[])
    vc = vc.sort_values("count", ascending=False).reset_index(drop=True)
    total = int(vc["count"].sum()) or 1
    vc["percent"] = vc["count"] * 100.0 / total
    vc["cum_percent"] = vc["percent"].cumsum().clip(upper=100.0).round(1)
    return vc[["reason", "count", "cum_percent"]]

@st.cache_data(show_spinner=False)
def _pivot_fail_matrix_cached(fails_all: pd.DataFrame, start_period: pd.Period) -> pd.DataFrame:
    """(portfolio, reason) rows × months (>= start_period) columns → counts."""
    fails_yr = fails_all[fails_all["_m"] >= start_period].copy()
    if fails_yr.empty:
        return pd.DataFrame()
    months = pd.period_range(start_period, fails_yr["_m"].max(), freq="M")
    month_labels = [pd.Period(m).to_timestamp().strftime("%b-%y") for m in months]
    g = fails_yr.groupby(["portfolio", "reason", "_m"]).size().reset_index(name="count")
    mat = g.pivot_table(index=["portfolio", "reason"], columns="_m", values="count", fill_value=0)
    mat = mat.reindex(columns=months, fill_value=0)
    mat.columns = month_labels
    return mat.sort_index()

# ======================
# Plots (styling)
# ======================
def _fig_mom(df: pd.DataFrame, title: str):
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    ax.plot(df["month"], df["pass_pct"], linewidth=3.2, color=_PASTEL_LINE)
    for x, y in zip(df["month"], df["pass_pct"]):
        ax.text(x, y + 2, f"{y:.0f}%", ha="center", va="bottom", fontsize=9, color=_DARK_GREY)
    for sp in ["left", "right", "top"]:
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color(_SOFT_GREY)
    ax.spines["bottom"].set_linewidth(1.25)
    ax.get_yaxis().set_visible(False)
    ax.set_xlabel(""); ax.set_ylabel(""); ax.grid(False)
    ax.set_title(title, pad=8, color=_DARK_BLUE)
    ax.set_ylim(bottom=0, top=100)
    return fig

def _fig_pareto_full(df: pd.DataFrame):
    fig, ax1 = plt.subplots(figsize=(8.6, 4.0))
    x = np.arange(len(df))
    colors = [_RCA1_BARS[i % len(_RCA1_BARS)] for i in range(len(df))]
    bars = ax1.bar(x, df["count"], color=colors)

    if len(df):
        lift = max(df["count"]) * 0.015
        for b in bars:
            ax1.text(b.get_x() + b.get_width()/2, b.get_height() + lift,
                     f"{int(b.get_height())}", ha="center", va="bottom",
                     fontsize=9, color=_DARK_GREY)

    ax2 = ax1.twinx()
    # light curve (dense only if there are enough bars)
    if len(df) > 2:
        x_dense = np.linspace(x.min(), x.max(), num=min(400, max(60, len(x) * 20)))
        y_dense = np.interp(x_dense, x, df["cum_percent"].values)
        ax2.plot(x_dense, y_dense, linewidth=2.8, color=_RCA1_CUM_LINE)
    else:
        ax2.plot(x, df["cum_percent"], linewidth=2.8, color=_RCA1_CUM_LINE)

    for xi, cp in zip(x, df["cum_percent"]):
        ax2.text(xi, cp + 2, f"{cp:.0f}%", ha="center", va="bottom",
                 fontsize=8, color=_DARK_GREY)

    ax1.set_xticks(x)
    ax1.set_xticklabels(df["reason"], rotation=90, ha="center", color=_DARK_GREY)

    for sp in ["left", "right", "top"]:
        ax1.spines[sp].set_visible(False)
    ax1.spines["bottom"].set_color(_SOFT_GREY)
    ax1.spines["bottom"].set_linewidth(1.25)

    ax1.get_yaxis().set_visible(False)
    ax2.get_yaxis().set_visible(False)
    for sp in ["left", "right", "top", "bottom"]:
        ax2.spines[sp].set_visible(False)

    ax1.set_xlabel(""); ax1.set_ylabel(""); ax2.set_ylabel("")
    ax1.grid(False)
    ax2.set_ylim(0, 100)
    return fig

# ======================
# Streamlit entry
# ======================
def run(store: Dict, params: Dict, user_text: str = "") -> Tuple[str, pd.DataFrame]:
    # Load once (cached) -----------------------------------------
    try:
        df_raw, _colmap, sig = _load_fpa()
    except FileNotFoundError as e:
        st.error(str(e)); return ("", pd.DataFrame())
    except KeyError as e:
        st.error(f"FPA file found, but a required column is missing: {e}")
        return ("", pd.DataFrame())

    # Common month range from Jan-25 to latest (reused everywhere)
    if df_raw["_m"].dropna().empty:
        st.info("No First-Pass Accuracy rows found."); return ("", pd.DataFrame())
    start = pd.Period("2025-01")
    latest = df_raw["_m"].max()
    months = pd.period_range(start, latest, freq="M")

    # Row 1: FPA MoM + table (cached) ----------------------------
    mom = _series_mom_cached(months, df_raw.set_index("_m")["result"])
    df_port_mo = df_raw[["_m", "portfolio", "result"]].copy()
    piv_portfolio_mom = _table_portfolio_mom_cached(df_port_mo, months)

    # Sidebar filters (Portfolio + Fail reasons) -----------------
    # Build (cached) labelled FAILS once
    df_fails_source = df_raw.loc[~df_raw["result"].astype(str).str.lower().str.startswith("pass"),
                                 ["_m", "portfolio", "comment", "rca2"]].copy()
    fails_all = _label_all_cached(sig, df_fails_source)
    if "portfolio" in fails_all.columns:
        fails_all["portfolio"] = fails_all["portfolio"].astype("category")

    st.sidebar.header("Filters — Fail reasons")
    # limit to 2025 for the matrix, but expose all present in 2025 only
    fails_2025 = fails_all[fails_all["_m"] >= pd.Period("2025-01")]
    all_reasons = sorted(fails_2025["reason"].cat.categories.tolist() if hasattr(fails_2025["reason"], "cat") else fails_2025["reason"].unique().tolist())
    all_portfolios = sorted(fails_2025["portfolio"].dropna().unique().tolist())
    sel_reasons = st.sidebar.multiselect("Fail reasons", options=all_reasons, default=all_reasons)
    sel_portfolios = st.sidebar.multiselect("Portfolios", options=all_portfolios, default=all_portfolios)

    # Layout ------------------------------------------------------
    c1, c2 = st.columns((1.1, 1.0), gap="large")
    with c1:
        st.pyplot(_fig_mom(mom, f"First-Pass Accuracy — Jan–{pd.Period(latest).to_timestamp().strftime('%b %y')}"))
    with c2:
        st.markdown(
            f"<h4 style='color:{_DARK_BLUE};margin:0 0 .5rem 0;'>FPA % by Portfolio — Month on Month</h4>",
            unsafe_allow_html=True
        )
        if not piv_portfolio_mom.empty:
            st.dataframe(piv_portfolio_mom, use_container_width=True)

    # Row 2: Pareto (latest) + matrix (2025) ---------------------
    st.markdown(
        f"<h4 style='color:{_DARK_BLUE};margin:1rem 0 .5rem 0;'>Reasons for Fail — {pd.Period(latest).to_timestamp().strftime('%b-%y')}</h4>",
        unsafe_allow_html=True
    )

    reasons_latest = _reasons_latest_cached(fails_all, latest)
    matrix_2025 = _pivot_fail_matrix_cached(fails_all, pd.Period("2025-01"))

    r1, r2 = st.columns((1.0, 1.2), gap="large")
    with r1:
        if not reasons_latest.empty:
            st.pyplot(_fig_pareto_full(reasons_latest))
        else:
            st.info("No fail reasons available for the latest month.")
    with r2:
        if not matrix_2025.empty:
            if sel_reasons:
                matrix_2025 = matrix_2025.loc[matrix_2025.index.get_level_values("reason").isin(sel_reasons)]
            if sel_portfolios:
                matrix_2025 = matrix_2025.loc[matrix_2025.index.get_level_values("portfolio").isin(sel_portfolios)]
            st.markdown(
                f"<h4 style='color:{_DARK_BLUE};margin:0 0 .5rem 0;'>Fail Reasons × Portfolio — Month on Month (2025)</h4>",
                unsafe_allow_html=True
            )
            st.dataframe(matrix_2025, use_container_width=True)
        else:
            st.info("No 2025 fail reason data available to populate the matrix.")

    return ("", pd.DataFrame())
