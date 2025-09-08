# questions/first_pass_accuracy.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple, Optional, List
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
# Data loading
# ======================
def _find_fpa_workbook() -> Optional[Path]:
    roots = [Path("data/first_pass_accuracy"), Path("first_pass_accuracy"), Path("data/first_pass_accuracy/")]
    patterns = ["FirstPassAccuracy*.xls*", "*FirstPassAccuracy*.xls*"]
    for root in roots:
        if not root.exists():
            continue
        for pat in patterns:
            hits = sorted(root.glob(pat))
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
    return df, col_map

# ======================
# Pass% and tables
# ======================
def _is_pass(x: str) -> bool:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return False
    t = str(x).strip().lower()
    return t.startswith("pass")

def _series_mom(df: pd.DataFrame) -> pd.DataFrame:
    s = _coerce_month(df["date"])
    df = df.assign(_m=s)
    if df["_m"].dropna().empty:
        return pd.DataFrame(columns=["month", "pass_pct"])
    start = pd.Period("2025-01")
    end = df["_m"].max()
    months = pd.period_range(start, end, freq="M")
    g = df.groupby("_m")["result"].agg(
        total="count", passed=lambda x: np.sum([_is_pass(v) for v in x])
    ).reindex(months, fill_value=0)
    pct = (g["passed"] * 100.0 / g["total"].replace(0, np.nan)).fillna(0.0).round(0)
    label = [pd.Period(m).to_timestamp().strftime("%b-%y") for m in months]
    return pd.DataFrame({"month": label, "pass_pct": pct.values})

def _table_portfolio_mom(df: pd.DataFrame) -> pd.DataFrame:
    """
    Portfolio (rows) × Month (columns) FPA% from Jan-25 to latest.
    """
    df = df.copy()
    df["_m"] = _coerce_month(df["date"])
    if df["_m"].dropna().empty:
        return pd.DataFrame()

    start = pd.Period("2025-01")
    end = df["_m"].max()
    months = pd.period_range(start, end, freq="M")

    # Compute pass% per portfolio per month
    grp = df.groupby(["portfolio", "_m"])["result"].agg(
        total="count", passed=lambda x: np.sum([_is_pass(v) for v in x])
    ).reset_index()
    grp["pass_%"] = (grp["passed"] * 100.0 / grp["total"]).round(0)

    piv = grp.pivot(index="portfolio", columns="_m", values="pass_%").reindex(columns=months)
    # Pretty month headers like Jan-25
    piv.columns = [pd.Period(m).to_timestamp().strftime("%b-%y") for m in piv.columns]
    piv = piv.sort_index().fillna(0).astype(int)
    return piv

# ======================
# Reasons — ALL + Pareto line to 100%
# ======================
def _label_all_latest(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Period]:
    from core.reason_labeller import label_dataframe

    df = df.assign(_m=_coerce_month(df["date"]))
    latest = df["_m"].max()
    if pd.isna(latest):
        return pd.DataFrame(), latest

    fails = df[(df["_m"] == latest) & (~df["result"].apply(_is_pass))].copy()
    if fails.empty:
        return pd.DataFrame(columns=["reason", "count", "percent", "cum_percent"]), latest

    lab_df = pd.DataFrame({
        "Case Comment": fails["comment"].fillna("").astype(str),
        "RCA2": (fails["rca2"].fillna("").astype(str) if "rca2" in fails.columns else "")
    })
    labels = label_dataframe(lab_df, text_col="Case Comment", rca2_col="RCA2")\
        .fillna("Other").astype(str)

    vc = labels.value_counts().rename_axis("reason").reset_index(name="count")
    vc = vc.sort_values("count", ascending=False).reset_index(drop=True)

    total = int(vc["count"].sum()) or 1
    vc["percent"] = (vc["count"] * 100.0 / total)
    vc["cum_percent"] = vc["percent"].cumsum().clip(upper=100.0)
    vc["percent"] = vc["percent"].round(1)
    vc["cum_percent"] = vc["cum_percent"].round(1)
    return vc, latest

# ======================
# Plots (styling)
# ======================
def _fig_mom(df: pd.DataFrame, title: str):
    """
    Smooth pastel line, no markers, soft baseline, no y-axis.
    """
    fig, ax = plt.subplots(figsize=(7.2, 3.2))

    ax.plot(df["month"], df["pass_pct"], linewidth=3.2, color=_PASTEL_LINE)

    # light labels above points
    for x, y in zip(df["month"], df["pass_pct"]):
        ax.text(x, y + 2, f"{y:.0f}%", ha="center", va="bottom", fontsize=9, color=_DARK_GREY)

    # Style: remove borders & y-axis; soft bottom spine only
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
    """
    Bars = counts (sorted desc), smooth cumulative % line.
    RCA1-style:
      - Soft pastel bars, teal cumulative line
      - NO plot title here (section title is rendered above)
      - NO plot border (all spines off except bottom in soft grey)
      - NO primary/secondary y-axes
    """
    fig, ax1 = plt.subplots(figsize=(8.6, 4.0))

    x = np.arange(len(df))

    # Bars with RCA1-like pastel palette
    colors = [_RCA1_BARS[i % len(_RCA1_BARS)] for i in range(len(df))]
    bars = ax1.bar(x, df["count"], color=colors)

    # Count labels on bars
    lift = max(df["count"]) * 0.015 if len(df) else 1
    for b in bars:
        ax1.text(
            b.get_x() + b.get_width() / 2,
            b.get_height() + lift,
            f"{int(b.get_height())}",
            ha="center",
            va="bottom",
            fontsize=9,
            color=_DARK_GREY,
        )

    # Smooth cumulative line (interpolated for gentle curve)
    ax2 = ax1.twinx()
    x_dense = np.linspace(x.min(), x.max(), num=max(200, len(x) * 20))
    y_dense = np.interp(x_dense, x, df["cum_percent"].values)
    ax2.plot(x_dense, y_dense, linewidth=2.8, color=_RCA1_CUM_LINE)

    # Minimal labels on the line at bar positions
    for xi, cp in zip(x, df["cum_percent"]):
        ax2.text(xi, cp + 2, f"{cp:.0f}%", ha="center", va="bottom", fontsize=8, color=_DARK_GREY)

    # X ticks / labels
    ax1.set_xticks(x)
    ax1.set_xticklabels(df["reason"], rotation=90, ha="center", color=_DARK_GREY)

    # Style: remove plot borders; keep only a soft bottom baseline on the primary axis
    for sp in ["left", "right", "top"]:
        ax1.spines[sp].set_visible(False)
    ax1.spines["bottom"].set_color(_SOFT_GREY)
    ax1.spines["bottom"].set_linewidth(1.25)

    # Hide both y-axes completely and hide twin axis spines
    ax1.get_yaxis().set_visible(False)
    ax2.get_yaxis().set_visible(False)
    for sp in ["left", "right", "top", "bottom"]:
        ax2.spines[sp].set_visible(False)

    ax1.set_xlabel("")
    ax1.set_ylabel("")
    ax2.set_ylabel("")
    ax1.grid(False)
    ax2.set_ylim(0, 100)

    return fig

# ======================
# Streamlit entry
# ======================
def run(store: Dict, params: Dict, user_text: str = "") -> Tuple[str, pd.DataFrame]:
    # Row 1: FPA MoM + table
    try:
        df_raw, _ = _load_fpa()
    except FileNotFoundError as e:
        st.error(str(e)); return ("", pd.DataFrame())
    except KeyError as e:
        st.error(f"FPA file found, but a required column is missing: {e}")
        return ("", pd.DataFrame())

    mom = _series_mom(df_raw)
    if mom.empty:
        st.info("No First-Pass Accuracy rows found from Jan-25 onward.")
        return ("", pd.DataFrame())

    df_raw = df_raw.assign(_m=_coerce_month(pd.to_datetime(df_raw["date"], errors="coerce", dayfirst=True)))
    latest = df_raw["_m"].max()

    # NEW: Month-on-Month FPA% by portfolio (rows × columns)
    piv_portfolio_mom = _table_portfolio_mom(df_raw)

    c1, c2 = st.columns((1.1, 1.0), gap="large")
    with c1:
        st.pyplot(_fig_mom(mom, f"First-Pass Accuracy — Jan–{pd.Period(latest).to_timestamp().strftime('%b %y')}"))
    with c2:
        st.markdown(
            f"<h4 style='color:{_DARK_BLUE};margin:0 0 .5rem 0;'>"
            f"FPA % by Portfolio — Month on Month"
            f"</h4>", unsafe_allow_html=True)
        if not piv_portfolio_mom.empty:
            st.dataframe(piv_portfolio_mom, use_container_width=True)

    # Row 2: Reasons — ALL + Pareto line
    reasons, lastp = _label_all_latest(df_raw)
    st.markdown(
        f"<h4 style='color:{_DARK_BLUE};margin:1rem 0 .5rem 0;'>"
        f"Reasons for Fail — {pd.Period(lastp).to_timestamp().strftime('%b-%y')}"
        f"</h4>", unsafe_allow_html=True)

    r1, r2 = st.columns(2, gap="large")
    with r1:
        if not reasons.empty:
            st.pyplot(_fig_pareto_full(reasons))
        else:
            st.info("No fail reasons available for the latest month.")
    with r2:
        if not reasons.empty:
            st.dataframe(reasons, use_container_width=True)

    return ("", pd.DataFrame())
