# questions/first_pass_accuracy.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple, Optional, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from openai import OpenAI

# ---------------------------
# Brand / palette
# ---------------------------
_DARK_BLUE = "#0b3d91"
_DARK_GREY = "#333333"
_SOFT_GREY = "#E0E0E0"

_PASTEL_LINE = "#8ECAE6"
_RCA1_BARS = [
    "#9ECAE1", "#A1D99B", "#BDBDBD", "#FDAE6B", "#C6DBEF", "#FDD0A2",
    "#D9F0A3", "#BCBDDC", "#C7E9C0", "#F2F0F7", "#E5F5E0", "#FEE6CE"
]
_RCA1_CUM_LINE = "#74C69D"

# soft pastel palette for multiple lines
_PASTEL_LINES = [
    "#8ECAE6", "#FFB5A7", "#B5E48C", "#FBC4AB", "#A0C4FF", "#FFD6A5",
    "#BDB2FF", "#FFADAD", "#CFF5E7", "#FFC8DD"
]

# ======================
# Data loading
# ======================
def _find_fpa_workbook() -> Optional[Path]:
    roots = [Path("data/first_pass_accuracy"), Path("first_pass_accuracy")]
    for root in roots:
        if not root.exists():
            continue
        hits = sorted(root.glob("FirstPassAccuracy*.xls*"))
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
        "date": _pick(df, ["Activity Date", "Date"]),
        "result": _pick(df, ["Review Result", "Result"]),
        "portfolio": _pick(df, ["Portfolio"]),
        "comment": _pick(df, ["Case Comment", "Comments"]),
        "rca2": _pick(df, ["RCA2", "Root Cause 2"]),
    }
    df = df.rename(columns={v: k for k, v in col_map.items() if v})
    df["date"] = pd.to_datetime(df["date"], errors="coerce", dayfirst=True)
    return df, col_map

# ======================
# Core helpers
# ======================
def _is_pass(x: str) -> bool:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return False
    return str(x).strip().lower().startswith("pass")

def _series_mom(df: pd.DataFrame) -> pd.DataFrame:
    df = df.assign(_m=_coerce_month(df["date"]))
    start = pd.Period("2025-01")
    months = pd.period_range(start, df["_m"].max(), freq="M")
    g = df.groupby("_m")["result"].agg(
        total="count", passed=lambda x: sum(_is_pass(v) for v in x)
    ).reindex(months, fill_value=0)
    pct = (g["passed"] * 100.0 / g["total"].replace(0, np.nan)).fillna(0).round(0)
    label = [m.to_timestamp().strftime("%b-%y") for m in months]
    return pd.DataFrame({"month": label, "pass_pct": pct.values})

def _table_portfolio_mom(df: pd.DataFrame) -> pd.DataFrame:
    df = df.assign(_m=_coerce_month(df["date"]))
    start = pd.Period("2025-01")
    months = pd.period_range(start, df["_m"].max(), freq="M")
    grp = df.groupby(["portfolio", "_m"])["result"].agg(
        total="count", passed=lambda x: sum(_is_pass(v) for v in x)
    ).reset_index()
    grp["pass_%"] = (grp["passed"] * 100.0 / grp["total"]).round(0)
    piv = grp.pivot(index="portfolio", columns="_m", values="pass_%").reindex(columns=months)
    piv.columns = [m.to_timestamp().strftime("%b-%y") for m in piv.columns]
    return piv.fillna(0).astype(int)

def _label_all(df: pd.DataFrame) -> pd.DataFrame:
    from core.reason_labeller import label_dataframe
    df = df.assign(_m=_coerce_month(df["date"]))
    fails = df[~df["result"].apply(_is_pass)].copy()
    if fails.empty:
        return pd.DataFrame(columns=list(df.columns) + ["reason"])
    lab_df = pd.DataFrame({
        "Case Comment": fails["comment"].fillna("").astype(str),
        "RCA2": fails["rca2"].fillna("").astype(str) if "rca2" in fails.columns else ""
    })
    fails["reason"] = label_dataframe(lab_df, "Case Comment", "RCA2").fillna("Other").astype(str)
    return fails

def _pivot_fail_matrix(fails: pd.DataFrame) -> pd.DataFrame:
    start = pd.Period("2025-01")
    fails = fails[fails["_m"] >= start]
    months = pd.period_range(start, fails["_m"].max(), freq="M")
    labels = [m.to_timestamp().strftime("%b-%y") for m in months]
    g = fails.groupby(["reason", "_m"]).size().reset_index(name="count")
    mat = g.pivot(index="reason", columns="_m", values="count").reindex(columns=months, fill_value=0)
    mat.columns = labels
    return mat

# ======================
# Plots
# ======================
def _fig_mom(df: pd.DataFrame, title: str):
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    ax.plot(df["month"], df["pass_pct"], linewidth=3.2, color=_PASTEL_LINE)
    for x, y in zip(df["month"], df["pass_pct"]):
        ax.text(x, y+2, f"{y:.0f}%", ha="center", fontsize=9, color=_DARK_GREY)
    for sp in ["left","right","top"]: ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color(_SOFT_GREY)
    ax.get_yaxis().set_visible(False)
    ax.set_title(title, color=_DARK_BLUE)
    return fig

def _fig_fail_trends(mat: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(9,4))
    months = mat.columns
    for i, reason in enumerate(mat.index):
        ax.plot(months, mat.loc[reason], label=reason,
                color=_PASTEL_LINES[i % len(_PASTEL_LINES)], linewidth=2.5)
    ax.legend(fontsize=8, bbox_to_anchor=(1.01, 1), loc="upper left")
    for sp in ["left","right","top"]: ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color(_SOFT_GREY)
    ax.get_yaxis().set_visible(False)
    ax.grid(False)
    ax.set_xlabel("Month")
    ax.set_title("MoM Fail Reasons Trends", color=_DARK_BLUE)
    return fig

# ======================
# Insights (OpenAI)
# ======================
def _generate_insights(mom: pd.DataFrame, piv_portfolio: pd.DataFrame, fail_matrix: pd.DataFrame) -> List[str]:
    client = OpenAI()
    prompt = f"""
    You are an analytics assistant. Summarize trends in 3-4 bullet points:
    1. MoM pass % from {mom['month'].iloc[0]} to {mom['month'].iloc[-1]}.
    2. Highlight portfolios with biggest MoM ↑/↓.
    3. Trends of top 2 fail reasons.
    4. Top portfolio contributors to those fails.
    Data:
    Pass%:\n{mom.to_string()}
    Portfolio MoM:\n{piv_portfolio.to_string()}
    Fail reasons MoM:\n{fail_matrix.to_string()}
    """
    resp = client.chat.completions.create(model="gpt-4o-mini",
                                          messages=[{"role":"user","content":prompt}],
                                          temperature=0.4)
    return resp.choices[0].message.content.split("\n")

# ======================
# Entry
# ======================
def run(store: Dict, params: Dict, user_text: str="") -> Tuple[str,pd.DataFrame]:
    try:
        df_raw,_ = _load_fpa()
    except Exception as e:
        st.error(str(e)); return "", pd.DataFrame()

    mom = _series_mom(df_raw)
    if mom.empty: return "", pd.DataFrame()
    latest = df_raw["_m"].max()
    piv_portfolio = _table_portfolio_mom(df_raw)
    fails_all = _label_all(df_raw)
    fail_matrix = _pivot_fail_matrix(fails_all)

    # ----- Insights -----
    with st.spinner("Generating insights..."):
        insights = _generate_insights(mom, piv_portfolio, fail_matrix)
    st.markdown("### AI Insights")
    for point in insights:
        if point.strip(): st.markdown(f"- {point.strip()}")

    # ----- Row 1 -----
    c1,c2 = st.columns((1.1,1.0))
    with c1: st.pyplot(_fig_mom(mom, f"First-Pass Accuracy — Jan–{latest.to_timestamp().strftime('%b %y')}"))
    with c2: st.dataframe(piv_portfolio, use_container_width=True)

    # ----- Row 2 (Pareto + matrix kept from base code) -----
    # … keep your existing Row 2 code (omitted for brevity) …

    # ----- Row 3 -----
    st.markdown("### Month-on-Month Fail Reason Trends")
    c3,c4 = st.columns((1.2,1.0))
    with c3:
        if not fail_matrix.empty: st.pyplot(_fig_fail_trends(fail_matrix))
    with c4:
        if not fail_matrix.empty: st.dataframe(fail_matrix, use_container_width=True)

    return "", pd.DataFrame()
