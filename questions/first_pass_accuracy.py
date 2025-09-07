# questions/first_pass_accuracy.py
from __future__ import annotations
import os, re
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

PASTEL = ["#9ec5fe", "#a3d2a3", "#f6c48f", "#f7a3a3", "#b8b8ff", "#ffd6a5", "#b9e6ff"]

def _load_latest_fpa_file(data_root: Path) -> Path:
    folder = data_root / "first_pass_accuracy"
    cand = sorted(folder.glob("FirstPassAccuracy*.xlsx"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not cand:
        raise FileNotFoundError("Could not find a FirstPassAccuracy workbook (FirstPassAccuracy*.xlsx).")
    return cand[0]

def _coerce_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Normalize column names
    cols = {c.lower().strip(): c for c in df.columns}
    # Map common aliases
    def _pick(*names):
        for n in names:
            if n in cols:
                return cols[n]
        return None

    col_res = _pick("review result", "result", "reviewresult")
    col_dt  = _pick("activity date", "date", "activitydate")
    col_cmt = _pick("case comment", "comment", "comments", "note", "notes")
    col_pf  = _pick("portfolio")
    col_sch = _pick("scheme", "plan", "plan name", "scheme name")

    missing = [n for n, v in {
        "review result": col_res, "activity date": col_dt, "portfolio": col_pf
    }.items() if v is None]
    if missing:
        raise KeyError(f"Missing columns: {missing}")

    out = pd.DataFrame({
        "review_result": df[col_res],
        "activity_date": pd.to_datetime(df[col_dt], errors="coerce"),
        "portfolio": df[col_pf],
    })
    out["scheme"] = df[col_sch] if col_sch else ""
    out["comment"] = df[col_cmt] if col_cmt else ""
    return out

def _month_range(start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
    start = pd.Timestamp(start.year, start.month, 1)
    end   = pd.Timestamp(end.year, end.month, 1)
    return pd.date_range(start, end, freq="MS")

def _draw_mom(ax: plt.Axes, idx: pd.DatetimeIndex, values: pd.Series, title: str):
    y = values.reindex(idx, fill_value=0.0)
    line, = ax.plot(idx, y, marker="o", linewidth=2.5, color="#5f8cff")
    for x, v in zip(idx, y.values):
        ax.text(x, v, f"{v:.0f}%", fontsize=10, ha="center", va="bottom")
    ax.set_title(title, color="#0d3b82", fontsize=14, pad=8)
    ax.set_xticks(idx)
    ax.set_xticklabels([d.strftime("%b-%y") for d in idx], rotation=0)
    ax.spines["left"].set_visible(False)
    ax.yaxis.set_visible(False)
    for spine in ("top","right"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#d0d4dd")
    ax.grid(False)

# ----------- Reason labelling (no-API version; robust keywords) -----------
_REASON_PATTERNS: List[Tuple[str, List[str]]] = [
    ("Bank / payment", ["bank","payment","paym","bacs","cheque","refund","funds","pay-out","payout"]),
    ("Trustee / AVC", ["trustee","avc","governance","approval","authoris","trust office"]),
    ("Data entry / setup", ["data","setup","key","input","record","update","address","dob","name","member data","typo","correct","amend"]),
    ("Postal / dispatch", ["post","postal","mail","dispatch","letter","sent","receive","courier"]),
    ("Manual calculation", ["manual calc","manual", "calc", "calculation","recalc","re-calc"]),
    ("System", ["system","technical","it issue","bug","workflow","automation","server","down"]),
    ("Waiting on member/TPA", ["waiting","awaiting","member response","no response","chase","tpa","third party"]),
]

def _label_reason(text: pd.Series) -> pd.Series:
    s = (text.fillna("").astype(str).str.lower())
    out = pd.Series(index=s.index, dtype="object")
    hit_any = pd.Series(False, index=s.index)
    for name, kws in _REASON_PATTERNS:
        # escape keywords safely & match whole words when possible
        pat = r"|".join([re.escape(k) for k in kws])
        mask = s.str.contains(pat, regex=True)
        out[mask & ~hit_any] = name
        hit_any |= mask
    out[~hit_any] = "Other"
    return out

def _top80_pareto(counts: pd.Series, title: str):
    # counts: Series indexed by reason
    df = counts.sort_values(ascending=False).rename_axis("reason").reset_index(name="count")
    total = df["count"].sum()
    df["percent"] = df["count"] / max(total, 1) * 100
    df["cum_percent"] = df["percent"].cumsum()
    # keep until we pass 80; remainder becomes "Other"
    cutoff = df[df["cum_percent"] <= 80]
    remainder = df[df["cum_percent"] > 80]
    if not remainder.empty:
        row = pd.DataFrame([{
            "reason": "Other",
            "count": int(remainder["count"].sum()),
            "percent": remainder["percent"].sum(),
            "cum_percent": 100.0
        }])
        pareto = pd.concat([cutoff, row], ignore_index=True)
    else:
        pareto = df

    fig, ax1 = plt.subplots(figsize=(7,4))
    # Bar (keep container!)
    bar_container = ax1.bar(pareto["reason"], pareto["count"], color=PASTEL[0])
    ax1.bar_label(bar_container, labels=[f"{int(v)}" for v in pareto["count"]], padding=3)
    ax1.yaxis.set_visible(False)
    for sp in ("top","right","left"): ax1.spines[sp].set_visible(False)
    ax1.spines["bottom"].set_color("#d0d4dd")
    ax1.set_xticklabels(pareto["reason"], rotation=90)
    ax1.grid(False)

    # Cumulative %
    ax2 = ax1.twinx()
    cum = pareto["percent"].cumsum()
    ax2.plot(pareto["reason"], cum, marker="o", linewidth=2.0, color="#4f6cdf")
    for x, v in zip(pareto["reason"], cum):
        ax2.text(x, v, f"{v:.0f}%", ha="center", va="bottom", fontsize=9)
    ax2.set_ylim(0, 104)
    ax2.set_ylabel("")
    ax2.spines["right"].set_visible(False)
    ax2.yaxis.set_visible(False)

    ax1.set_title(title, color="#0d3b82", fontsize=14, pad=8)
    fig.tight_layout()
    return pareto, fig

# ---------------- Main render ----------------
def run(store: Dict, params: Dict, q: str):
    root = store.get("root", Path(__file__).parents[1])
    data_root = store.get("data", root / "data")

    f = _load_latest_fpa_file(data_root)
    df_raw = pd.read_excel(f)
    df = _coerce_columns(df_raw)

    # Pass flag
    df["passed"] = df["review_result"].astype(str).str.contains("pass", case=False, na=False)
    # Month
    df["month"] = df["activity_date"].dt.to_period("M").dt.to_timestamp()

    # Month range Jan-2025..latest present in data
    first = pd.Timestamp(2025,1,1)
    last  = df["month"].dropna().max()
    months = _month_range(max(first, df["month"].min()), last)

    # MoM pass %
    mom = (df.groupby("month")["passed"].mean()*100).reindex(months, fill_value=0.0)

    # Title
    st.markdown(f"## First-Pass Accuracy — {months[0].strftime('%b-%y')}–{months[-1].strftime('%b-%y')}")

    # Row 1 — MoM + Pass% by portfolio×scheme (last month)
    col1, col2 = st.columns([1.1, 1.2])
    with col1:
        fig, ax = plt.subplots(figsize=(7,3.4))
        _draw_mom(ax, months, mom, "Pass % — MoM")
        st.pyplot(fig, use_container_width=True)

    with col2:
        latest = months[-1]
        df_latest = df[df["month"] == latest].copy()
        if df_latest.empty:
            st.info("No rows for the most recent month in the file.")
        else:
            tbl = (
                df_latest.groupby(["portfolio","scheme"])["passed"]
                .mean().mul(100).round(0)
                .reset_index(name="pass_%")
                .sort_values(["portfolio","scheme"])
            )
            st.markdown(f"#### Pass % by Portfolio × Scheme — {latest.strftime('%b-%y')}")
            st.dataframe(tbl, use_container_width=True, hide_index=True)

    # Row 2 — Fail reasons (latest month): Table + Pareto
    st.markdown(f"### Reasons for Fail — {months[-1].strftime('%b-%y')}")

    fail_latest = df[(df["month"] == months[-1]) & (~df["passed"])].copy()
    if fail_latest.empty:
        st.info("No failed records for the most recent month.")
        return

    fail_latest["reason"] = _label_reason(fail_latest["comment"])
    counts = fail_latest["reason"].value_counts()

    pr1, pr2 = st.columns([1.1, 1.2])

    with pr1:
        pareto_df, fig = _top80_pareto(counts, title="Fail reasons — Pareto (top 80%)")
        st.pyplot(fig, use_container_width=True)

    with pr2:
        tbl = (
            counts.rename_axis("reason").reset_index(name="count")
            .assign(percent=lambda d: d["count"]/d["count"].sum()*100)
            .assign(cum_percent=lambda d: d["percent"].cumsum())
        )
        st.markdown(f"#### Reason breakdown (top 80%) — {months[-1].strftime('%b-%y')}")
        st.dataframe(tbl, use_container_width=True, hide_index=True)
