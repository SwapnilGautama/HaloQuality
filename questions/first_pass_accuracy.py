# -*- coding: utf-8 -*-
# questions/first_pass_accuracy.py
from __future__ import annotations

import os, re, glob
from io import BytesIO
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib import patheffects as pe

# Optional OpenAI labelling (used only if OPENAI_API_KEY is available)
_OPENAI_READY = False
try:
    import openai  # type: ignore
    if os.getenv("OPENAI_API_KEY"):
        openai.api_key = os.getenv("OPENAI_API_KEY")
        _OPENAI_READY = True
except Exception:
    _OPENAI_READY = False


# ---------- helpers ----------

PASTELS = ["#9ecae1", "#a1d99b", "#bdbdbd", "#fdd0a2", "#bdb8ff", "#c7e9c0"]


def _clean_cols(df: pd.DataFrame) -> pd.DataFrame:
    m = {c: re.sub(r"[^a-z0-9]+", "_", c.strip().lower()) for c in df.columns}
    df = df.rename(columns=m)

    # Flexible remap
    alias = {
        "portfolio": ["portfolio", "portfolios", "site", "region"],
        "scheme": ["scheme", "scheme_name", "plan", "product"],
        "activity_date": ["activity_date", "date", "created_date", "review_date"],
        "review_result": ["review_result", "result", "review_outcome", "outcome"],
        "case_comment": ["case_comment", "comment", "comments", "remarks", "notes"],
        "case_id": ["case_id", "id", "reference", "ref", "ref_id"],
    }
    remap = {}
    for want, candidates in alias.items():
        for c in candidates:
            if c in df.columns:
                remap[c] = want
                break
    df = df.rename(columns=remap)

    required = ["portfolio", "scheme", "activity_date", "review_result"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in FPA: {missing}")

    return df


def _read_fpa() -> pd.DataFrame:
    """
    Read First Pass Accuracy file from:
      - data/first_pass_accuracy/FirstPassAccuracy*.xlsx  (preferred)
      - /mnt/data/FirstPassAccuracy_Aug'25.xlsx          (fallback)
    """
    # Preferred: repo data folder
    candidates = glob.glob("data/first_pass_accuracy/FirstPassAccuracy*.xlsx")
    if not candidates:
        # Any Excel in the subfolder
        candidates = glob.glob("data/first_pass_accuracy/*.xlsx")

    # Fallback: uploaded path used during development
    fallback = "/mnt/data/FirstPassAccuracy_Aug'25.xlsx"
    path = candidates[0] if candidates else (fallback if os.path.exists(fallback) else None)
    if not path:
        raise FileNotFoundError(
            "FPA source file not found. Place FirstPassAccuracy*.xlsx under data/first_pass_accuracy/"
        )

    df = pd.read_excel(path, engine="openpyxl")
    return _clean_cols(df)


def _month_key(dt: pd.Series) -> pd.Series:
    s = pd.to_datetime(dt, errors="coerce")
    return s.dt.to_period("M").astype(str)  # e.g., '2025-06'


def _month_label(ym: str) -> str:
    # '2025-06' -> 'Jun-25'
    y, m = ym.split("-")
    d = pd.to_datetime(f"{ym}-01")
    return d.strftime("%b-%y")


def _is_pass(val: str) -> bool:
    s = str(val).strip().lower()
    if s in ("pass", "passed", "p", "ok", "success", "true", "yes"):
        return True
    # Treat everything explicitly marked fail as fail
    if "fail" in s:
        return False
    # Conservative default if ambiguous
    return False


# Simple fallback taxonomy for fail drivers
_FALLBACK_RULES: List[Tuple[str, List[str]]] = [
    ("Waiting on member/TPA", ["waiting", "chase", "await", "member reply", "no response", "outstanding"]),
    ("Bank/Payment issue", ["bacs", "bank", "payment", "sort code", "account", "cheque"]),
    ("Postal delay", ["post", "postal", "mail", "courier", "shipment", "delivered late"]),
    ("Manual calculation", ["manual calc", "manual calculation", "calc error", "calculation error", "spreadsheet"]),
    ("Data entry error", ["keyed", "typo", "data entry", "transposed", "mismatch", "input error"]),
    ("Case not created", ["case not created", "not created", "missing case", "no case"]),
    ("Trustee", ["trustee", "trustees"]),
    ("AVC", ["avc"]),
    ("Death benefits payout", ["death benefit", "death payout", "dbp"]),
    ("Pension set up", ["set up", "setup", "onboarding", "pension set"]),
    ("Scheme rules", ["scheme rule", "rule", "rules"]),
    ("System", ["system", "it issue", "workflow", "bug", "outage"]),
]


def _label_comment_openai(txts: List[str]) -> List[str]:
    """
    Only used if OPENAI_API_KEY present. We keep this simple and batch
    friendly — if anything goes wrong we silently fall back to rules.
    """
    assert _OPENAI_READY
    out = []
    prompt = (
        "You are classifying complaint review failure comments into a single best reason. "
        "Choose one of: [Waiting on member/TPA, Bank/Payment issue, Postal delay, Manual calculation, "
        "Data entry error, Case not created, Trustee, AVC, Death benefits payout, Pension set up, "
        "Scheme rules, System, Other]. Only answer with the label.\n"
        "Comment: "
    )
    try:
        # small, inexpensive model
        resp = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": "You classify failure reasons."}]
            + [{"role": "user", "content": prompt + t[:500]} for t in txts],
            temperature=0.1,
        )
        for choice in resp["choices"]:
            lab = (choice["message"]["content"] or "Other").strip()
            # guardrail to known labels
            if lab not in [r[0] for r in _FALLBACK_RULES] + ["Other"]:
                lab = "Other"
            out.append(lab)
        # If API returns fewer (shouldn't), pad with Other
        while len(out) < len(txts):
            out.append("Other")
        return out
    except Exception:
        return []  # signals caller to fallback


def _label_comment_rules(txt: str) -> str:
    s = str(txt or "").lower()
    for label, keys in _FALLBACK_RULES:
        if any(k in s for k in keys):
            return label
    return "Other"


def _label_fail_drivers(df: pd.DataFrame) -> pd.DataFrame:
    fails = df.loc[~df["passed"].astype(bool)].copy()
    if fails.empty:
        return pd.DataFrame(columns=["reason", "count", "percent", "cum_percent"])

    reasons: List[str] = []
    comments = fails.get("case_comment")
    if comments is None:
        comments = pd.Series([""] * len(fails))

    if _OPENAI_READY:
        labs = _label_comment_openai(comments.fillna("").tolist())
        if labs:
            reasons = labs
    if not reasons:
        reasons = [ _label_comment_rules(x) for x in comments.fillna("").tolist() ]

    fails["reason"] = reasons
    agg = fails.groupby("reason", dropna=False).size().reset_index(name="count").sort_values("count", ascending=False)
    agg["percent"] = 100 * agg["count"] / agg["count"].sum()
    agg["cum_percent"] = agg["percent"].cumsum()
    # Pretty rounding (single decimal for percents)
    agg["percent"] = agg["percent"].round(1)
    agg["cum_percent"] = agg["cum_percent"].round(1)
    return agg.reset_index(drop=True)


def _style_table(df: pd.DataFrame, title: str):
    st.markdown(f"<h4 style='color:#0b3d91;margin-top:0.75rem'>{title}</h4>", unsafe_allow_html=True)
    st.dataframe(df, use_container_width=True)


def _plot_mom_pass(mom: pd.DataFrame, title: str):
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    ax.plot(mom["label"], mom["pass_pct"], marker="o", linewidth=2.2, color="#9ecae1")
    for i, r in mom.iterrows():
        ax.text(
            i, r["pass_pct"] + 0.8, f"{r['pass_pct']:.1f}%",
            ha="center", va="bottom", color="#555", fontsize=9,
            path_effects=[pe.withStroke(linewidth=3, foreground="white", alpha=.6)]
        )
    ax.set_ylim(0, max(100, np.ceil(mom["pass_pct"].max()/10)*10))
    ax.set_ylabel("")
    ax.set_xlabel("")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.grid(axis="y", alpha=.15)
    # Soft grey x-axis line
    ax.axhline(0, color="#e6e6e6", linewidth=1.5)
    plt.xticks(rotation=0, color="#333")
    ax.set_title(title, color="#0b3d91", fontsize=12, pad=6)
    st.pyplot(fig, clear_figure=True)


def _prepare(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Dates & month keys
    df["activity_date"] = pd.to_datetime(df["activity_date"], errors="coerce")
    df["month"] = df["activity_date"].dt.to_period("M").astype(str)
    # Pass flag
    df["passed"] = df["review_result"].apply(_is_pass).astype(bool)
    # Normalise text cols
    for c in ["portfolio", "scheme"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()
    return df


# ---------- public entry point ----------

def run(store: Dict, params: Dict, user_text: str):
    """
    Renders First-Pass Accuracy dashboard. Also returns a tuple to be friendly
    with engines that expect (summary, df).
    """
    df_raw = _read_fpa()
    df = _prepare(df_raw)

    # Month window: Jan–Jun 2025 (adjust here if you want dynamic)
    want_months = pd.period_range("2025-01", "2025-06", freq="M").astype(str).tolist()
    df_win = df[df["month"].isin(want_months)].copy()

    total_rows = len(df_win)
    total_pass = int(df_win["passed"].sum())
    overall_pct = 100 * total_pass / max(1, total_rows)

    st.markdown("<h3 style='color:#0b3d91;margin:.25rem 0'>First-Pass Accuracy — Jan–Jun 2025</h3>", unsafe_allow_html=True)
    st.caption(f"Rows={total_rows:,}  ·  Passed={total_pass:,}  ·  Overall={overall_pct:.1f}%")

    # ---- Row 1: MoM + portfolio×scheme table (Jun-25) ----
    col1, col2 = st.columns([1.05, 1.1], gap="large")

    with col1:
        mom = (
            df_win.groupby("month")["passed"]
            .mean()
            .mul(100)
            .reset_index(name="pass_pct")
            .sort_values("month")
        )
        mom["label"] = mom["month"].apply(_month_label)
        _plot_mom_pass(mom, "Pass % — MoM")

    with col2:
        # Pass % by Portfolio × Scheme (June 2025)
        june_key = "2025-06"
        june = df_win[df_win["month"] == june_key].copy()
        if june.empty:
            out = pd.DataFrame(columns=["portfolio", "scheme", "cases", "pass_pct"])
        else:
            grp = june.groupby(["portfolio", "scheme"])["passed"].agg(["count", "mean"]).reset_index()
            grp = grp.rename(columns={"count": "cases", "mean": "pass_pct"})
            grp["pass_pct"] = (grp["pass_pct"] * 100).round(1)
            out = grp.sort_values(["portfolio", "pass_pct"], ascending=[True, False])
        _style_table(out, "Pass % by Portfolio × Scheme — Jun-25")

    # ---- Row 2: RCA on fails (drivers) ----
    st.markdown("<div style='height:.25rem'></div>", unsafe_allow_html=True)
    col3, col4 = st.columns([1, 1.05], gap="large")

    with col3:
        # Pareto bars of RCA1-style (grouped reasons)
        drivers = _label_fail_drivers(df_win)
        if not drivers.empty:
            # Order by count desc
            drv = drivers.copy().sort_values("count", ascending=False).reset_index(drop=True)
            # Make a pared-down bar chart in descending order
            fig, ax = plt.subplots(figsize=(6.2, 3.8))
            bars = ax.bar(drv["reason"], drv["count"], color=PASTELS, edgecolor="#ddd")
            # cumulative line
            ax2 = ax.twinx()
            ax2.plot(drv["reason"], drv["cum_percent"], marker="o", color="#79a8d8", linewidth=2.0)
            for i, r in drv.iterrows():
                ax2.text(i, r["cum_percent"] + 1.2, f"{r['cum_percent']:.0f}%",
                         ha="center", va="bottom", color="#555", fontsize=9,
                         path_effects=[pe.withStroke(linewidth=3, foreground="white", alpha=.6)])
            # cosmetics
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_visible(False)
            ax2.spines["top"].set_visible(False)
            ax2.spines["left"].set_visible(False)
            ax2.spines["right"].set_visible(False)
            ax.grid(axis="y", alpha=.12)
            ax.set_ylabel("Count", color="#555")
            ax2.set_ylabel("Cum. %", color="#555")
            ax2.set_ylim(0, 110)
            plt.xticks(rotation=90)
            ax.set_title("Fail drivers — Pareto", color="#0b3d91", fontsize=12, pad=6)
            st.pyplot(fig, clear_figure=True)
        else:
            st.info("No fails in the selected window — nice!")

    with col4:
        if not drivers.empty:
            tbl = drivers.rename(columns={
                "reason": "RCA (label)",
                "count": "count",
                "percent": "percent",
                "cum_percent": "cum_percent",
            })
            # Single-decimal for percents, no index col
            tbl["percent"] = tbl["percent"].round(1)
            tbl["cum_percent"] = tbl["cum_percent"].round(1)
            _style_table(tbl[["RCA (label)", "count", "percent", "cum_percent"]], "Fail drivers — table")
        else:
            st.empty()

    # Return a minimal tuple for engines that expect a value
    # Provide the June portfolio×scheme table as the second element
    summary = f"FPA Jan–Jun 2025. Overall pass={overall_pct:.1f}% on {total_rows:,} reviews."
    return summary, out if "out" in locals() else pd.DataFrame()
