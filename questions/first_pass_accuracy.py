# -*- coding: utf-8 -*-
# questions/first_pass_accuracy.py
from __future__ import annotations

import os
import re
from io import BytesIO
from typing import Dict, List, Tuple
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# Optional OpenAI enrichment
# -----------------------------------------------------------------------------
_OPENAI_READY = False
_OPENAI_MODEL = os.getenv("OPENAI_FPA_MODEL", "gpt-4o-mini")  # lightweight & cheap
try:
    import openai  # type: ignore
    if os.getenv("OPENAI_API_KEY"):
        openai.api_key = os.getenv("OPENAI_API_KEY")
        _OPENAI_READY = True
except Exception:
    _OPENAI_READY = False


# -----------------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def _load_fpa(root: str) -> pd.DataFrame:
    """
    Load the latest FirstPassAccuracy workbook from data/first_pass_accuracy/*.
    We allow flexible file names that start with 'FirstPassAccuracy'.
    """
    base = os.path.join(root, "data", "first_pass_accuracy")
    if not os.path.isdir(base):
        raise FileNotFoundError(f"Folder not found: {base}")

    # pick latest FirstPassAccuracy*.xlsx
    xl_files = sorted(
        [f for f in os.listdir(base) if f.lower().startswith("firstpassaccuracy") and f.lower().endswith(".xlsx")]
    )
    if not xl_files:
        raise FileNotFoundError("Could not find a FirstPassAccuracy workbook (FirstPassAccuracy*.xlsx).")

    path = os.path.join(base, xl_files[-1])
    df = pd.read_excel(path)

    # normalize columns
    rename_map = {
        "Activity Date": "activity_date",
        "Review Result": "review_result",
        "Portfolio": "portfolio",
        "Scheme": "scheme",
        "Scheme Name": "scheme",
        "Case Comment": "comment",
        "Case Comments": "comment",
    }
    for k, v in rename_map.items():
        if k in df.columns and v not in df.columns:
            df[v] = df[k]

    # minimally required columns
    needed = ["activity_date", "review_result", "portfolio", "scheme"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns in FPA data: {missing}. Found: {list(df.columns)}")

    # comment column is optional (only needed for reasons)
    if "comment" not in df.columns:
        df["comment"] = ""

    # coerce types
    df["activity_date"] = pd.to_datetime(df["activity_date"], errors="coerce")
    df["review_result"] = df["review_result"].astype(str).str.strip().str.lower()
    df["portfolio"] = df["portfolio"].astype(str).str.strip()
    df["scheme"] = df["scheme"].astype(str).str.strip()
    df["comment"] = df["comment"].fillna("").astype(str)

    # keep sane rows
    df = df[~df["activity_date"].isna()].copy()
    df["year_month"] = df["activity_date"].dt.to_period("M").astype(str)

    return df


# -----------------------------------------------------------------------------
# Helper: Month range & fill
# -----------------------------------------------------------------------------
def _mmm_yy(dt: pd.Timestamp) -> str:
    return dt.strftime("%b-%y")

def _month_frame(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    ix = pd.period_range(start=start, end=end, freq="M").to_timestamp("M")
    return pd.DataFrame({"month": ix, "mmm_yy": [_mmm_yy(x) for x in ix]})

def _pass_pct_mom(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("year_month", as_index=False).agg(
        total=("review_result", "size"),
        passed=("review_result", lambda s: (s == "pass").sum()),
    )
    g["pass_pct"] = (g["passed"] / g["total"]).fillna(0) * 100

    # full Jan->latest with 0 fills
    min_dt = df["activity_date"].min().to_period("M").to_timestamp("M")
    max_dt = df["activity_date"].max().to_period("M").to_timestamp("M")
    months = _month_frame(min_dt, max_dt)

    g2 = months.merge(
        g.assign(month=pd.to_datetime(g["year_month"]).dt.to_period("M").dt.to_timestamp("M")),
        on="month",
        how="left",
    ).fillna({"total": 0, "passed": 0, "pass_pct": 0})
    g2["pass_pct"] = g2["pass_pct"].round(1)
    return g2[["mmm_yy", "pass_pct"]]


# -----------------------------------------------------------------------------
# Helper: latest-month table (Pass % by portfolio × scheme)
# -----------------------------------------------------------------------------
def _latest_pass_by_portfolio_scheme(df: pd.DataFrame) -> pd.DataFrame:
    last_ym = df["year_month"].max()
    d1 = df[df["year_month"] == last_ym].copy()
    if d1.empty:
        return pd.DataFrame(columns=["portfolio", "scheme", "cases", "pass_%"])

    grp = d1.groupby(["portfolio", "scheme"]).agg(
        cases=("review_result", "size"),
        passed=("review_result", lambda s: (s == "pass").sum()),
    )
    grp["pass_%"] = (grp["passed"] / grp["cases"] * 100).round(0)
    out = grp.reset_index().sort_values(["portfolio", "pass_%", "cases"], ascending=[True, False, False])
    return out[["portfolio", "scheme", "cases", "pass_%"]]


# -----------------------------------------------------------------------------
# Reason classification (keyword + optional OpenAI enrichment)
# -----------------------------------------------------------------------------
_REASON_PATTERNS: Dict[str, List[str]] = {
    "Bank / payment": [
        r"\b(bac[hs]|bank|payment|paid|credit|debit|cheque|refund|charge|bacs)\b",
        r"sort\s*code|iban|swift|account\s*no|payee",
    ],
    "Trustee / AVC": [r"\b(trustee|avc|additional voluntary|with[-\s]*profits)\b"],
    "Data entry / setup": [r"\b(data\s*entry|setup|set[-\s]*up|onboard|on[-\s]*boarding|register|index|scan)\b"],
    "Postal / dispatch": [r"\b(post|mail|dispatch|envelope|letter|sent|deliver|courier|doc(ument)?(ation)?)\b"],
    "Manual calculation": [r"\b(manual|calc|recalc|spreadsheet|hand[-\s]*calc)\b"],
    "System": [r"\b(system|portal|technical|bug|glitch|down|error|timeout|access|login)\b"],
    "Waiting on member/TPA": [r"\b(chase|await|waiting|member\s*reply|no\s*response|tpa)\b"],
    "Other": [r".*"],
}
_REASON_REGEX: List[Tuple[str, re.Pattern]] = []
for label, patterns in _REASON_PATTERNS.items():
    _REASON_REGEX.append((label, re.compile("|".join(patterns), flags=re.I)))

def _label_reason_kw(texts: List[str]) -> List[str]:
    out = []
    for t in texts:
        t0 = t.strip()
        if not t0:
            out.append("Other")
            continue
        label = "Other"
        for lab, rx in _REASON_REGEX:
            if rx.search(t0):
                label = lab
                break
        out.append(label)
    return out

def _label_reason_ai_batch(texts: List[str]) -> List[str]:
    if not _OPENAI_READY:
        return _label_reason_kw(texts)

    sys_prompt = (
        "You classify pension admin 'Case Comments' into one of these labels only:\n"
        " - Bank / payment\n - Trustee / AVC\n - Data entry / setup\n - Postal / dispatch\n"
        " - Manual calculation\n - System\n - Waiting on member/TPA\n - Other\n\n"
        "Rules: Return ONLY the label text. If unsure, pick the closest label (avoid 'Other' if a clear match exists)."
    )
    labels = []
    for chunk_start in range(0, len(texts), 25):
        chunk = texts[chunk_start:chunk_start+25]
        user_prompt = "Classify each line on its own line:\n" + "\n".join([f"- {t[:500]}" for t in chunk])

        try:
            resp = openai.ChatCompletion.create(
                model=_OPENAI_MODEL,
                messages=[{"role": "system", "content": sys_prompt},
                          {"role": "user", "content": user_prompt}],
                temperature=0.0,
                max_tokens=200,
            )
            content = resp["choices"][0]["message"]["content"]
            batch_out = []
            for line in content.splitlines():
                lab = line.strip("-• \t\r\n")
                if lab not in _REASON_PATTERNS:
                    k = lab.lower()
                    mapped = next((L for L in _REASON_PATTERNS.keys() if L.lower() == k), None)
                    lab = mapped if mapped else "Other"
                batch_out.append(lab)
            if len(batch_out) != len(chunk):
                batch_out = _label_reason_kw(chunk)
            labels.extend(batch_out)
        except Exception:
            labels.extend(_label_reason_kw(chunk))
    return labels


def _top80_pareto(series_counts: pd.Series, title: str) -> Tuple[pd.DataFrame, plt.Figure]:
    dfc = series_counts.sort_values(ascending=False).rename_axis("reason").reset_index(name="count")
    dfc["percent"] = (dfc["count"] / dfc["count"].sum() * 100)
    dfc["cum_percent"] = dfc["percent"].cumsum()

    cutoff = dfc[dfc["cum_percent"] <= 80.0]
    tail = dfc[dfc["cum_percent"] > 80.0]
    if not tail.empty:
        other_row = pd.DataFrame(
            {"reason": ["Other"], "count": [int(tail["count"].sum())], "percent": [tail["percent"].sum()], "cum_percent": [100.0]}
        )
        top80 = pd.concat([cutoff, other_row], ignore_index=True)
    else:
        top80 = dfc.copy()

    fig, ax1 = plt.subplots(figsize=(7.2, 4.2))
    bars = ax1.bar(top80["reason"], top80["count"])
    for b in bars:
        ax1.bar_label([b], labels=[f"{int(b.get_height())}"], padding=3)

    cum = top80["percent"].cumsum()
    ax1.plot(top80["reason"], cum, marker="o", linewidth=2, color="#6aa0f8")

    ax1.grid(False)
    ax1.set_ylabel("")
    ax1.set_xlabel("")
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.spines["left"].set_visible(False)
    ax1.tick_params(axis="y", left=False, labelleft=False)
    ax1.tick_params(axis="x", rotation=90)
    ax1.set_title(title, loc="left", color="#0b3d91", fontweight="bold")

    return top80, fig


def _render_pass_mom(df_m: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(7.6, 2.8))
    ax.plot(df_m["mmm_yy"], df_m["pass_pct"], marker="o", linewidth=2, color="#6aa0f8")
    for x, y in zip(df_m["mmm_yy"], df_m["pass_pct"]):
        ax.text(x, y + 1.2, f"{y:.0f}%", ha="center", va="bottom", fontsize=10)

    ax.grid(False)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", which="both", left=False, labelleft=False)
    ax.set_title("Pass % — MoM", loc="left", color="#0b3d91", fontweight="bold")
    st.pyplot(fig, clear_figure=True)


def _latest_pass_by_portfolio_scheme(df: pd.DataFrame) -> pd.DataFrame:
    last_ym = df["year_month"].max()
    d1 = df[df["year_month"] == last_ym].copy()
    if d1.empty:
        return pd.DataFrame(columns=["portfolio", "scheme", "cases", "pass_%"])

    grp = d1.groupby(["portfolio", "scheme"]).agg(
        cases=("review_result", "size"),
        passed=("review_result", lambda s: (s == "pass").sum()),
    )
    grp["pass_%"] = (grp["passed"] / grp["cases"] * 100).round(0)
    out = grp.reset_index().sort_values(["portfolio", "pass_%", "cases"], ascending=[True, False, False])
    return out[["portfolio", "scheme", "cases", "pass_%"]]


# -----------------------------------------------------------------------------
# Main entry
# -----------------------------------------------------------------------------
def run(store: Dict, params: Dict, q: str):
    """
    First-pass accuracy analysis (Jan–latest):
      • Pass% MoM (Jan -> latest; 0 for missing months)
      • Pass % by Portfolio × Scheme (latest month)
      • Fail reasons (OpenAI enrichment if available) — Pareto (top 80%) + table side-by-side
    """
    # Safe fallback if the app didn't pass root
    root = store.get("root")
    if not root:
        # /questions/<this_file> -> project root is parents[1]
        root = str(Path(__file__).resolve().parents[1])

    df = _load_fpa(root)

    min_dt = df["activity_date"].min()
    max_dt = df["activity_date"].max()
    st.markdown(f"## First-Pass Accuracy — {min_dt.strftime('%b-%y')}–{max_dt.strftime('%b-%y')}")

    c1, c2 = st.columns([1.1, 1.3], gap="large")
    with c1:
        _render_pass_mom(_pass_pct_mom(df))
    with c2:
        last_ym = df["year_month"].max()
        st.markdown(f"**Pass % by Portfolio × Scheme — {pd.to_datetime(last_ym).strftime('%b-%y')}**")
        st.dataframe(_latest_pass_by_portfolio_scheme(df), use_container_width=True, hide_index=True)

    # Reasons (latest month, fails only)
    st.markdown("---")
    last_ym = df["year_month"].max()
    d_last = df[df["year_month"] == last_ym].copy()
    fails = d_last[d_last["review_result"] != "pass"].copy()

    st.markdown(f"### Reasons for Fail — {pd.to_datetime(last_ym).strftime('%b-%y')}")
    if fails.empty or fails["comment"].str.strip().eq("").all():
        st.info("No fail records with comments found for the latest month.")
        return

    texts = fails["comment"].astype(str).tolist()
    labels = _label_reason_ai_batch(texts)
    fails["reason"] = labels

    counts = fails["reason"].value_counts()
    pareto_df, fig = _top80_pareto(counts, title="Fail reasons — Pareto (top 80%)")

    left, right = st.columns([1.15, 1.0], gap="large")
    with left:
        st.pyplot(fig, clear_figure=True)
    with right:
        tbl = pareto_df[["reason", "count", "percent", "cum_percent"]].copy()
        tbl["percent"] = tbl["percent"].round(1)
        tbl["cum_percent"] = tbl["cum_percent"].round(1)
        st.markdown(f"**Reason breakdown (top 80%) — {pd.to_datetime(last_ym).strftime('%b-%y')}**")
        st.dataframe(tbl, use_container_width=True, hide_index=True)
