# -*- coding: utf-8 -*-
# questions/first_pass_accuracy.py
from __future__ import annotations

import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt


# -----------------------------------------------------------------------------
# 0) OpenAI key bootstrap (safe, lazy; never prints the key)
# -----------------------------------------------------------------------------
def _get_openai_key() -> Optional[str]:
    """
    Find an OpenAI key in ENV or Streamlit secrets. If found in secrets,
    export it back to ENV so SDKs can pick it up.
    """
    key = os.environ.get("OPENAI_API_KEY")
    if key:
        return key
    try:
        key = st.secrets.get("OPENAI_API_KEY")  # type: ignore[attr-defined]
    except Exception:
        key = None
    if key:
        os.environ["OPENAI_API_KEY"] = key
    return key


def _openai_client_or_none():
    """
    Lazy-create an OpenAI client if key + SDK are present.
    Returns the client or None (and never raises).
    """
    key = _get_openai_key()
    if not key:
        return None
    try:
        from openai import OpenAI  # type: ignore
        return OpenAI(api_key=key)
    except Exception:
        return None


# -----------------------------------------------------------------------------
# 1) Load data (keeps your earlier behavior)
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def _find_fpa_workbook() -> str:
    # Prefer folder-based layout
    candidate_folder = "data/first_pass_accuracy"
    if os.path.isdir(candidate_folder):
        for f in sorted(os.listdir(candidate_folder)):
            if f.lower().endswith(".xlsx"):
                return os.path.join(candidate_folder, f)

    # Fallbacks for the apostrophe name etc.
    fallbacks = [
        "data/FirstPassAccuracy_Aug'25.xlsx",
        "data/FirstPassAccuracy_Aug25.xlsx",
    ]
    for p in fallbacks:
        if os.path.exists(p):
            return p

    raise FileNotFoundError("Could not find a FirstPassAccuracy workbook (*.xlsx).")


@st.cache_data(show_spinner=True)
def _load_fpa() -> pd.DataFrame:
    path = _find_fpa_workbook()
    df = pd.read_excel(path)

    # Flexible column mapping
    cols = {c.lower(): c for c in df.columns}

    def pick(*names: str) -> str:
        for n in names:
            if n.lower() in cols:
                return cols[n.lower()]
        raise KeyError(f"Missing column; expected one of: {names}")

    c_date = pick("Activity Date", "activity_date", "Date")
    c_res = pick("Review Result", "review_result", "Result")
    c_port = pick("Portfolio", "portfolio")
    c_scheme = pick("Scheme", "scheme")
    c_comment = pick("Case Comment", "case comment", "comment", "case_comment")

    df = df.rename(
        columns={
            c_date: "activity_date",
            c_res: "review_result",
            c_port: "portfolio",
            c_scheme: "scheme",
            c_comment: "comment",
        }
    )

    df["activity_date"] = pd.to_datetime(df["activity_date"], errors="coerce")
    df = df.dropna(subset=["activity_date"])
    df["review_result"] = df["review_result"].astype(str).str.strip().str.lower()
    df["is_pass"] = df["review_result"].str.contains("pass")
    df["month_key"] = df["activity_date"].dt.to_period("M").dt.to_timestamp()
    df["comment"] = df["comment"].astype(str).fillna("").str.strip()
    return df


# -----------------------------------------------------------------------------
# 2) KPI helpers (unchanged logic)
# -----------------------------------------------------------------------------
def _pass_rate_mom(df: pd.DataFrame) -> pd.DataFrame:
    start = pd.Timestamp("2025-01-01")
    end = df["month_key"].max()
    idx = pd.date_range(start, end, freq="MS")
    s = df.groupby("month_key")["is_pass"].mean().reindex(idx, fill_value=0.0)
    out = s.mul(100).rename("pass_pct").reset_index().rename(columns={"index": "month"})
    return out


def _pass_by_portfolio_scheme(df: pd.DataFrame, month: pd.Timestamp) -> pd.DataFrame:
    sub = df[df["month_key"] == month].copy()
    if sub.empty:
        return pd.DataFrame(columns=["portfolio", "scheme", "cases", "pass_%"])
    g = sub.groupby(["portfolio", "scheme"])["is_pass"].agg(["mean", "count"]).reset_index()
    g = g.rename(columns={"mean": "pass_%", "count": "cases"})
    g["pass_%"] = (g["pass_%"] * 100).round(1)
    return g.sort_values(["portfolio", "scheme"]).reset_index(drop=True)


# -----------------------------------------------------------------------------
# 3) Reason labelling
#     - OpenAI (if available)
#     - Fallback keyword model (your existing approach)
# -----------------------------------------------------------------------------
_CATEGORIES: List[str] = [
    "Communication / update",
    "Data entry / setup",
    "Document missing / incorrect",
    "Bank / payment",
    "Postal / dispatch",
    "Manual calculation / review",
    "Trustee / AVC",
    "Waiting on member / TPA",
    "System / portal",
    "Case not created / routing",
    "Death benefits / special cases",
    "Other",
]

_KEYWORD_MAP: Dict[str, List[str]] = {
    "Communication / update": [
        "email", "chased", "follow up", "update", "call back", "ring back",
        "correspondence", "awaiting response", "phone", "voicemail"
    ],
    "Data entry / setup": [
        "data entry", "setup", "keyed", "address change", "dob", "ni", "ni number",
        "input error", "typo", "incorrect details", "mis-key"
    ],
    "Document missing / incorrect": [
        "document", "doc", "form", "missing", "proof", "id", "evidence", "incorrect form",
        "invalid form", "photo id", "passport", "driving licence", "certificate"
    ],
    "Bank / payment": [
        "bank", "bacs", "payment", "payroll", "cheque", "refund", "transfer", "sort code",
        "account number", "standing order", "direct debit"
    ],
    "Postal / dispatch": ["post", "postal", "mail", "dispatch", "returned", "undelivered", "courier"],
    "Manual calculation / review": ["calculate", "calculation", "benefit calc", "reviewed", "rework", "qa", "quality", "recalc"],
    "Trustee / AVC": ["trustee", "avc", "additional voluntary", "trust", "board approval"],
    "Waiting on member / TPA": ["waiting on member", "waiting for member", "tpa", "third party", "employer", "provider", "external"],
    "System / portal": ["system", "portal", "it", "down", "bug", "error", "service now", "servicenow"],
    "Case not created / routing": ["case not created", "routing", "queue", "workbasket", "not allocated"],
    "Death benefits / special cases": ["death", "bereavement", "executor", "probate", "special", "exception"],
    "Other": [],
}


def _keyword_label_one(text: str) -> str:
    t = text.lower()
    for cat, kws in _KEYWORD_MAP.items():
        if cat == "Other":
            continue
        for kw in kws:
            if kw in t:
                return cat
    return "Other"


def _openai_label_many(texts: List[str]) -> List[str]:
    client = _openai_client_or_none()
    if client is None:
        return ["Other"] * len(texts)

    system = (
        "You are a classification helper for service operations.\n"
        "Given a short case comment, assign exactly one label from this list:\n"
        f"{', '.join(_CATEGORIES)}.\n"
        "Return ONLY the label text. If unsure, return 'Other'."
    )
    out: List[str] = []
    for t in texts:
        try:
            resp = client.chat.completions.create(
                model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
                temperature=0,
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": f"Comment: {t[:2000]}"}],
            )
            label = (resp.choices[0].message.content or "").strip()
            if label not in _CATEGORIES:
                label = "Other"
        except Exception:
            label = "Other"
        out.append(label)
    return out


def _label_reasons(df_fail: pd.DataFrame) -> pd.DataFrame:
    comments = df_fail["comment"].fillna("").astype(str).tolist()

    # Prefer OpenAI if ready; else keyword
    labels = _openai_label_many(comments)
    if all(l == "Other" for l in labels):
        labels = [_keyword_label_one(t) for t in comments]

    ser = pd.Series(labels, name="reason")
    tbl = ser.value_counts(dropna=False).rename_axis("reason").reset_index(name="count")
    tbl = tbl.sort_values("count", ascending=False).reset_index(drop=True)

    # Pareto = top 80% + collapse the remainder into 'Other'
    tbl["percent"] = (tbl["count"] / max(1, tbl["count"].sum()) * 100).round(1)
    cutoff = tbl["percent"].cumsum() <= 80
    top = tbl[cutoff].copy()
    rem = tbl[~cutoff].copy()
    if not rem.empty:
        other_count = rem["count"].sum()
        # merge with existing 'Other' if present in 'top'
        if (top["reason"] == "Other").any():
            top.loc[top["reason"] == "Other", "count"] += other_count
        else:
            top = pd.concat([top, pd.DataFrame([{"reason": "Other", "count": int(other_count)}])],
                            ignore_index=True)
    tbl = top.copy()
    total = tbl["count"].sum()
    tbl["percent"] = (tbl["count"] / max(1, total) * 100).round(1)
    tbl["cum_percent"] = tbl["percent"].cumsum().round(1)
    return tbl


# -----------------------------------------------------------------------------
# 4) Simple Pareto bar
# -----------------------------------------------------------------------------
def _plot_pareto(df_counts: pd.DataFrame, title: str):
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    ax.bar(df_counts["reason"], df_counts["count"])
    for i, v in enumerate(df_counts["count"].tolist()):
        ax.text(i, v, str(v), ha="center", va="bottom", fontsize=9)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.grid(False)
    ax.set_ylabel("")
    ax.tick_params(axis="y", length=0, labelleft=False)
    ax.set_title(title, loc="left", fontsize=16, fontweight="bold", color="#1e3a8a")
    ax.set_xticklabels(df_counts["reason"].tolist(), rotation=90)
    st.pyplot(fig, use_container_width=True)


# -----------------------------------------------------------------------------
# 5) ENTRY POINT (signature unchanged)
# -----------------------------------------------------------------------------
def run(store: Dict, params: Dict, q: str):
    # Load
    df = _load_fpa()
    latest_month = df["month_key"].max()

    # Header
    st.subheader(f"First-Pass Accuracy — Jan–{latest_month.strftime('%b %y')}")

    # MoM + table
    left, right = st.columns([1.1, 1])
    with left:
        st.markdown("##### Pass % — MoM")
        mom = _pass_rate_mom(df)
        fig, ax = plt.subplots(figsize=(8.5, 2.6))
        ax.plot(mom["month"], mom["pass_pct"], marker="o")
        for x, y in zip(mom["month"], mom["pass_pct"]):
            ax.text(x, y, f"{int(round(y))}%", ha="center", va="bottom", fontsize=9)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.grid(False)
        ax.set_ylabel("")
        ax.tick_params(axis="y", length=0, labelleft=False)
        ax.set_xticks(mom["month"])
        ax.set_xticklabels([d.strftime("%Y-%m") for d in mom["month"]], rotation=0)
        st.pyplot(fig, use_container_width=True)

    with right:
        st.markdown(f"##### Pass % by Portfolio × Scheme — {latest_month.strftime('%b-%y')}")
        st.dataframe(_pass_by_portfolio_scheme(df, latest_month), use_container_width=True)

    # Reasons (Pareto)
    st.markdown(f"### Reasons for Fail — {latest_month.strftime('%b-%y')}")
    st.markdown("##### Fail reasons — Pareto (top 80% + Other)")

    df_latest = df[(df["month_key"] == latest_month) & (~df["is_pass"])].copy()
    if df_latest.empty:
        st.info("No failed cases for the selected month.")
        return

    pareto = _label_reasons(df_latest)
    c1, c2 = st.columns([1.1, 1])
    with c1:
        _plot_pareto(pareto, "Fail reasons — Pareto (top 80% + Other)")
    with c2:
        st.dataframe(pareto, use_container_width=True)
