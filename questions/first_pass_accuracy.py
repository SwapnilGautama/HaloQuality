# -*- coding: utf-8 -*-
# questions/first_pass_accuracy.py
from __future__ import annotations

import os
import re
from io import BytesIO
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# 0) OpenAI key bootstrap (safe, preserves existing UX)
# -----------------------------------------------------------------------------
def _get_openai_key() -> str | None:
    """
    Look for an OpenAI API key and make it available to downstream libs.
    Order:
      1) Environment (OPENAI_API_KEY)
      2) Streamlit Secrets (OPENAI_API_KEY), then export back to env
    """
    k = os.environ.get("OPENAI_API_KEY")
    if k:
        return k
    try:
        k = st.secrets.get("OPENAI_API_KEY")
    except Exception:
        k = None
    if k:
        os.environ["OPENAI_API_KEY"] = k
    return k


OPENAI_API_KEY = _get_openai_key()

# Best-effort import of the OpenAI SDK (do NOT break if missing)
_openai_available = False
_openai_client = None
if OPENAI_API_KEY:
    try:
        from openai import OpenAI  # type: ignore
        _openai_client = OpenAI(api_key=OPENAI_API_KEY)
        _openai_available = True
    except Exception:
        _openai_available = False


# -----------------------------------------------------------------------------
# 1) Data loading helpers (robust to file name changes)
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def _find_fpa_workbook() -> str:
    """
    Return the first workbook path under data/first_pass_accuracy that looks like FPA.
    We keep this tolerant to naming like "FirstPassAccuracy_Aug'25.xlsx".
    """
    root = "data/first_pass_accuracy"
    if not os.path.isdir(root):
        # fallbacks for older layouts
        candidates = [
            "data/FirstPassAccuracy_Aug'25.xlsx",
            "data/FirstPassAccuracy_Aug25.xlsx",
        ]
        for c in candidates:
            if os.path.exists(c):
                return c
        raise FileNotFoundError("FPA workbook folder not found.")
    # pick first xlsx
    for f in sorted(os.listdir(root)):
        if f.lower().endswith(".xlsx"):
            return os.path.join(root, f)
    raise FileNotFoundError("Could not find a First Pass Accuracy workbook (*.xlsx).")


@st.cache_data(show_spinner=True)
def _load_fpa() -> pd.DataFrame:
    """
    Load and lightly normalize your FPA workbook.
    Expected columns (case-insensitive):
      - Activity Date
      - Review Result (Pass/Fail)
      - Portfolio
      - Scheme
      - Case Comment  (free text used for reason classification)
    """
    path = _find_fpa_workbook()
    df = pd.read_excel(path)
    cols = {c.lower(): c for c in df.columns}

    # Map likely names to normalized ones
    def pick(*opts: str) -> str:
        for o in opts:
            if o.lower() in cols:
                return cols[o.lower()]
        raise KeyError(f"Missing columns: one of {opts}")

    c_date = pick("Activity Date", "activity_date", "Date")
    c_res = pick("Review Result", "review_result", "Result")
    c_port = pick("Portfolio", "portfolio")
    c_scheme = pick("Scheme", "scheme")
    c_comment = pick("Case Comment", "case comment", "Comment", "case_comment")

    df = df.rename(
        columns={
            c_date: "activity_date",
            c_res: "review_result",
            c_port: "portfolio",
            c_scheme: "scheme",
            c_comment: "comment",
        }
    )

    # Coerce date
    df["activity_date"] = pd.to_datetime(df["activity_date"], errors="coerce")
    df = df.dropna(subset=["activity_date"])

    # Normalize result
    df["review_result"] = df["review_result"].astype(str).str.strip().str.lower()
    df["is_pass"] = df["review_result"].str.contains("pass")

    # Month key
    df["month_key"] = df["activity_date"].dt.to_period("M").dt.to_timestamp()

    # Clean text
    df["comment"] = df["comment"].astype(str).fillna("").str.strip()

    return df


# -----------------------------------------------------------------------------
# 2) KPI/visual helpers you already had (unchanged in spirit)
# -----------------------------------------------------------------------------
def _pass_rate_mom(df: pd.DataFrame) -> pd.DataFrame:
    """Pass % MoM from Jan-2025 to latest; missing months = 0%."""
    start = pd.Timestamp("2025-01-01")
    end = df["month_key"].max()
    timeline = pd.date_range(start, end, freq="MS")
    agg = (
        df.groupby("month_key")["is_pass"]
        .mean()
        .reindex(timeline, fill_value=0.0)
        .mul(100)
        .rename("pass_pct")
        .reset_index()
        .rename(columns={"index": "month"})
    )
    return agg


def _pass_by_portfolio_scheme(df: pd.DataFrame, for_month: pd.Timestamp) -> pd.DataFrame:
    """Pass % by portfolio×scheme for a given month."""
    sub = df[df["month_key"] == for_month].copy()
    if sub.empty:
        return pd.DataFrame(columns=["portfolio", "scheme", "cases", "pass_%"])
    g = sub.groupby(["portfolio", "scheme"])["is_pass"].agg(["mean", "count"]).reset_index()
    g = g.rename(columns={"mean": "pass_%", "count": "cases"})
    g["pass_%"] = (g["pass_%"] * 100).round(1)
    g = g.sort_values(["portfolio", "scheme"]).reset_index(drop=True)
    return g


# -----------------------------------------------------------------------------
# 3) Reason labelling
#     - OpenAI (if key+sdk available)
#     - Fallback: keyword model (your previous approach)
# -----------------------------------------------------------------------------
_CATEGORIES = [
    # Broader buckets to make the output more actionable
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
    "Postal / dispatch": [
        "post", "postal", "mail", "dispatch", "sent", "returned", "undelivered", "courier"
    ],
    "Manual calculation / review": [
        "calculate", "calculation", "benefit calc", "reviewed", "checked", "rework",
        "qa", "quality", "recalc"
    ],
    "Trustee / AVC": [
        "trustee", "avc", "additional voluntary", "trust", "board approval"
    ],
    "Waiting on member / TPA": [
        "waiting on member", "waiting for member", "tpa", "third party", "employer",
        "provider", "external"
    ],
    "System / portal": [
        "system", "portal", "it", "down", "bug", "error", "service now", "servicenow"
    ],
    "Case not created / routing": [
        "case not created", "routing", "queue", "workbasket", "not allocated"
    ],
    "Death benefits / special cases": [
        "death", "bereavement", "executor", "probate", "special", "exception"
    ],
    "Other": []
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
    """
    Label a list of texts using OpenAI. Keeps tokens small, deterministic-ish.
    We return categories defined in _CATEGORIES; unknown => 'Other'.
    """
    assert _openai_client is not None
    system = (
        "You are a classification helper for service operations.\n"
        "Given a short case comment, assign exactly one label from this list:\n"
        f"{', '.join(_CATEGORIES)}.\n"
        "Return ONLY the label text. If unsure, return 'Other'."
    )
    out: List[str] = []
    for t in texts:
        t = t or ""
        try:
            resp = _openai_client.chat.completions.create(
                model="gpt-4o-mini",
                temperature=0,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": f"Comment: {t[:2000]}"},
                ],
            )
            label = (resp.choices[0].message.content or "").strip()
            if label not in _CATEGORIES:
                label = "Other"
        except Exception:
            label = "Other"
        out.append(label)
    return out


def _label_reasons(df_fail: pd.DataFrame) -> pd.DataFrame:
    """
    Produce a dataframe of fail reasons with counts and cumulative %, and a Pareto
    subset that includes top 80% + 'Other'.
    """
    comments = df_fail["comment"].fillna("").astype(str).tolist()

    if _openai_available:
        labels = _openai_label_many(comments)
    else:
        labels = [ _keyword_label_one(t) for t in comments ]

    ser = pd.Series(labels, name="reason")
    tbl = ser.value_counts(dropna=False).rename_axis("reason").reset_index(name="count")
    tbl["percent"] = (tbl["count"] / max(1, tbl["count"].sum()) * 100).round(1)
    tbl["cum_percent"] = tbl["percent"].cumsum().round(1)

    # Pareto (top 80% + Other)
    tbl = tbl.sort_values("count", ascending=False).reset_index(drop=True)

    # Identify top-80 band; always include 'Other' as a line item
    cutoff = tbl["percent"].cumsum() <= 80
    top = tbl[cutoff].copy()
    others = tbl[~cutoff].copy()
    if not others.empty:
        other_row = others[others["reason"] == "Other"]
        non_other = others[others["reason"] != "Other"]
        if not other_row.empty:
            # keep the existing 'Other' row but add the rest into it
            extra = non_other["count"].sum()
            tbl.loc[other_row.index, "count"] = other_row["count"].iloc[0] + extra
            tbl = pd.concat([top, tbl.loc[other_row.index]]).reset_index(drop=True)
        else:
            # create Other row from the remainder
            new_other = pd.DataFrame([{
                "reason": "Other",
                "count": int(non_other["count"].sum()),
            }])
            tbl = pd.concat([top, new_other], ignore_index=True)
    else:
        tbl = top.copy()

    # Recompute percent and cumulative
    total = tbl["count"].sum()
    tbl["percent"] = (tbl["count"] / max(1, total) * 100).round(1)
    tbl["cum_percent"] = tbl["percent"].cumsum().round(1)

    return tbl


# -----------------------------------------------------------------------------
# 4) Simple Pareto bar (keeps your visual language)
# -----------------------------------------------------------------------------
def _plot_pareto(df_counts: pd.DataFrame, title: str):
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    ax.bar(df_counts["reason"], df_counts["count"])
    for i, v in enumerate(df_counts["count"].tolist()):
        ax.text(i, v, str(v), ha="center", va="bottom", fontsize=9)
    # Style: keep x baseline, remove y axis & grids per your preference
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.grid(False)
    ax.set_ylabel("")  # remove y label
    ax.tick_params(axis="y", length=0, labelleft=False)
    ax.set_title(title, loc="left", fontsize=16, fontweight="bold", color="#1e3a8a")
    ax.set_xticklabels(df_counts["reason"].tolist(), rotation=90)
    st.pyplot(fig, use_container_width=True)


# -----------------------------------------------------------------------------
# 5) Entry point used by the app
# -----------------------------------------------------------------------------
def run(store: Dict, params: Dict, q: str):
    # Guard: data root expected by app integration (do not remove)
    # (Keeping this no-op reference to 'store' so we don't break the call signature)
    _ = store

    # Diagnostic caption (safe to keep; never prints secrets)
    try:
        installed = {"openai" in {m.name for m in __import__("pkgutil").iter_modules()}}
        st.caption(
            "Diag · openai sdk installed: "
            f"{'openai' in {m.name for m in __import__('pkgutil').iter_modules()}} · "
            f"env_has_key: {bool(os.environ.get('OPENAI_API_KEY'))} · "
            f"secrets_has_key: {'OPENAI_API_KEY' in getattr(st, 'secrets', {})}"
        )
    except Exception:
        pass

    # Info banner when OpenAI is inactive (keeps your existing UX)
    if not _openai_available:
        st.info(
            "OpenAI labelling inactive (no OPENAI_API_KEY). "
            "Fail reasons will default to ‘Other’ or keyword labels."
        )

    # Load data
    df = _load_fpa()
    latest_month = df["month_key"].max()

    # ---- Header
    st.subheader(f"First-Pass Accuracy — Jan–{latest_month.strftime('%b %y')}")
    # ---- Pass % MoM
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
        tbl = _pass_by_portfolio_scheme(df, latest_month)
        st.dataframe(tbl, use_container_width=True)

    # ---- Reasons for Fail (Pareto)
    st.markdown(f"### Reasons for Fail — {latest_month.strftime('%b-%y')}")
    st.markdown("##### Fail reasons — Pareto (top 80% + Other)")

    df_latest = df[(df["month_key"] == latest_month) & (~df["is_pass"])].copy()
    if df_latest.empty:
        st.info("No failed cases for the selected month.")
        return

    pareto = _label_reasons(df_latest)
    # Chart + table side-by-side
    c1, c2 = st.columns([1.1, 1])
    with c1:
        _plot_pareto(pareto, "Fail reasons — Pareto (top 80% + Other)")
    with c2:
        st.dataframe(pareto, use_container_width=True)
