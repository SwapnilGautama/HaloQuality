# questions/first_pass_accuracy.py
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# ========= Helpers =========
ROOT = Path(__file__).resolve().parents[1]  # repo root
DATA_DIR = ROOT / "data" / "first_pass_accuracy"

# Try OpenAI; stay resilient if not available
OPENAI_READY = False
def _call_openai_batch(texts: List[str], labels: List[str]) -> List[str]:
    """
    Classify 'texts' into one of 'labels'. Returns a list of chosen labels (len == len(texts)).
    If no OpenAI credentials, we fall back to a regex/keyword model below.
    """
    global OPENAI_READY
    if not OPENAI_READY:
        return _regex_label(texts, labels)

    # Build a robust client (supports both old & new SDKs)
    try:
        from openai import OpenAI  # new SDK
        client = OpenAI()
        def complete(batch: List[str]) -> str:
            sys = (
                "You are a data labeler. For each input, choose exactly ONE label "
                f"from this list: {labels}. If nothing fits, return 'Other'. "
                "Return a pure JSON list of labels, e.g. [\"Delay\", \"System\"]."
            )
            user = "Classify these lines:\n" + json.dumps(batch, ensure_ascii=False)
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role":"system","content":sys},{"role":"user","content":user}],
                response_format={"type":"json_object"}
            )
            # new SDK returns .choices[0].message.content
            content = resp.choices[0].message.content
            # expected as {"labels": ["...", "..."]} or just the list; try both
            try:
                obj = json.loads(content)
                if isinstance(obj, dict) and "labels" in obj:
                    return obj["labels"]
                return obj  # already a list
            except Exception:
                # Try to salvage list-looking content
                start = content.find("[")
                end = content.rfind("]")
                obj = json.loads(content[start:end+1])
                return obj
        _ = client  # lint
    except Exception:
        # try old SDK
        try:
            import openai  # type: ignore
            openai.api_key = os.getenv("OPENAI_API_KEY", "")
            if not openai.api_key:
                return _regex_label(texts, labels)
            OPENAI_READY = True
            def complete(batch: List[str]) -> List[str]:
                sys = (
                    "You are a data labeler. For each input, choose exactly ONE label "
                    f"from this list: {labels}. If nothing fits, return 'Other'. "
                    "Return a pure JSON list of labels, e.g. [\"Delay\", \"System\"]."
                )
                user = "Classify these lines:\n" + json.dumps(batch, ensure_ascii=False)
                resp = openai.ChatCompletion.create(
                    model="gpt-4o-mini",
                    messages=[{"role":"system","content":sys},{"role":"user","content":user}],
                    temperature=0,
                )
                content = resp["choices"][0]["message"]["content"]
                try:
                    obj = json.loads(content)
                    if isinstance(obj, dict) and "labels" in obj:
                        return obj["labels"]
                    return obj
                except Exception:
                    start = content.find("[")
                    end = content.rfind("]")
                    obj = json.loads(content[start:end+1])
                    return obj
        except Exception:
            return _regex_label(texts, labels)

    # If we got here we have a working client
    OPENAI_READY = True
    out: List[str] = []
    BATCH = 80  # small batches to keep things cheap & robust
    for i in range(0, len(texts), BATCH):
        chunk = texts[i:i+BATCH]
        try:
            labels_chunk = complete(chunk)
        except Exception:
            labels_chunk = _regex_label(chunk, labels)
        # sanitize lengths
        if not isinstance(labels_chunk, list) or len(labels_chunk) != len(chunk):
            labels_chunk = _regex_label(chunk, labels)
        out.extend(labels_chunk)
    return out


def _regex_label(texts: List[str], labels: List[str]) -> List[str]:
    """
    Zero-cost fallback: fast keyword mapping.
    We trim 'Other' by being a bit more aggressive than before.
    """
    import re

    # Taxonomy and keyword sets (tweakable without touching Q1)
    buckets = {
        "Bank / payment": r"(?i)\b(bank|bacs|cheque|payment|refund|payee|account|sort code|iban)\b",
        "Communication / update": r"(?i)\b(update|contact|phone|call|email|letter|communication|chasing|remind)\b",
        "Data entry / setup": r"(?i)\b(form|data entry|setup|capture|record|mis-?key|wrong field|typo|input)\b",
        "Postal / dispatch": r"(?i)\b(post|mail|dispatch|send|delivered|courier|royal mail|address)\b",
        "Manual calculation": r"(?i)\b(manual calc|calc error|recalc|re-calculation|spreadsheet)\b",
        "System": r"(?i)\b(system|portal|technical|bug|down|crash|timeout|access)\b",
        "Trustee / AVC": r"(?i)\b(trustee|avc|additional voluntary|governance|approval)\b",
        "Waiting on member/TPA": r"(?i)\b(wait|await|pending|member to|tp[ap]|3rd party|third party)\b",
        "Scheme rules / procedure": r"(?i)\b(rule|policy|procedure|timescale|sla|scheme)\b",
    }
    compiled = {k: re.compile(v) for k, v in buckets.items()}
    labset = set(labels)

    outs: List[str] = []
    for t in texts:
        t0 = (t or "").strip()
        if not t0:
            outs.append("Other")
            continue
        assigned = None
        for name, rx in compiled.items():
            if rx.search(t0):
                assigned = name if name in labset else "Other"
                break
        outs.append(assigned or "Other")
    return outs


@st.cache_data(show_spinner=False)
def _load_fpa_excel() -> pd.DataFrame:
    # look for *any* file that starts with FirstPassAccuracy_
    candidates = sorted(list(DATA_DIR.glob("FirstPassAccuracy_*.xlsx")))
    if not candidates:
        raise FileNotFoundError("Could not find a FirstPassAccuracy workbook (FirstPassAccuracy_*.xlsx).")
    df = pd.read_excel(candidates[-1])  # latest file
    return df


def _prep(df: pd.DataFrame) -> pd.DataFrame:
    # expected columns: Activity Date, Review Result, Portfolio, Scheme, Case Comment (or similar)
    cols = {c.lower(): c for c in df.columns}
    def _find(*keys):
        for k in keys:
            if k.lower() in cols: return cols[k.lower()]
        return None

    activity = _find("Activity Date")
    result = _find("Review Result")
    comment = _find("Case Comment")
    portfolio = _find("Portfolio")
    scheme = _find("Scheme")

    if not all([activity, result, portfolio, scheme]):
        raise KeyError("Required columns missing (Activity Date, Review Result, Portfolio, Scheme).")

    out = pd.DataFrame({
        "date": pd.to_datetime(df[activity], errors="coerce"),
        "result": df[result].astype(str).str.strip(),
        "portfolio": df[portfolio].astype(str).str.strip(),
        "scheme": df[scheme].astype(str).str.strip(),
        "comment": df[comment].astype(str).fillna("") if comment else "",
    })
    out["month"] = out["date"].dt.to_period("M").astype(str)
    out["is_pass"] = out["result"].str.contains("pass", case=False, na=False)
    return out


def _mom_pass(df: pd.DataFrame) -> pd.DataFrame:
    # Jan-2025 to most recent month, 0 when absent
    if df["date"].min() is pd.NaT:
        return pd.DataFrame(columns=["month", "pass_%"])
    start = pd.Timestamp("2025-01-01")
    end = (df["date"].max() or pd.Timestamp.today()).to_period("M").to_timestamp()
    months = pd.period_range(start, end, freq="M").astype(str)

    s = df.groupby("month")["is_pass"].mean().reindex(months).fillna(0.0) * 100
    return pd.DataFrame({"month": months, "pass_%": s.round(1).values})


def _latest_pass_by_portfolio_scheme(df: pd.DataFrame) -> Tuple[str, pd.DataFrame]:
    latest = df["date"].max()
    if pd.isna(latest):
        return "", pd.DataFrame(columns=["portfolio", "scheme", "cases", "pass_%"])
    ymon = latest.strftime("%b-%y")
    dfl = df[df["date"].dt.to_period("M") == latest.to_period("M")].copy()
    grp = dfl.groupby(["portfolio", "scheme"])["is_pass"].mean().mul(100).round(1).reset_index(name="pass_%")
    grp["cases"] = dfl.groupby(["portfolio","scheme"])["is_pass"].size().values
    grp = grp[["portfolio","scheme","cases","pass_%"]].sort_values(["portfolio","scheme"])
    return ymon, grp


def _classify_reasons(df: pd.DataFrame) -> Tuple[str, pd.DataFrame, pd.DataFrame]:
    """
    On the latest month, label FAIL comments into taxonomy with OpenAI assist.
    Returns (label_month, counts_df, pareto_df[top80 + Other])
    """
    latest = df["date"].max()
    if pd.isna(latest):
        return "", pd.DataFrame(columns=["reason","count","percent","cum_percent"]), pd.DataFrame()

    month_key = latest.to_period("M")
    dfl = df[df["date"].dt.to_period("M") == month_key]
    fails = dfl.loc[~dfl["is_pass"]].copy()

    labels = [
        "Bank / payment",
        "Communication / update",
        "Data entry / setup",
        "Postal / dispatch",
        "Manual calculation",
        "System",
        "Trustee / AVC",
        "Waiting on member/TPA",
        "Scheme rules / procedure",
        "Other",
    ]

    # Try OpenAI; cache by hash of the month’s comments (so it’s fast on re-runs)
    txts = fails["comment"].astype(str).fillna("").tolist()
    digest = hashlib.md5(("||".join(txts)).encode("utf-8")).hexdigest()
    @st.cache_data(show_spinner=False)
    def _do_label(_digest: str, _texts: List[str], _labels: List[str]) -> List[str]:
        # First attempt OpenAI; if not available, we’ll drop to regex
        key = os.getenv("OPENAI_API_KEY", "")
        global OPENAI_READY
        OPENAI_READY = bool(key)
        return _call_openai_batch(_texts, _labels)

    y = _do_label(digest, txts, labels)
    fails["reason"] = pd.Series(y, index=fails.index).astype("category")

    counts = (
        fails["reason"].value_counts(dropna=False)
        .rename_axis("reason")
        .reset_index(name="count")
    )

    total = counts["count"].sum()
    counts["percent"] = (counts["count"] / max(total, 1) * 100).round(1)

    # Pareto top-80 + "Other"
    counts = counts.sort_values("count", ascending=False)
    counts["cum_percent"] = counts["percent"].cumsum().round(1)
    top = counts.copy()
    # collapse to 80% + Other
    mask_top = counts["cum_percent"] <= 80.0
    top_part = counts[mask_top]
    other_count = counts.loc[~mask_top, "count"].sum()
    other_pct = counts.loc[~mask_top, "percent"].sum().round(1)
    if other_count:
        top_part = pd.concat([
            top_part,
            pd.DataFrame([{"reason":"Other","count":int(other_count),"percent":float(other_pct)}])
        ], ignore_index=True)
    top_part["cum_percent"] = top_part["percent"].cumsum().round(1)

    label_month = latest.strftime("%b-%y")
    return label_month, counts, top_part


# ============================== RENDER ==============================
def run(store: Dict, params: Dict, user_text: str | None = None):
    st.subheader("First-Pass Accuracy — Jan–Most Recent")

    df_raw = _load_fpa_excel()
    df = _prep(df_raw)

    # Pass % — MoM
    mom = _mom_pass(df)
    col1, col2 = st.columns([1.2, 1.2])

    with col1:
        st.markdown("**Pass % — MoM**")
        fig, ax = plt.subplots(figsize=(6.5, 2.6))
        ax.plot(mom["month"], mom["pass_%"], marker="o", linewidth=2)
        # styling: no gridlines & hide y-axis
        ax.grid(False)
        ax.set_ylim(0, max(100, mom["pass_%"].max() + 5))
        ax.set_ylabel("")
        ax.set_xlabel("")
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.spines["left"].set_visible(False)
        for label in ax.get_xticklabels():
            label.set_rotation(0)
        st.pyplot(fig, use_container_width=True)

    # Pass % by Portfolio × Scheme (latest month)
    with col2:
        latest_label, by_ps = _latest_pass_by_portfolio_scheme(df)
        st.markdown(f"**Pass % by Portfolio × Scheme — {latest_label}**")
        st.dataframe(by_ps, use_container_width=True, height=330)

    # Reasons for fail (latest)
    label_month, counts, pareto = _classify_reasons(df)
    st.markdown(f"### Reasons for Fail — {label_month}")

    c3, c4 = st.columns([1.2, 1.2])
    with c3:
        st.markdown("**Fail reasons — Pareto (top 80% + Other)**")
        if not pareto.empty:
            fig2, ax2 = plt.subplots(figsize=(6.5, 3.0))
            bars = ax2.bar(pareto["reason"], pareto["count"])
            ax2.grid(False)
            ax2.set_ylabel("")
            ax2.set_xlabel("")
            ax2.spines["right"].set_visible(False)
            ax2.spines["top"].set_visible(False)
            ax2.spines["left"].set_visible(False)
            for b in bars:
                ax2.bar_label([b], labels=[f"{int(b.get_height())}"], padding=3)
            plt.xticks(rotation=90)
            st.pyplot(fig2, use_container_width=True)
        else:
            st.info("No fails in the latest month.")

    with c4:
        if not pareto.empty:
            st.dataframe(
                pareto[["reason","count","percent","cum_percent"]],
                use_container_width=True, height=300
            )
        else:
            st.empty()

    return ("",), pd.DataFrame()
