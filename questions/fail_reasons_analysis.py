# questions/fail_reasons_analysis.py
from __future__ import annotations

import os
import re
from typing import Dict, Optional, Tuple, List

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# ------------- helpers: resilient access to FPA dataframe -----------------
def _first(df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    return df if isinstance(df, pd.DataFrame) and not df.empty else None

def _get_df_from_store(store: Dict) -> Optional[pd.DataFrame]:
    """
    Try common locations/shapes used by the app for FPA data.
    We avoid assumptions to keep Q1/Q2 isolated and unmodified.
    """
    # obvious keys
    for k in ("fpa", "fpa_df", "df_fpa", "first_pass_accuracy", "fpa_data"):
        if k in store and isinstance(store[k], pd.DataFrame):
            return _first(store[k])

    # nested bundle patterns (e.g., {'fpa': {'df': ...}})
    if "fpa" in store and isinstance(store["fpa"], dict):
        for k in ("df", "data"):
            df = store["fpa"].get(k)
            if isinstance(df, pd.DataFrame):
                return _first(df)

    # scan all values and pick the most plausible candidate
    for v in store.values():
        if isinstance(v, pd.DataFrame):
            cols = [c.lower() for c in v.columns]
            if any("comment" in c or "reason" in c for c in cols) and any(
                "pass" in c or "result" in c or "status" in c for c in cols
            ):
                return _first(v)

    return None

# ------------------- month parsing (from router hint) ---------------------
_MONTHS = "jan feb mar apr may jun jul aug sep oct nov dec".split()

def _to_month_key(text: Optional[str]) -> Optional[str]:
    if not text:
        return None
    t = str(text).lower()
    m = re.search(r"(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s*(\d{2,4})?", t)
    if not m:
        return None
    mon = m.group(1)
    yr = m.group(2)
    yr = f"20{yr}" if yr and len(yr) == 2 else yr
    if yr is None:
        # default to current year-like in your dataset (2025 is common in this app)
        yr = "2025"
    return f"{yr}-{_MONTHS.index(mon)+1:02d}"

def _pick_month(df: pd.DataFrame, hint: Optional[str]) -> Tuple[str, pd.DataFrame]:
    """
    Returns (YYYY-MM, filtered_df)
    If hint present try best-effort; else use most recent month in data.
    """
    # try to locate a date-like column
    date_col = None
    for c in df.columns:
        lc = str(c).lower()
        if lc in ("date", "created", "created_at", "received", "dt", "month"):
            date_col = c
            break
    # build a month key series
    if date_col is None:
        # if no date, don't filter; show 'Overall'
        return "Overall", df.copy()

    s = pd.to_datetime(df[date_col], errors="coerce")
    mk = s.dt.strftime("%Y-%m")
    df2 = df.copy()
    df2["__mk__"] = mk

    if hint:
        mk_hint = _to_month_key(hint)
        if mk_hint and mk_hint in df2["__mk__"].unique():
            return mk_hint, df2[df2["__mk__"] == mk_hint].copy()

    # pick most recent available
    valid = df2["__mk__"].dropna()
    if valid.empty:
        return "Overall", df2.drop(columns="__mk__", errors="ignore")
    last = valid.max()
    return last, df2[df2["__mk__"] == last].copy()

# ------------------------ OPENAI optional assist --------------------------
def _get_openai_client():
    """
    Returns (client, model) if available, else (None, None).
    We do *not* hard-require openai to keep Q1/Q2 safe.
    """
    key = None
    try:
        key = st.secrets.get("OPENAI_API_KEY")
    except Exception:
        pass
    if not key:
        key = os.getenv("OPENAI_API_KEY")
    if not key:
        return None, None

    try:
        from openai import OpenAI  # requires openai>=1.x in requirements (optional)
    except Exception:
        return None, None

    # allow overriding model via secrets/env; fallback to gpt-4o-mini
    model = (
        os.getenv("OPENAI_MODEL")
        or (st.secrets.get("OPENAI_MODEL") if hasattr(st, "secrets") else None)
        or (st.secrets.get("openai", {}).get("model") if hasattr(st, "secrets") and "openai" in st.secrets else None)
        or "gpt-4o-mini"
    )
    client = OpenAI(api_key=key)
    return client, model

def _llm_categorize(client, model: str, texts: List[str], taxonomy: List[str]) -> List[str]:
    """
    Batch label with OpenAI (very small context to control cost).
    If any LLM call fails we fall back to 'Other'.
    """
    if not texts:
        return []
    labels: List[str] = []
    cats = ", ".join(taxonomy)
    system = (
        "You are a classifier. Choose exactly one category from the list for each item: "
        f"{cats}. If nothing fits, answer 'Other'. Only output the category name."
    )
    for t in texts:
        try:
            msg = [
                {"role": "system", "content": system},
                {"role": "user", "content": f"Text: {t}"},
            ]
            r = client.chat.completions.create(model=model, messages=msg, temperature=0)
            lab = (r.choices[0].message.content or "").strip()
            labels.append(lab if lab in taxonomy else "Other")
        except Exception:
            labels.append("Other")
    return labels

# ------------------------ Keyword taxonomy (fallback) ----------------------
TAXONOMY = {
    "Communication / update": [
        "email", "chase", "follow up", "update", "call", "phone", "contact", "reminder",
        "awaiting response", "no reply", "communication", "advise", "inform"
    ],
    "Data entry / setup": [
        "data entry", "setup", "record", "wrong field", "typo", "mis-key", "input error",
        "address update", "national insurance", "dob", "ni number"
    ],
    "Postal / dispatch": [
        "post", "postal", "mail", "dispatched", "sent letter", "document sent",
        "returned mail", "undelivered", "post room", "courier"
    ],
    "Bank / payment": [
        "bank", "bacs", "payment", "cheque", "chq", "transfer", "refund", "sort code",
        "iban", "bsb", "swift"
    ],
    "Trustee / AVC": [
        "trustee", "tpa", "avc", "scheme", "administrator approval", "trust deed",
        "consent", "board approval"
    ],
    "Manual calculation": [
        "manual calc", "manual calculation", "spreadsheet", "calc check", "recalc",
        "hand calc", "formula check"
    ],
    "System": [
        "system", "portal", "workflow", "bug", "error code", "performance", "timeout",
        "access", "login", "permission"
    ],
    "Waiting on member/TPA": [
        "await", "waiting on member", "waiting on tpa", "await details", "await forms",
        "chasing member", "await evidence"
    ],
    "Document / ID": [
        "id", "identity", "passport", "driving licence", "proof of", "evidence",
        "birth certificate", "marriage certificate"
    ],
    "Address / contact": [
        "address", "postcode", "zip", "phone number", "mobile", "email address",
        "contact details"
    ],
    "Third-party dependency": [
        "employer", "hmrc", "bank error", "post office", "payroll", "insurer", "vendor"
    ],
    "Work allocation / queue": [
        "queue", "workstack", "allocation", "work load", "assigned", "reassign",
        "handover", "case owner"
    ],
}

KEYWORD_ORDER = list(TAXONOMY.keys())  # deterministic order

def _kw_label(text: str) -> str:
    t = str(text or "").lower()
    for cat in KEYWORD_ORDER:
        kws = TAXONOMY[cat]
        for kw in kws:
            # simple word-ish containment
            if re.search(rf"\b{re.escape(kw)}\b", t):
                return cat
    return "Other"

# ------------------------------ Pareto utils -------------------------------
def _pareto_top80(counts: pd.Series) -> pd.DataFrame:
    dfc = counts.reset_index()
    dfc.columns = ["reason", "count"]
    dfc = dfc.sort_values("count", ascending=False, ignore_index=True)
    dfc["percent"] = (dfc["count"] / dfc["count"].sum() * 100).round(1)
    dfc["cum_percent"] = dfc["percent"].cumsum().round(1)

    # keep rows up to 80%, group the rest as 'Other'
    mask = dfc["cum_percent"] <= 80
    kept = dfc[mask].copy()
    rest = dfc[~mask]
    if not rest.empty:
        other = pd.DataFrame(
            [{"reason": "Other", "count": int(rest["count"].sum())}]
        )
        other["percent"] = (other["count"] / dfc["count"].sum() * 100).round(1)
        other["cum_percent"] = 100.0
        kept = pd.concat([kept, other], ignore_index=True)
    else:
        # fix cum%
        kept["cum_percent"] = kept["percent"].cumsum().round(1)

    return kept

# ------------------------------ Main render --------------------------------
def run(store: Dict, params: Dict, user_text: Optional[str] = None):
    """
    Returns (title, subtitle?), dataframe
    """
    df = _get_df_from_store(store)
    if df is None or df.empty:
        return (
            "Reasons for Fail — No data",
            "Could not find a First-Pass Accuracy dataset in the store.",
        ), pd.DataFrame()

    # identify pass/fail column & comment column
    cols = {c.lower(): c for c in df.columns}
    comment_col = None
    for cand in ("comment", "comments", "case_comment", "case comments", "reason", "remarks", "notes"):
        if cand in cols:
            comment_col = cols[cand]
            break
    if comment_col is None:
        return ("Reasons for Fail — No comments", "No comment/reason text field found."), pd.DataFrame()

    # derive fail mask
    fail_mask = None
    if "passed" in cols:
        fail_mask = df[cols["passed"]].astype(int) == 0
    elif "pass" in cols:
        fail_mask = df[cols["pass"]].astype(int) == 0
    elif "result" in cols:
        fail_mask = df[cols["result"]].astype(str).str.lower().isin(["fail", "failed", "f", "0"])
    elif "status" in cols:
        fail_mask = df[cols["status"]].astype(str).str.lower().str.contains("fail")
    else:
        # best-effort: keep all rows (still useful when only comments exist)
        fail_mask = pd.Series(True, index=df.index)

    df_fail = df.loc[fail_mask].copy()
    if df_fail.empty:
        return ("Reasons for Fail — No failed rows", "No failures found to analyze."), pd.DataFrame()

    # pick month (router may have set 'hint_month')
    month_key, dfm = _pick_month(df_fail, params.get("hint_month"))

    # --- labeling (OpenAI optional; robust fallback to keywords) ---
    client, model = _get_openai_client()
    texts = dfm[comment_col].astype(str).fillna("").tolist()

    if client is not None:
        # only call LLM for rows that keyword model returns 'Other'
        kw_labels = [ _kw_label(t) for t in texts ]
        to_fix_idx = [i for i,l in enumerate(kw_labels) if l == "Other"]
        if to_fix_idx:
            llm_labels = _llm_categorize(client, model, [texts[i] for i in to_fix_idx], list(TAXONOMY.keys()))
            for j, idx in enumerate(to_fix_idx):
                kw_labels[idx] = llm_labels[j] if llm_labels[j] in TAXONOMY or llm_labels[j]=="Other" else "Other"
        labels = kw_labels
    else:
        labels = [ _kw_label(t) for t in texts ]

    dfm = dfm.assign(_reason_=labels)
    counts = dfm["_reason_"].value_counts(dropna=False)

    pareto = _pareto_top80(counts)

    # ----------------------------- render ---------------------------------
    title = f"Reasons for Fail — {month_key.replace('-', '–') if month_key!='Overall' else 'Overall'}"

    st.subheader(title)

    left, right = st.columns((3, 2))
    with left:
        fig, ax = plt.subplots(figsize=(7.2, 4.0))
        ax.bar(pareto["reason"], pareto["count"])
        # annotations
        for i, v in enumerate(pareto["count"]):
            ax.text(i, v, f"{int(v)}", ha="center", va="bottom", fontsize=9)
        # style: no gridlines, no y-axis
        ax.grid(False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.set_yticks([])
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(axis="x", labelrotation=90)
        st.pyplot(fig, clear_figure=True)

    with right:
        st.dataframe(
            pareto[["reason", "count", "percent", "cum_percent"]],
            use_container_width=True,
            hide_index=True,
        )

    subtitle = "Fail reasons — Pareto (top 80% + Other)"
    return (title, subtitle), pareto[["reason", "count", "percent", "cum_percent"]]
