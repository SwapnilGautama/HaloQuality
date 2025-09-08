# questions/first_pass_accuracy.py
from __future__ import annotations

import os
import re
import json
from pathlib import Path
from typing import Dict, Tuple, Optional, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

_DARK_BLUE = "#0b3d91"
_DARK_GREY = "#333333"
_SOFT_GREY = "#DDDDDD"

# --- Optional OpenAI assist (kept OFF by default; safe fallback to rules) ---
_OPENAI = False
try:
    import openai  # type: ignore
    if os.getenv("OPENAI_API_KEY"):
        openai.api_key = os.getenv("OPENAI_API_KEY")
        _OPENAI = True
except Exception:
    _OPENAI = False


# ---------------------------
# Data loading (unchanged)
# ---------------------------
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


# ---------------------------
# Column helpers (unchanged)
# ---------------------------
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
    }
    missing = [k for k, v in col_map.items() if k in ("date", "result") and v is None]
    if missing:
        raise KeyError(f"Missing required columns for FPA: {missing}")
    return df.rename(columns={v: k for k, v in col_map.items() if v}), col_map


# ---------------------------
# Pass% + table logic (unchanged)
# ---------------------------
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

def _table_portfolio_scheme(df: pd.DataFrame, last_m: pd.Period) -> pd.DataFrame:
    df = df.assign(_m=_coerce_month(df["date"]))
    sub = df[df["_m"] == last_m]
    if sub.empty:
        return pd.DataFrame(columns=["portfolio", "scheme", "cases", "pass_%"])
    grp = sub.groupby(["portfolio", "scheme"])["result"].agg(
        cases="count", passed=lambda x: np.sum([_is_pass(v) for v in x])
    ).reset_index()
    grp["pass_%"] = (grp["passed"] * 100.0 / grp["cases"]).round(0)
    return grp[["portfolio", "scheme", "cases", "pass_%"]].sort_values(
        ["portfolio", "pass_%", "scheme"], ascending=[True, False, True]
    )


# ---------------------------
# Fail reason classification (IMPROVED, *surgical change*)
# ---------------------------

# 1) Primary rulebook — ordered, earliest match wins (expanded)
_RULES = {
    "Bank / payment": [
        r"\b(bank|payment|refund|bacs|chaps|cheque|sort\s*code|iban|bic|account|transfer|credit|debit)\b",
        r"\bpaid\s*to\s*wrong|duplicate\s*payment|missing\s*payment\b",
        r"\bbank\s*(detail|form|mandate)\b",
    ],
    "Communication / update": [
        r"\b(no|missing)\s*(reply|response|update)\b",
        r"\bupdate|communicat|clarif|explain|advise|inform(ed|ation)?\b",
        r"\bconfus|unclear|mis(lead|understand)\b",
        r"\bcall(s|ed)?|email(s|ed)?|letter(s)?\b",
        r"\bholding\s*letter|chaser|timescale(s)?\b",
    ],
    "Data entry / setup": [
        r"\bwrong|incorrect|mis-?key|typo|misallocat|miscode|set\s*up|setup\b",
        r"\bdata\s*(entry|load|issue)|capture|key(ed|ing)\b",
        r"\bdate\s*error|dob|ni\s*number|nino\b",
        r"\baddress\s*(change|update)|postcode\b",
        r"\bid|identity|poa|proof\s*of\s*address\b",
    ],
    "Postal / dispatch": [
        r"\b(post|mail|postal|dispatch|despatch|send|sent|deliver(y|ed)?)\b",
        r"\breturned\s*mail|wrong\s*address\b",
        r"\bpack|document(s)?\s*(sent|received|missing)\b",
    ],
    "Manual calculation": [
        r"\bmanual\b.*calc|re-?calc|recalculation|calc(ulation)?\s*error\b",
        r"\bquote|estimate|CETV|cash\s*equivalent\b",
    ],
    "Waiting on member/TPA": [
        r"\bawait|waiting\s*for|chase(d|s|ing)?\b",
        r"\b(3rd|third)\s*party|tpa|ifa|insurer|administrator|provider|employer|payroll|trustee\b",
        r"\bmember\s*to\s*(respond|confirm|provide)\b",
    ],
    "Trustee / AVC": [
        r"\btrustee|avc|additional\s*voluntary\s*contribution\b",
    ],
    "System / workflow": [
        r"\bsystem|portal|platform|workflow|work\s*queue|technical|bug|defect|automation|script\b",
        r"\baccess|permission|role|profile\b",
        r"\bdowntime|outage|crash|error\s*code\b",
    ],
    "Rules / process": [
        r"\bscheme\s*rules?|policy|procedure|process|template|guidance|standard\b",
        r"\bvalidation|checklist|qa\s*(check)?\b",
        r"\bsla|timescale(s)?\b",
    ],
    "Disinvestment / funds": [
        r"\bdisinvest|switch|fund\s*(move|value|price)\b",
    ],
    "Bereavement / death": [
        r"\bdeath|deceased|probate|bereave(ment)?\b",
    ],
    "Divorce / split": [
        r"\bdivorce|pension\s*sharing|pension\s*split|pension\s*credit\b",
    ],
}

_COMPILED = [(label, [re.compile(p, re.I) for p in pats]) for label, pats in _RULES.items()]

# 2) Token → label hints (used to split "Other" in a second pass, low risk)
_TOKEN_HINTS = {
    "poa": "Data entry / setup",
    "proof address": "Data entry / setup",
    "address": "Data entry / setup",
    "pack": "Postal / dispatch",
    "returned": "Postal / dispatch",
    "holding letter": "Communication / update",
    "timescale": "Communication / update",
    "sla": "Rules / process",
    "quote": "Manual calculation",
    "cetv": "Manual calculation",
    "gmp": "Manual calculation",
    "disinvest": "Disinvestment / funds",
    "payroll": "Waiting on member/TPA",
    "employer": "Waiting on member/TPA",
    "provider": "Waiting on member/TPA",
    "3rd party": "Waiting on member/TPA",
    "portal": "System / workflow",
    "access": "System / workflow",
    "bug": "System / workflow",
    "error": "System / workflow",
}

def _clean_text(t: str) -> str:
    t = str(t or "").lower()
    t = re.sub(r"[_/\\\-]+", " ", t)
    t = re.sub(r"[^a-z0-9\s]+", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

def _label_reason_rules(text: str) -> str:
    t = _clean_text(text)
    if not t:
        return "Insufficient detail"
    for label, pats in _COMPILED:
        for p in pats:
            if p.search(t):
                return label
    return "Other"

def _ai_label_many(texts: List[str]) -> List[str]:
    if not _OPENAI or not texts:
        return [_label_reason_rules(t) for t in texts]

    labels = [_label_reason_rules(t) for t in texts]  # fallback
    try:
        allowed = list(_RULES.keys()) + ["Other", "Insufficient detail"]
        sys_msg = "You classify complaint review comments. Return only a JSON array of labels."
        instruction = (
            "Classify each bullet into exactly one of the following labels (prefer the most specific): "
            + ", ".join(allowed) + ".\nReturn ONLY a JSON array of strings (no prose)."
        )
        bullets = "\n".join(f"- {t}" for t in texts[:1500])
        resp = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            temperature=0,
            messages=[{"role": "system", "content": sys_msg},
                      {"role": "user", "content": instruction + "\n\n" + bullets}],
        )
        raw = resp["choices"][0]["message"]["content"]
        ai = json.loads(raw)
        if isinstance(ai, list) and len(ai) == len(texts[:len(ai)]):
            out = []
            for t, lab in zip(texts, ai):
                lab = str(lab).strip()
                if lab not in allowed:
                    lab = _label_reason_rules(t)
                out.append(lab)
            if len(out) < len(texts):
                out.extend(_label_reason_rules(t) for t in texts[len(out):])
            labels = out
    except Exception:
        pass
    return labels

def _second_pass_split_other(df_other: pd.DataFrame) -> pd.Series:
    """
    Split 'Other' using token hints. This is intentionally lightweight (no model),
    so it's safe and fast but breaks big 'Other' piles into meaningful buckets.
    """
    if df_other.empty:
        return pd.Series(dtype=str)

    def map_token(t: str) -> Optional[str]:
        t2 = _clean_text(t)
        for tok, lab in _TOKEN_HINTS.items():
            # token may be two words; check as substring safely
            if tok in t2:
                return lab
        return None

    mapped = df_other["comment"].astype(str).map(map_token)
    # Keep only confident mappings; None stays None and will be counted back to Other
    return mapped

def _reasons_latest(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Period]:
    df = df.assign(_m=_coerce_month(df["date"]))
    latest = df["_m"].max()
    if pd.isna(latest):
        return pd.DataFrame(), latest

    fails = df[(df["_m"] == latest) & (~df["result"].apply(_is_pass))]
    if fails.empty:
        return pd.DataFrame(columns=["reason", "count", "percent", "cum_percent"]), latest

    texts = fails.get("comment", pd.Series([], dtype=str)).astype(str).fillna("").tolist()
    ai_labels = _ai_label_many(texts)
    labels = [lab if lab in _RULES or lab in ("Other", "Insufficient detail") else _label_reason_rules(t)
              for t, lab in zip(texts, ai_labels)]

    base = pd.DataFrame({"reason": labels, "comment": fails["comment"].astype(str).values})
    # second pass: try to split Other with token hints
    oth = base[base["reason"] == "Other"]
    if not oth.empty:
        hinted = _second_pass_split_other(oth)
        idx = hinted[hinted.notna()].index
        base.loc[idx, "reason"] = hinted.loc[idx].values

    s = base["reason"].value_counts().rename_axis("reason").reset_index(name="count")
    s = s.sort_values("count", ascending=False).reset_index(drop=True)

    # Pareto: Top 80% + Other (ensure Other is at the end)
    total = int(s["count"].sum()) or 1
    s["percent"] = (s["count"] * 100.0 / total)
    s = s.sort_values("count", ascending=False).reset_index(drop=True)
    s["cum_percent"] = s["percent"].cumsum()

    head = s[s["cum_percent"] <= 80.0].copy()
    tail = s[s["cum_percent"] > 80.0].copy()

    # push any 'Other' in head into tail so it never hides real drivers
    if not head.empty and (head["reason"] == "Other").any():
        move = head[head["reason"] == "Other"]
        head = head[head["reason"] != "Other"]
        tail = pd.concat([tail, move], ignore_index=True)

    if not tail.empty:
        other_row = pd.DataFrame([{
            "reason": "Other",
            "count": int(tail["count"].sum()),
            "percent": float(tail["percent"].sum()),
            "cum_percent": 100.0
        }])
        head = pd.concat([head, other_row], ignore_index=True)

    head["percent"] = head["percent"].round(1)
    head["cum_percent"] = head["cum_percent"].round(1)
    return head, latest


# ---------------------------
# Plots (unchanged)
# ---------------------------
def _fig_mom(df: pd.DataFrame, title: str):
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    ax.plot(df["month"], df["pass_pct"], marker="o", linewidth=2.5, color="#9ecae1")
    for x, y in zip(df["month"], df["pass_pct"]):
        ax.text(x, y + 1, f"{y:.0f}%", ha="center", va="bottom", fontsize=9, color=_DARK_GREY)
    ax.set_title(title, pad=8, color=_DARK_BLUE)
    ax.set_ylim(bottom=0, top=100)
    for sp in ["left", "right", "top"]:
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color(_SOFT_GREY)
    ax.spines["bottom"].set_linewidth(1.25)
    ax.get_yaxis().set_visible(False)
    ax.set_xlabel(""); ax.set_ylabel(""); ax.grid(False)
    return fig

def _fig_reasons_bar(df: pd.DataFrame, title: str):
    fig, ax = plt.subplots(figsize=(7.0, 3.4))
    bars = ax.bar(df["reason"], df["count"])
    for b in bars:
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.5, f"{int(b.get_height())}",
                ha="center", va="bottom", fontsize=9, color=_DARK_GREY)
    ax.set_title(title, pad=8, color=_DARK_BLUE)
    for sp in ["left", "right", "top"]:
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color(_SOFT_GREY)
    ax.spines["bottom"].set_linewidth(1.25)
    ax.get_yaxis().set_visible(False)
    plt.setp(ax.get_xticklabels(), rotation=90, ha="center", color=_DARK_GREY)
    ax.grid(False)
    return fig


# ---------------------------
# Streamlit entry point (UNCHANGED)
# ---------------------------
def run(store: Dict, params: Dict, user_text: str = "") -> Tuple[str, pd.DataFrame]:
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
    table = _table_portfolio_scheme(df_raw, latest)

    c1, c2 = st.columns((1.1, 1.0), gap="large")
    with c1:
        st.pyplot(_fig_mom(mom, f"First-Pass Accuracy — Jan–{pd.Period(latest).to_timestamp().strftime('%b %y')}"))
    with c2:
        st.markdown(
            f"<h4 style='color:{_DARK_BLUE};margin:0 0 .5rem 0;'>"
            f"Pass % by Portfolio × Scheme — {pd.Period(latest).to_timestamp().strftime('%b-%y')}"
            f"</h4>", unsafe_allow_html=True)
        if not table.empty:
            st.dataframe(table, use_container_width=True)

    reasons, lastp = _reasons_latest(df_raw)
    st.markdown(
        f"<h4 style='color:{_DARK_BLUE};margin:1rem 0 .5rem 0;'>"
        f"Reasons for Fail — {pd.Period(lastp).to_timestamp().strftime('%b-%y')}"
        f"</h4>", unsafe_allow_html=True)
    r1, r2 = st.columns(2, gap="large")
    with r1:
        if not reasons.empty:
            st.pyplot(_fig_reasons_bar(reasons[["reason", "count"]], "Fail reasons — Pareto (top 80% + Other)"))
    with r2:
        if not reasons.empty:
            st.dataframe(reasons, use_container_width=True)

    return ("", pd.DataFrame())
