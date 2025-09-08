# questions/first_pass_accuracy.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple, Optional, List
import os
import re
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# ---------------------------
# Brand colours / small style
# ---------------------------
_DARK_BLUE = "#0b3d91"
_DARK_GREY = "#333333"
_SOFT_GREY = "#DDDDDD"

# Optional line colour for MoM; bars use mpl defaults
_PARETO = "#6ab6e1"

# ---------------------------
# Optional OpenAI assist (used only if key present)
# ---------------------------
_OPENAI = False
try:
    # prefer Streamlit secrets; then ENV
    _OPENAI_KEY = (st.secrets.get("OPENAI_API_KEY") if hasattr(st, "secrets") else None) or os.getenv("OPENAI_API_KEY")
    if _OPENAI_KEY:
        _OPENAI = True
        import openai  # type: ignore
        openai.api_key = _OPENAI_KEY
except Exception:
    _OPENAI = False


# =====================================================================
# Data loading — unchanged behaviour
# =====================================================================
def _find_fpa_workbook() -> Optional[Path]:
    """
    Look in the standard locations and pick the newest matching file:
    data/first_pass_accuracy/FirstPassAccuracy*.xlsx (or .xls)
    """
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
        # fall back to safe header
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
        # if you ever add RCA columns in the source, we can reference them here
        # "rca2": _pick(df, ["RCA2", "Root Cause 2"]),
    }
    missing = [k for k, v in col_map.items() if k in ("date", "result") and v is None]
    if missing:
        raise KeyError(f"Missing required columns for FPA: {missing}")
    return df.rename(columns={v: k for k, v in col_map.items() if v}), col_map


# =====================================================================
# Pass% + table — unchanged behaviour
# =====================================================================
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


# =====================================================================
# Fail reason classification — now wired to core.reason_labeller
# with safe fallbacks to in-file rules and (optionally) OpenAI
# =====================================================================

# ---- 1) Fallback rulebook (kept in-file so Q1 remains self-contained) ----
_RULES = {
    "Bank / payment": [
        r"\b(bank|payment|refund|bacs|chaps|cheque|sort\s*code|iban|bic|account|transfer|credit|debit)\b",
        r"\bpaid\s*to\s*wrong|duplicate\s*payment|missing\s*payment\b",
    ],
    "Communication / update": [
        r"\b(no|missing)\s*(reply|response|update)\b",
        r"\bupdate|communicat|clarif|explain|advise|inform(ed|ation)?\b",
        r"\bconfus|unclear|mis(lead|understand)\b",
        r"\bcall(s|ed)?|email(s|ed)?|letter(s)?\b",
    ],
    "Data entry / setup": [
        r"\bwrong|incorrect|mis-?key|typo|misallocat|miscode|set\s*up|setup\b",
        r"\bdata\s*(entry|load|issue)|capture|key(ed|ing)\b",
        r"\bdate\s*error|dob|ni\s*number|nino\b",
    ],
    "Postal / dispatch": [
        r"\b(post|mail|postal|dispatch|despatch|send|sent|deliver(y|ed)?)\b",
        r"\breturned\s*mail|wrong\s*address\b",
    ],
    "Manual calculation": [
        r"\bmanual\b.*calc|re-?calc|recalculation|calc(ulation)?\s*error\b",
    ],
    "Waiting on member/TPA": [
        r"\bawait|waiting\s*for|chase(d|s|ing)?\b",
        r"\bthird\s*party|tpa|ifa|insurer|administrator|employer|payroll|trustee\b",
        r"\bmember\s*to\s*(respond|confirm|provide)\b",
    ],
    "Trustee / AVC": [
        r"\btrustee|avc|additional\s*voluntary\s*contribution\b",
    ],
    "System / workflow": [
        r"\bsystem|portal|platform|workflow|work\s*queue|technical|bug|defect|automation|script\b",
        r"\baccess|permission|role|profile\b",
    ],
    "Rules / process": [
        r"\bscheme\s*rules?|policy|procedure|process|template|guidance|standard\b",
        r"\bvalidation|checklist|qa\s*(check)?\b",
    ],
}
_COMPILED = [(lab, [re.compile(p, re.I) for p in pats]) for lab, pats in _RULES.items()]


def _clean_text(t: str) -> str:
    t = str(t or "").lower()
    t = re.sub(r"[_/\\\-]+", " ", t)
    t = re.sub(r"[^a-z0-9\s]+", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _label_reason_rules(text: str) -> str:
    t = _clean_text(text)
    for label, pats in _COMPILED:
        for p in pats:
            if p.search(t):
                return label
    return "Other"


def _ai_label_many(texts: List[str]) -> List[str]:
    """
    If OPENAI_API_KEY is available, ask the model to label items using our allowed set.
    We still validate each suggestion against the local rulebook to avoid creative answers.
    """
    if not _OPENAI or not texts:
        return [_label_reason_rules(t) for t in texts]

    labels = [_label_reason_rules(t) for t in texts]  # safe default
    try:
        allowed = list(_RULES.keys()) + ["Other"]
        sys_msg = "You classify complaint review comments. Only return valid JSON array of labels."
        instruction = (
            "Classify each bullet into exactly one of the following labels (prefer the most specific): "
            + ", ".join(allowed)
            + ".\nReturn ONLY a JSON array of strings (no prose)."
        )
        bullets = "\n".join(f"- {t}" for t in texts[:1500])  # safety cap for payload
        import openai  # type: ignore

        resp = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            temperature=0,
            messages=[
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": instruction + "\n\n" + bullets},
            ],
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


def _label_with_reason_labeller(texts: List[str]) -> Optional[List[str]]:
    """
    Try to use core.reason_labeller (your centralized labelling logic that can
    combine YAML patterns + RCA + optional OpenAI). If anything fails, return None.
    """
    try:
        from core.reason_labeller import ReasonLabeller  # your shared module
        rl = ReasonLabeller(
            # keep paths stable with your repo
            patterns_path="data/surveys/fpa_patterns.yml",
            rca_patterns_path="data/surveys/rca_patterns.yml",
            openai_key=(st.secrets.get("OPENAI_API_KEY") if hasattr(st, "secrets") else None) or os.getenv("OPENAI_API_KEY"),
            openai_model="gpt-4o-mini",
        )
        return rl.label_many(texts)
    except Exception:
        return None  # fall back to in-file logic


def _reasons_latest(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Period]:
    df = df.assign(_m=_coerce_month(df["date"]))
    latest = df["_m"].max()
    if pd.isna(latest):
        return pd.DataFrame(), latest

    fails = df[(df["_m"] == latest) & (~df["result"].apply(_is_pass))]
    if fails.empty:
        return pd.DataFrame(columns=["reason", "count", "percent", "cum_percent"]), latest

    if "comment" not in fails.columns:
        return pd.DataFrame(columns=["reason", "count", "percent", "cum_percent"]), latest

    texts = fails["comment"].astype(str).fillna("").tolist()

    # ---- Prefer your shared reason_labeller, then fall back to OpenAI+rules, then rules only
    labels = _label_with_reason_labeller(texts)
    if labels is None:
        labels = _ai_label_many(texts)

    s = pd.Series(labels).value_counts().rename_axis("reason").reset_index(name="count")
    s = s.sort_values("count", ascending=False).reset_index(drop=True)

    # ---- Pareto: Top 80% + collapse the rest into "Other"
    total = int(s["count"].sum()) or 1
    s["percent"] = (s["count"] * 100.0 / total)
    s = s.sort_values("count", ascending=False).reset_index(drop=True)
    s["cum_percent"] = s["percent"].cumsum()

    head = s[s["cum_percent"] <= 80.0].copy()
    tail = s[s["cum_percent"] > 80.0].copy()

    # ensure 'Other' doesn’t occupy the head bucket if present
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


# =====================================================================
# Plot helpers — unchanged visuals
# =====================================================================
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


# =====================================================================
# Streamlit entry — unchanged signature/UI
# =====================================================================
def run(store: Dict, params: Dict, user_text: str = "") -> Tuple[str, pd.DataFrame]:
    # 1) Load & validate FPA
    try:
        df_raw, _ = _load_fpa()
    except FileNotFoundError as e:
        st.error(str(e)); return ("", pd.DataFrame())
    except KeyError as e:
        st.error(f"FPA file found, but a required column is missing: {e}")
        return ("", pd.DataFrame())

    # 2) Pass % MoM
    mom = _series_mom(df_raw)
    if mom.empty:
        st.info("No First-Pass Accuracy rows found from Jan-25 onward.")
        return ("", pd.DataFrame())

    # 3) Latest month, portfolio×scheme table
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

    # 4) Reasons Pareto (now via reason_labeller where available)
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
