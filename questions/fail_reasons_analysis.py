# questions/fail_reasons_analysis.py
from __future__ import annotations

# --- stdlib / typing
import os
import re
import json
from pathlib import Path
from typing import Dict, Tuple, Optional, List

# --- third-party (already in your app)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# ---------------------------
# Theme (kept consistent with Q2)
# ---------------------------
_DARK_BLUE = "#0b3d91"
_DARK_GREY = "#333333"
_SOFT_GREY = "#DDDDDD"

# ---------------------------
# Optional OpenAI assist (safe fallback when no key)
# ---------------------------
_OPENAI = False
try:
    import openai  # type: ignore
    if os.getenv("OPENAI_API_KEY"):
        openai.api_key = os.getenv("OPENAI_API_KEY")
        _OPENAI = True
except Exception:
    _OPENAI = False


# ============================================================================
#                           DATA LOADING (same as Q2)
# ============================================================================
def _find_fpa_workbook() -> Optional[Path]:
    """
    Find the FirstPassAccuracy workbook in the same places Q2 uses.
    """
    roots = [
        Path("data/first_pass_accuracy"),
        Path("first_pass_accuracy"),
        Path("data/first_pass_accuracy/"),
    ]
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
    """
    Load the FPA workbook and harmonize column names just like Q2.
    Required: Activity Date (-> date), Review Result (-> result)
    Optional: Case Comment (-> comment), Portfolio, Scheme.
    """
    p = _find_fpa_workbook()
    if not p:
        raise FileNotFoundError(
            "Could not find a FirstPassAccuracy workbook (FirstPassAccuracy*.xlsx)."
        )

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


# ============================================================================
#                         FAIL REASON CLASSIFICATION
# ============================================================================
def _is_pass(x: str) -> bool:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return False
    t = str(x).strip().lower()
    return t.startswith("pass")


# Expanded rulebook (ordered: first match wins)
_RULES: Dict[str, List[str]] = {
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

_COMPILED = [(label, [re.compile(p, re.I) for p in pats]) for label, pats in _RULES.items()]


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
    If an OpenAI key is present, ask the model for labels, but validate against the rulebook.
    Falls back to pure rules when the key/model isn’t available.
    """
    if not _OPENAI or not texts:
        return [_label_reason_rules(t) for t in texts]

    labels = [_label_reason_rules(t) for t in texts]  # default fallback
    try:
        allowed = list(_RULES.keys()) + ["Other"]
        sys_msg = "You classify complaint review comments. Only return valid JSON array of labels."
        instruction = (
            "Classify each bullet into exactly one of the following labels (prefer the most specific): "
            + ", ".join(allowed)
            + ".\nReturn ONLY a JSON array of strings (no prose)."
        )
        bullets = "\n".join(f"- {t}" for t in texts[:1500])  # safety cap
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
        # keep rule-based labels
        pass
    return labels


def _reasons_latest(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Period]:
    """
    Build a Pareto for the latest month: top 80% categories + collapse tail into 'Other'.
    """
    df = df.assign(_m=_coerce_month(df["date"]))
    latest = df["_m"].max()
    if pd.isna(latest):
        return pd.DataFrame(), latest

    fails = df[(df["_m"] == latest) & (~df["result"].apply(_is_pass))]
    if fails.empty:
        return pd.DataFrame(), latest

    if "comment" not in fails.columns:
        return pd.DataFrame(), latest

    texts = fails["comment"].astype(str).fillna("").tolist()

    # 1) AI assist (optional) then validate with rules
    ai_labels = _ai_label_many(texts)
    labels = [
        lab if lab in _RULES or lab == "Other" else _label_reason_rules(t)
        for t, lab in zip(texts, ai_labels)
    ]

    s = pd.Series(labels).value_counts().rename_axis("reason").reset_index(name="count")
    s = s.sort_values("count", ascending=False).reset_index(drop=True)

    # 2) Pareto head (<=80% cum %) + pack the rest as Other
    total = int(s["count"].sum()) or 1
    s["percent"] = (s["count"] * 100.0 / total)
    s["cum_percent"] = s["percent"].cumsum()

    head = s[s["cum_percent"] <= 80.0].copy()
    tail = s[s["cum_percent"] > 80.0].copy()

    # keep informative categories in head; push any 'Other' to tail
    if not head.empty and (head["reason"] == "Other").any():
        move = head[head["reason"] == "Other"]
        head = head[head["reason"] != "Other"]
        tail = pd.concat([tail, move], ignore_index=True)

    if not tail.empty:
        other_row = pd.DataFrame(
            [{"reason": "Other", "count": int(tail["count"].sum()), "percent": float(tail["percent"].sum())}]
        )
        head = pd.concat([head, other_row], ignore_index=True)

    head["percent"] = head["percent"].round(1)
    head["cum_percent"] = head["percent"].cumsum().round(1)
    return head, latest


# ============================================================================
#                                  PLOTS
# ============================================================================
def _fig_reasons_bar(df: pd.DataFrame, title: str):
    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    bars = ax.bar(df["reason"], df["count"])
    for b in bars:
        ax.text(
            b.get_x() + b.get_width() / 2,
            b.get_height() + 0.5,
            f"{int(b.get_height())}",
            ha="center",
            va="bottom",
            fontsize=9,
            color=_DARK_GREY,
        )
    ax.set_title(title, pad=8, color=_DARK_BLUE)
    for sp in ["left", "right", "top"]:
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color(_SOFT_GREY)
    ax.spines["bottom"].set_linewidth(1.25)
    ax.get_yaxis().set_visible(False)
    plt.setp(ax.get_xticklabels(), rotation=90, ha="center", color=_DARK_GREY)
    ax.grid(False)
    return fig


# ============================================================================
#                           STREAMLIT ENTRY POINT
# ============================================================================
def run(store: Dict, params: Dict, user_text: str = "") -> Tuple[str, pd.DataFrame]:
    """
    Signature identical to Q1/Q2. Does not read or mutate `store`
    (so nothing you do here can break Q1 or Q2).
    """
    # 1) Load the same workbook Q2 uses
    try:
        df_raw, _ = _load_fpa()
    except FileNotFoundError as e:
        st.error(str(e))
        return ("", pd.DataFrame())
    except KeyError as e:
        st.error(f"FPA file found, but a required column is missing: {e}")
        return ("", pd.DataFrame())

    # 2) Build latest-month Pareto of failure reasons
    reasons, latest = _reasons_latest(df_raw)

    # 3) Render
    st.markdown(
        f"<h3 style='color:{_DARK_BLUE};margin:0 0 .75rem 0;'>"
        f"Reasons for Fail — {pd.Period(latest).to_timestamp().strftime('%b-%y') if pd.notna(latest) else 'No date'}"
        f"</h3>",
        unsafe_allow_html=True,
    )

    if reasons.empty:
        st.info("No failing cases with comments were found for the latest month.")
        return ("", pd.DataFrame())

    left, right = st.columns(2, gap="large")
    with left:
        st.pyplot(_fig_reasons_bar(reasons[["reason", "count"]], "Fail reasons — Pareto (top 80% + Other)"))
    with right:
        st.dataframe(reasons, use_container_width=True)

    # Optional helper note so you know whether AI was used
    if not _OPENAI:
        st.caption("OpenAI labelling inactive (no OPENAI_API_KEY). Rule-based labels shown.")

    return ("", pd.DataFrame())
