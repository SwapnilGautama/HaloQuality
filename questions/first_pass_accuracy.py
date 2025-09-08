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
# Data loading — unchanged behaviour (with optional RCA2 pickup)
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
        # Optional RCA2 if present in the workbook — helps ML generalise
        "rca2": _pick(df, ["RCA2", "Root Cause 2", "RCA 2"]),
    }
    missing = [k for k, v in col_map.items() if k in ("date", "result") and v is None]
    if missing:
        raise KeyError(f"Missing required columns for FPA: {missing}")

    df = df.rename(columns={v: k for k, v in col_map.items() if v})
    df["date"] = pd.to_datetime(df["date"], errors="coerce", dayfirst=True)
    return df, col_map


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
# Fail reason classification — FIXED to use shared reason_labeller
# (functions: get_or_fit_model + label_dataframe) with safe fallbacks
# =====================================================================

# ---- 1) Fallback rulebook (kept in-file so FPA remains self-contained) ----
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


def _label_with_reason_labeller(texts: List[str], rca2_vals: Optional[List[str]], df_all: pd.DataFrame) -> Optional[List[str]]:
    """
    Use shared core.reason_labeller functional API.
    Falls back to None if import fails, so caller can try AI/rules.
    """
    try:
        # Import **functions** (there is no ReasonLabeller class)
        from core.reason_labeller import get_or_fit_model, label_dataframe  # type: ignore

        # Build a DataFrame with canonical column names that the labeller expects
        df_all_for_model = pd.DataFrame({
            "Case Comme
