# questions/first_pass_accuracy.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple, Optional, List
import os
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

_DARK_BLUE = "#0b3d91"
_DARK_GREY = "#333333"
_SOFT_GREY = "#DDDDDD"
_PARETO = "#6ab6e1"

# Optional: OpenAI to tighten 'Other' classification. Falls back to keyword rules.
_OPENAI = False
try:
    import openai  # type: ignore
    if os.getenv("OPENAI_API_KEY"):
        openai.api_key = os.getenv("OPENAI_API_KEY")
        _OPENAI = True
except Exception:
    _OPENAI = False


# ---------------------------
# Data loading (self-contained)
# ---------------------------
def _find_fpa_workbook() -> Optional[Path]:
    # Look in both "data/first_pass_accuracy/" and "first_pass_accuracy/" with flexible names
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
    # single sheet read; if your workbook has a named sheet, you can pass sheet_name=...
    try:
        return pd.read_excel(path)  # engine auto-detected
    except Exception:
        # occasionally first row is header row #1 — try header=0 explicitly
        return pd.read_excel(path, header=0)


# ---------------------------
# Column helpers
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
# Pass% + table logic
# ---------------------------
def _is_pass(x: str) -> bool:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return False
    t = str(x).strip().lower()
    # treat anything starting with 'pass' as pass; everything else becomes fail
    return t.startswith("pass")

def _series_mom(df: pd.DataFrame) -> pd.DataFrame:
    s = _coerce_month(df["date"])
    df = df.assign(_m=s)
    # Restrict from Jan-2025 to latest
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
    return grp[["portfolio", "scheme", "cases", "pass_%"]].sort_values(["portfolio", "pass_%", "scheme"], ascending=[True, False, True])

# ---------------------------
# Fail reason classification
# ---------------------------
_RULES = {
    "Incorrect data / miskeying": [r"\bwrong|incorrect|mis-?key|typo|misallocat|data entry\b"],
    "Missing docs / evidence":    [r"\bmissing\b.*(doc|document|evidence|letter|form)|\bnot (provided|received)\b"],
    "Communication / update":     [r"\bno (reply|response)|update|communicat|unclear|confus|call|email|letter\b"],
    "Bank / payment":             [r"\b(bank|payment|refund|bacs|chaps|cheque|sort code|account)\b"],
    "System / workflow":          [r"\bsystem|portal|workflow|technical|bug|automation\b"],
    "Waiting on member/TPA":      [r"\bawait|chase|ifa|third party|insurer|trustee\b"],
    "Rules / process":            [r"\bscheme rules?|procedure|process|template\b"],
    "Manual calc":                [r"\bmanual\b.*calc|re-?calc|recalculation\b"],
}

def _label_reason(text: str) -> str:
    t = str(text or "").lower()
    t = re.sub(r"[^a-z0-9 ]+", " ", t)
    for label, pats in _RULES.items():
        for p in pats:
            if re.search(p, t):
                return label
    return "Other"

def _ai_label_many(texts: List[str]) -> List[str]:
    if not _OPENAI:
        return [_label_reason(t) for t in texts]
    try:
        prompt = (
            "Classify each case comment into one of these labels: "
            + ", ".join(sorted(list(_RULES.keys()) + ["Other"]))
            + ". Prefer a specific label over 'Other'. "
            "Return a JSON array of strings only."
        )
        msgs = [{"role": "system", "content": "Return only valid JSON."},
                {"role": "user", "content": prompt + "\n\n" + "\n".join(f"- {t}" for t in texts)}]
        # gpt-4o-mini or similar
        resp = openai.ChatCompletion.create(model="gpt-4o-mini", messages=msgs, temperature=0)
        arr = pd.read_json(resp["choices"][0]["message"]["content"], typ="series")
        labels = [str(v) for v in arr.tolist()]
        # Fallback to rules if anything off
        if len(labels) != len(texts):
            return [_label_reason(t) for t in texts]
        return labels
    except Exception:
        return [_label_reason(t) for t in texts]

def _reasons_latest(df: pd.DataFrame) -> pd.DataFrame:
    df = df.assign(_m=_coerce_month(df["date"]))
    latest = df["_m"].max()
    if pd.isna(latest):
        return pd.DataFrame(), latest
    sub = df[(df["_m"] == latest) & (~df["result"].apply(_is_pass))]
    comments_col = "comment" if "comment" in sub.columns else None
    if comments_col is None or sub.empty:
        return pd.DataFrame(columns=["reason", "count", "percent", "cum_percent"]), latest

    texts = sub[comments_col].astype(str).fillna("").tolist()
    labels = _ai_label_many(texts)
    s = pd.Series(labels).value_counts().rename_axis("reason").reset_index(name="count")

    # Pareto: top 80% + Other bucket
    s["percent"] = (s["count"] * 100.0 / max(1, s["count"].sum()))
    s["cum_percent"] = s["percent"].cumsum()
    s = s.sort_values("count", ascending=False).reset_index(drop=True)
    # keep top 80% explicitly; merge the tail as "Other"
    head = s[s["cum_percent"] <= 80.0].copy()
    tail = s[s["cum_percent"] > 80.0]
    if not tail.empty:
        head = pd.concat(
            [head, pd.DataFrame([{
                "reason": "Other",
                "count": int(tail["count"].sum()),
                "percent": float(tail["percent"].sum()),
                "cum_percent": 100.0
            }])],
            ignore_index=True
        )
    head["percent"] = head["percent"].round(1)
    head["cum_percent"] = head["cum_percent"].round(1)
    return head, latest


# ---------------------------
# Plots
# ---------------------------
def _fig_mom(df: pd.DataFrame, title: str):
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    ax.plot(df["month"], df["pass_pct"], marker="o", linewidth=2.5, color="#9ecae1")
    for x, y in zip(df["month"], df["pass_pct"]):
        ax.text(x, y + 1, f"{y:.0f}%", ha="center", va="bottom", fontsize=9, color=_DARK_GREY)
    ax.set_title(title, pad=8, color=_DARK_BLUE)
    ax.set_ylim(bottom=0, top=100)
    # clean
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
    for i, b in enumerate(bars):
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.5, f"{int(b.get_height())}", ha="center", va="bottom", fontsize=9, color=_DARK_GREY)
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
# Streamlit entry point
# ---------------------------
def run(store: Dict, params: Dict, user_text: str = "") -> Tuple[str, pd.DataFrame]:
    try:
        df_raw, cmap = _load_fpa()
    except FileNotFoundError as e:
        st.error(str(e))
        return ("", pd.DataFrame())
    except KeyError as e:
        st.error(f"FPA file found, but a required column is missing: {e}")
        return ("", pd.DataFrame())

    # Series + table
    mom = _series_mom(df_raw)
    if mom.empty:
        st.info("No First-Pass Accuracy rows found from Jan-25 onward.")
        return ("", pd.DataFrame())

    # Right-side table: pass% by Portfolio × Scheme for latest month
    df_raw = df_raw.assign(_m=_coerce_month(pd.to_datetime(df_raw["date"], errors="coerce", dayfirst=True)))
    latest = df_raw["_m"].max()
    table = _table_portfolio_scheme(df_raw, latest)

    # Draw MoM + table
    c1, c2 = st.columns((1.1, 1.0), gap="large")
    with c1:
        st.pyplot(_fig_mom(mom, f"First-Pass Accuracy — Jan–{pd.Period(latest).strftime('%b %y')}"))
    with c2:
        st.markdown(f"<h4 style='color:{_DARK_BLUE};margin:0 0 .5rem 0;'>Pass % by Portfolio × Scheme — {pd.Period(latest).strftime('%b-%y')}</h4>", unsafe_allow_html=True)
        if not table.empty:
            st.dataframe(table, use_container_width=True)

    # Reasons (latest month) — chart + table side by side
    reasons, lastp = _reasons_latest(df_raw)
    st.markdown(f"<h4 style='color:{_DARK_BLUE};margin:1rem 0 .5rem 0;'>Reasons for Fail — {pd.Period(lastp).strftime('%b-%y')}</h4>", unsafe_allow_html=True)
    r1, r2 = st.columns(2, gap="large")
    with r1:
        if not reasons.empty:
            st.pyplot(_fig_reasons_bar(reasons[["reason", "count"]], "Fail reasons — Pareto (top 80% + Other)"))
    with r2:
        if not reasons.empty:
            st.dataframe(reasons, use_container_width=True)

    # Nothing extra to return; we rendered our own layout
    return ("", pd.DataFrame())
