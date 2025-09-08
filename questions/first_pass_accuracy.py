# questions/first_pass_accuracy.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple, Optional, List
import os
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# ---------- Styles (unchanged) ----------
_DARK_BLUE = "#0b3d91"
_DARK_GREY = "#333333"
_SOFT_GREY = "#DDDDDD"

# ---------- OpenAI-only classification ----------
_OAI_READY = False
_OAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
try:
    import openai  # type: ignore
    if os.getenv("OPENAI_API_KEY"):
        openai.api_key = os.getenv("OPENAI_API_KEY")
        _OAI_READY = True
except Exception:
    _OAI_READY = False

# Allowed categories (broader, more actionable buckets)
_ALLOWED_LABELS: List[str] = [
    "Bank / payment",
    "Communication / update",
    "Data entry / setup",
    "Postal / dispatch",
    "Manual calculation",
    "Waiting on member/TPA",
    "Trustee / AVC",
    "System / workflow",
    "Rules / process",
    "Other",
]

_ALLOWED_SET = {x.lower(): x for x in _ALLOWED_LABELS}


# =========================================================
# Data loading (unchanged)
# =========================================================
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


# =========================================================
# Pass% + table logic (unchanged)
# =========================================================
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


# =========================================================
# OpenAI-only classification
# =========================================================
def _normalize_label(label: str) -> str:
    """Map any model output to the nearest allowed label (strict)."""
    if not isinstance(label, str):
        return "Other"
    t = label.strip().lower()
    # exact or near-exact match
    if t in _ALLOWED_SET:
        return _ALLOWED_SET[t]
    # simple loose match by token overlap
    for allowed in _ALLOWED_LABELS:
        toks = set(allowed.lower().replace("/", " ").split())
        if any(tok in t for tok in toks):
            return allowed
    return "Other"

def _oai_label_batch(texts: List[str]) -> List[str]:
    """Call OpenAI once for a batch of comments and return a label per item."""
    system = (
        "You are a quality analyst. Classify each bullet point into ONE label.\n"
        "Use ONLY one of these labels exactly:\n"
        f"{', '.join(_ALLOWED_LABELS)}.\n"
        "If the text is unclear or doesn't fit, use 'Other'.\n"
        "Return ONLY a valid JSON array of strings (no extra text)."
    )
    bullets = "\n".join(f"- {t}" for t in texts)
    user = "Classify the following items:\n\n" + bullets
    resp = openai.ChatCompletion.create(
        model=_OAI_MODEL,
        temperature=0,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    raw = resp["choices"][0]["message"]["content"]
    try:
        arr = json.loads(raw)
        if not isinstance(arr, list):
            raise ValueError("Not a list")
    except Exception:
        # On parsing issues, fall back to all Other (but app still renders)
        return ["Other"] * len(texts)
    # Normalize to allowed set
    return [_normalize_label(x) for x in arr][: len(texts)]

def _ai_label_many(texts: List[str]) -> List[str]:
    """Chunk to stay safe on very large datasets."""
    if not _OAI_READY:
        # No API key => render the rest of the app; label as Other
        return ["Other"] * len(texts)

    out: List[str] = []
    BATCH = 60  # conservative chunk size
    for i in range(0, len(texts), BATCH):
        chunk = texts[i : i + BATCH]
        out.extend(_oai_label_batch(chunk))
    # If the model returned fewer than expected for any reason, pad with Other
    if len(out) < len(texts):
        out.extend(["Other"] * (len(texts) - len(out)))
    return out


# =========================================================
# Reasons (OpenAI-only) + Pareto
# =========================================================
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

    texts = fails["comment"].fillna("").astype(str).tolist()

    labels = _ai_label_many(texts)  # <-- OpenAI-only
    s = pd.Series(labels).value_counts().rename_axis("reason").reset_index(name="count")
    s = s.sort_values("count", ascending=False).reset_index(drop=True)

    total = int(s["count"].sum()) or 1
    s["percent"] = s["count"] * 100.0 / total
    s["cum_percent"] = s["percent"].cumsum()

    # Pareto: Top 80% + Other (push pre-existing 'Other' into the tail if it's within top-80 by count)
    head = s[s["cum_percent"] <= 80.0].copy()
    tail = s[s["cum_percent"] > 80.0].copy()
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


# =========================================================
# Plots (unchanged)
# =========================================================
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


# =========================================================
# Streamlit entry point (unchanged interface)
# =========================================================
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

    if not _OAI_READY:
        st.info(
            "OpenAI labelling inactive (no OPENAI_API_KEY). "
            "Fail reasons will be shown as ‘Other’ to preserve rendering."
        )

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
