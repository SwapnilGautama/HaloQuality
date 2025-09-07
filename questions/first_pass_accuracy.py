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

_DARK_BLUE = "#0b3d91"
_DARK_GREY = "#333333"
_SOFT_GREY = "#DDDDDD"

# =========================
# Optional OpenAI
# =========================
_OPENAI = False
_new_client = None
_old_openai = None
try:
    # Prefer the new SDK
    from openai import OpenAI  # type: ignore
    if os.getenv("OPENAI_API_KEY"):
        _new_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        _OPENAI = True
except Exception:
    _new_client = None
try:
    # Fallback to legacy SDK if present
    import openai  # type: ignore
    if not _OPENAI and os.getenv("OPENAI_API_KEY"):
        openai.api_key = os.getenv("OPENAI_API_KEY")
        _old_openai = openai
        _OPENAI = True
except Exception:
    _old_openai = None


# =========================
# Where to find FPA workbook
# =========================
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


# =========================
# Column mapping helpers
# =========================
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


# =========================
# Pass % logic
# =========================
def _is_pass(x: str) -> bool:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return False
    return str(x).strip().lower().startswith("pass")

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


# =========================
# Classification (labels + rules + OpenAI)
# =========================
# Canonical, human-friendly labels you’ll see in the UI:
_LABELS = [
    "Bank / payment",
    "Postal / dispatch",
    "Data entry / setup",
    "Manual calculation",
    "Communication / update",
    "Waiting on member/TPA",
    "Trustee / AVC",
    "Rules / process",
    "System / workflow",
    "Death benefits",
    "Pension set up",
    "Case not created",
    "Other",
]

# Stronger keyword rules (offline fallback)
_RULES = {
    "Bank / payment": [
        r"\bbank\b", r"payment|paid|refund|re-?fund", r"\bbacs\b|\bchaps\b|\bcheque\b",
        r"sort\s?code|account\s?no|account number|iban|swift|transfer"
    ],
    "Postal / dispatch": [r"post|postal|dispatch|mail|sent|deliver|letter\s+sent|not\s+received"],
    "Data entry / setup": [r"mis-?key|key(ing)?\s?error|data entry|setup|set[- ]?up|load(ed)?\s?wrong|typo"],
    "Manual calculation": [r"manual.*calc|manual\s+calc|re-?calc|recalculation|calc(ulation)?\s+error"],
    "Communication / update": [
        r"no\s+(reply|response)|await(ing)?\s+update|follow-?up|communicat|unclear|confus|call(ed)?|emailed?",
        r"letter\b(?!\s*sent)"  # mentions of letter without 'sent'
    ],
    "Waiting on member/TPA": [r"await(ing)?\s*(member|customer|client|tp[a]?|third\s+party|insurer|ifa)"],
    "Trustee / AVC": [r"trustee|avc\b|additional\s+voluntary|scheme\s+trustee"],
    "Rules / process": [r"scheme\s+rule|rules\b|procedure|process|template|policy|guidance"],
    "System / workflow": [r"system|portal|workflow|technical|bug|down|crash|automation|script"],
    "Death benefits": [r"death\s+benefit|bereave|deceas"],
    "Pension set up": [r"pension\s+set\s?up|new\s+joiner|enrol(ment|lment)|scheme\s+setup"],
    "Case not created": [r"case\s+not\s+created|no\s+case|missing\s+case|unlogged|not\s+logged"],
}

def _label_reason_rule(text: str) -> str:
    t = str(text or "").lower()
    t = re.sub(r"[^a-z0-9 ]+", " ", t)
    for label, pats in _RULES.items():
        for p in pats:
            if re.search(p, t):
                return label
    return "Communication / update" if re.search(r"call|email|reply|response|update|communicat|confus", t) else "Other"

def _label_many_rule(texts: List[str]) -> List[str]:
    return [_label_reason_rule(t) for t in texts]

def _label_many_openai(texts: List[str]) -> List[str]:
    """
    Classify with OpenAI, returning exactly one label from _LABELS for each input line.
    Robust to failures: any parsing issues fall back to rules for that batch.
    """
    if not _OPENAI or not texts:
        return _label_many_rule(texts)

    allowed = [l for l in _LABELS if l != "Other"] + ["Other"]
    batches = []
    # Keep batches conservative for long comments
    chunk = 120
    for i in range(0, len(texts), chunk):
        batches.append(texts[i:i + chunk])

    out: List[str] = []
    system = (
        "You are a strict classifier for pension administration QA notes. "
        "Pick exactly ONE label for each note from the allowed set. "
        "Return ONLY a valid JSON array of strings with length equal to the number of notes. "
        "No explanations."
    )
    allowed_str = ", ".join(allowed)

    for batch in batches:
        user = (
            "Allowed labels (choose exactly ONE for each item): "
            f"{allowed_str}\n\n"
            "Classify the following notes (one label per item, same order). "
            "If the note doesn't fit any label, use 'Other'.\n\n" +
            "\n".join(f"- {t}" for t in batch)
        )

        try:
            # New SDK first
            if _new_client is not None:
                resp = _new_client.chat.completions.create(
                    model=os.getenv("OPENAI_CLASSIFY_MODEL", "gpt-4o-mini"),
                    temperature=0,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                )
                content = resp.choices[0].message.content
            # Legacy SDK fallback
            else:
                resp = _old_openai.ChatCompletion.create(
                    model=os.getenv("OPENAI_CLASSIFY_MODEL", "gpt-4o-mini"),
                    temperature=0,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                )
                content = resp["choices"][0]["message"]["content"]

            labels = json.loads(content)
            if not isinstance(labels, list) or len(labels) != len(batch):
                raise ValueError("Malformed JSON or wrong length")

            # force to allowed set; fall back to rules per item if needed
            cleaned = []
            for t, lab in zip(batch, labels):
                lab = str(lab).strip()
                cleaned.append(lab if lab in allowed else _label_reason_rule(t))

            out.extend(cleaned)

        except Exception:
            out.extend(_label_many_rule(batch))

    return out


# =========================
# Reasons (latest month) + Pareto
# =========================
def _reasons_latest(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Period]:
    df = df.assign(_m=_coerce_month(df["date"]))
    latest = df["_m"].max()
    if pd.isna(latest):
        return pd.DataFrame(), latest

    sub = df[(df["_m"] == latest) & (~df["result"].apply(_is_pass))]
    if sub.empty:
        return pd.DataFrame(columns=["reason", "count", "percent", "cum_percent"]), latest

    comments_col = "comment" if "comment" in sub.columns else None
    if comments_col is None:
        return pd.DataFrame(columns=["reason", "count", "percent", "cum_percent"]), latest

    texts = sub[comments_col].astype(str).fillna("").tolist()
    labels = _label_many_openai(texts)  # OpenAI if available; otherwise rules

    s = pd.Series(labels).value_counts().rename_axis("reason").reset_index(name="count")

    # Merge any stray/unknown labels to "Other"
    s["reason"] = s["reason"].apply(lambda x: x if x in _LABELS else "Other")
    s = s.groupby("reason", as_index=False)["count"].sum()

    # Sort and build Pareto
    s = s.sort_values("count", ascending=False).reset_index(drop=True)
    s["percent"] = (s["count"] * 100.0 / max(1, s["count"].sum()))
    s["cum_percent"] = s["percent"].cumsum()

    # Keep explicit top-80% and aggregate the tail into a single 'Other'
    head = s[s["cum_percent"] <= 80.0].copy()
    tail = s[s["cum_percent"] > 80.0].copy()

    # Sum any 'Other' in head with 'Other' from tail
    other_head = head[head["reason"] == "Other"]["count"].sum()
    other_tail = tail["count"].sum() if not tail.empty else 0
    # Remove 'Other' rows from head
    head = head[head["reason"] != "Other"]

    if other_head + other_tail > 0:
        head = pd.concat(
            [head, pd.DataFrame([{
                "reason": "Other",
                "count": int(other_head + other_tail),
                "percent": float((other_head + other_tail) * 100.0 / s["count"].sum()),
                "cum_percent": 100.0
            }])],
            ignore_index=True
        )

    head = head.sort_values("count", ascending=False).reset_index(drop=True)
    head["percent"] = head["percent"].round(1)
    head["cum_percent"] = head["cum_percent"].round(1)
    return head, latest


# =========================
# Plots
# =========================
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
    fig, ax = plt.subplots(figsize=(7.0, 3.6))
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


# =========================
# Streamlit entrypoint
# =========================
def run(store: Dict, params: Dict, user_text: str = "") -> Tuple[str, pd.DataFrame]:
    try:
        df_raw, cmap = _load_fpa()
    except FileNotFoundError as e:
        st.error(str(e));  return ("", pd.DataFrame())
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

    # MoM + table
    c1, c2 = st.columns((1.1, 1.0), gap="large")
    with c1:
        st.pyplot(_fig_mom(mom, f"First-Pass Accuracy — Jan–{pd.Period(latest).strftime('%b %y')}"))
    with c2:
        st.markdown(f"<h4 style='color:{_DARK_BLUE};margin:0 0 .5rem 0;'>Pass % by Portfolio × Scheme — {pd.Period(latest).strftime('%b-%y')}</h4>", unsafe_allow_html=True)
        if not table.empty:
            st.dataframe(table, use_container_width=True)

    # Reasons latest month — chart + table
    reasons, lastp = _reasons_latest(df_raw)
    st.markdown(f"<h4 style='color:{_DARK_BLUE};margin:1rem 0 .5rem 0;'>Reasons for Fail — {pd.Period(lastp).strftime('%b-%y')}</h4>", unsafe_allow_html=True)
    r1, r2 = st.columns(2, gap="large")
    with r1:
        if not reasons.empty:
            st.pyplot(_fig_reasons_bar(reasons[["reason", "count"]], "Fail reasons — Pareto (top 80% + Other)"))
    with r2:
        if not reasons.empty:
            st.dataframe(reasons, use_container_width=True)

    return ("", pd.DataFrame())
