# -*- coding: utf-8 -*-
"""
First-Pass Accuracy analysis from Jan-2025 to the most recent month available.

- Reads the latest FirstPassAccuracy*.xlsx (robust finder).
- Activity Date -> monthly MoM from 2025-01..latest (missing months shown as 0%).
- Pass% by Portfolio x Scheme for the latest month.
- Reasons for Fail across Jan-2025..latest (OpenAI labelling if API is present; robust keyword fallback).
- No sidebar; clean charts (no gridlines; hidden y-axis; soft x-axis line); cached I/O and transforms.

This module renders directly to Streamlit.
"""

from __future__ import annotations
import os, re, glob, math
from io import BytesIO
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# Optional OpenAI (transparent fallback to keyword rules if missing)
_OPENAI_READY = False
try:
    import openai  # type: ignore
    if os.getenv("OPENAI_API_KEY"):
        openai.api_key = os.getenv("OPENAI_API_KEY")
        _OPENAI_READY = True
except Exception:
    _OPENAI_READY = False

# -----------------------------
# Styling helpers
# -----------------------------
PALE_LINE = "#9fb7d1"
SOFT_GREY = "#cfcfcf"
DARK_BLUE = "#0a3b8f"
DARK_GREY = "#3c3c3c"

def _axis_clean(ax):
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    # keep x axis line as soft grey
    ax.spines["bottom"].set_color(SOFT_GREY)
    ax.tick_params(axis='y', which='both', left=False, labelleft=False)
    ax.tick_params(axis='x', colors=DARK_GREY, labelrotation=0)

# -----------------------------
# File loading & caching
# -----------------------------
@st.cache_data(show_spinner=False)
def _find_fpa_file() -> str | None:
    # Try app /repo paths
    search_roots = [
        "data",
        "first_pass_accuracy",
        ".",            # repo root (in case file is dropped here)
        "/mnt/data",    # user-bundled upload path
    ]
    patterns = ["FirstPassAccuracy*.xlsx", "FPA*.xlsx"]
    for root in search_roots:
        for pat in patterns:
            matches = sorted(glob.glob(os.path.join(root, pat)))
            if matches:
                # pick the most recently modified
                matches.sort(key=lambda p: os.path.getmtime(p), reverse=True)
                return matches[0]
    return None

@st.cache_data(show_spinner=False)
def _load_fpa_dataframe(fpath: str) -> pd.DataFrame:
    df = pd.read_excel(fpath)
    # expected columns (robust casing)
    cols = {c.lower(): c for c in df.columns}
    # normalize accessors
    def _col(name: str) -> str:
        for k, v in cols.items():
            if k.strip() == name.strip().lower():
                return v
        return name  # fallback to exact if already proper cased

    # Make a copy and standard fields
    df = df.copy()
    df["ActivityDate"] = pd.to_datetime(df[_col("Activity Date")], errors="coerce")
    df["ReviewResult"] = df[_col("Review Result")].astype(str).str.strip().str.lower()
    if "Portfolio" in df.columns:
        df["Portfolio"] = df["Portfolio"].astype(str).str.strip()
    else:
        # try common variant
        df["Portfolio"] = df[_col("portfolio")].astype(str).str.strip()
    if "Scheme" in df.columns:
        df["Scheme"] = df["Scheme"].astype(str).str.strip()
    else:
        df["Scheme"] = df[_col("scheme")].astype(str).str.strip()

    # comments for reasons
    case_comment_col = cols.get("case comment") or cols.get("comments") or _col("Case Comment")
    df["CaseComment"] = df.get(case_comment_col, pd.Series([""] * len(df))).fillna("").astype(str)

    # A canonical pass flag
    df["is_pass"] = df["ReviewResult"].str.contains(r"pass", na=False)
    return df

# -----------------------------
# Month prep: Jan-2025 to latest
# -----------------------------
def _month_key(dt: pd.Series) -> pd.Series:
    return dt.dt.to_period("M")

def _ensure_jan2025_to_latest(series: pd.Series, values: pd.Series) -> pd.DataFrame:
    # series: month Period[M], values: pass%
    if series.empty:
        idx = pd.period_range("2025-01", "2025-01", freq="M")
        return pd.DataFrame({"month": idx, "pass_pct": [0.0]})

    first = pd.Period("2025-01", freq="M")
    last  = series.max()
    if last < first:
        last = first
    idx = pd.period_range(first, last, freq="M")
    s = (
        pd.Series(values.values, index=series.values)
        .groupby(level=0).mean()
        .reindex(idx)
        .fillna(0.0)
    )
    return pd.DataFrame({"month": s.index, "pass_pct": s.values})

# -----------------------------
# Reasons labeller (OpenAI then rules)
# -----------------------------
@st.cache_data(show_spinner=False)
def _load_keyword_rules() -> List[Tuple[str, re.Pattern]]:
    # You can externalize to YAML if present, else fallback to these curated rules
    # rule order matters; first match wins
    rules = [
        ("Bank/Payment issue", re.compile(r"\b(bacs|bank|payment|payrun|credit|debit)\b", re.I)),
        ("Postal delay", re.compile(r"\b(post|postal|royal mail|mailroom)\b", re.I)),
        ("Manual calculation", re.compile(r"\b(recalc|re-calc|manual calc|recalculation)\b", re.I)),
        ("Trustee", re.compile(r"\btrustee\b", re.I)),
        ("AVC", re.compile(r"\bavc\b", re.I)),
        ("Data entry error", re.compile(r"\bkeying|data entry|typo|transpos|wrong field\b", re.I)),
        ("Death benefits payout", re.compile(r"\bdeath|bereave|deceased\b", re.I)),
        ("Case not created", re.compile(r"\bcase not created|missing case|no case\b", re.I)),
        ("Pension set up", re.compile(r"\b(pension set ?up|scheme set ?up|onboarding)\b", re.I)),
        ("Actuarial info required", re.compile(r"\bactuar|factor|commutation|gilt\b", re.I)),
        ("Correspondence/letter required", re.compile(r"\bletter|correspondence|draft\b", re.I)),
        ("Workflow/admin routing", re.compile(r"\bassign|route|workbasket|queue\b", re.I)),
        ("Information review/clarification", re.compile(r"\breview info|clarify|chase info|await info\b", re.I)),
        ("Calculation correction", re.compile(r"\bcalc(ulation)? error|wrong calc|rework\b", re.I)),
        ("Delay", re.compile(r"\bdelay|sla|overdue|late\b", re.I)),
        ("Procedure", re.compile(r"\bprocedure|process step|checklist\b", re.I)),
        ("Communication", re.compile(r"\bcommunicat|email|phone|contact\b", re.I)),
        ("Incorrect/incomplete information", re.compile(r"\bincorrect|incomplete|missing info\b", re.I)),
        ("System", re.compile(r"\bsystem|it issue|bug|defect\b", re.I)),
    ]
    return rules

def _label_reason_openai(texts: List[str]) -> List[str]:
    """
    Optional GPT-powered labeller (compact prompt).
    Falls back to rules if API is absent / errors.
    """
    try:
        if not _OPENAI_READY or not texts:
            raise RuntimeError("OpenAI not active")
        categories = [
            "Delay", "Procedure", "Communication", "System",
            "Incorrect/incomplete information", "Bank/Payment issue",
            "Manual calculation", "Postal delay", "Trustee", "AVC",
            "Data entry error", "Death benefits payout", "Case not created",
            "Pension set up", "Actuarial info required", "Correspondence/letter required",
            "Workflow/admin routing", "Information review/clarification",
            "Calculation correction", "Other"
        ]
        cat_str = ", ".join(categories)
        out = []
        for chunk_start in range(0, len(texts), 20):
            chunk = texts[chunk_start:chunk_start+20]
            prompt = (
                "Categorize each case comment into one of: "
                f"{cat_str}. Use only one label per row; if unclear, 'Other'.\n\n"
                + "\n".join([f"- {t}" for t in chunk])
            )
            resp = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role":"user","content": prompt}],
                temperature=0.0,
            )
            raw = resp["choices"][0]["message"]["content"]
            # Split lines; map 1:1
            lines = [ln.strip("- ").strip() for ln in raw.splitlines() if ln.strip()]
            # naive alignment; if counts mismatch, fall back to rules for that chunk
            if len(lines) != len(chunk):
                raise RuntimeError("Mismatch from model")
            out.extend(lines)
        # sanity-fix out-of-vocab
        fixed = [r if r in categories else "Other" for r in out]
        return fixed
    except Exception:
        # fallback to rules
        rules = _load_keyword_rules()
        def lab(txt: str) -> str:
            for name, pat in rules:
                if pat.search(txt or ""):
                    return name
            return "Other"
        return [lab(t) for t in texts]

# -----------------------------
# Core render
# -----------------------------
def run():
    # No sidebar. Everything inline.
    fpath = _find_fpa_file()
    if not fpath or not os.path.exists(fpath):
        st.error("Could not find a FirstPassAccuracy workbook (FirstPassAccuracy*.xlsx).")
        return

    df = _load_fpa_dataframe(fpath)
    # Filter to >= Jan 2025 only
    df = df.loc[df["ActivityDate"] >= pd.Timestamp(2025,1,1)].copy()

    if df.empty:
        st.warning("No rows on/after Jan-2025 in the FPA file.")
        return

    # ---------------- MoM pass% (Jan-2025 .. latest) ----------------
    df["month"] = _month_key(df["ActivityDate"])
    monthly = (
        df.groupby("month", as_index=False)
          .agg(cases=("is_pass","count"), passed=("is_pass","sum"))
    )
    monthly["pass_pct"] = np.where(monthly["cases"]>0,
                                   monthly["passed"]/monthly["cases"]*100.0,
                                   0.0)
    mom = _ensure_jan2025_to_latest(monthly["month"], monthly["pass_pct"])

    st.markdown(f"### First-Pass Accuracy — Jan–{mom['month'].iloc[-1].strftime('%b-%y')}")
    with st.container():
        fig, ax = plt.subplots(figsize=(8,1.8), dpi=140)
        x = [m.to_timestamp() for m in mom["month"]]
        y = mom["pass_pct"].astype(float).values
        ax.plot(x, y, marker="o", linewidth=2.5, color=PALE_LINE)
        # labels on points
        for xi, yi in zip(x, y):
            ax.text(xi, yi, f"{yi:.1f}%", color=DARK_GREY, fontsize=8, ha="center", va="bottom")
        ax.set_xticks(x)
        ax.set_xticklabels([pd.to_datetime(v).strftime("%b-%y") for v in x], fontsize=9, rotation=0, color=DARK_GREY)
        _axis_clean(ax)
        st.pyplot(fig, use_container_width=True)

    # ---------------- Latest month — pass% by portfolio × scheme ----------------
    latest_month = df["month"].max()
    dfx = df.loc[df["month"].eq(latest_month)].copy()
    by_ps = (
        dfx.groupby(["Portfolio","Scheme"], as_index=False)
           .agg(cases=("is_pass","count"), passed=("is_pass","sum"))
    )
    by_ps["pass_pct"] = np.where(by_ps["cases"]>0, by_ps["passed"]/by_ps["cases"]*100.0, 0.0)
    by_ps = by_ps.sort_values(["Portfolio","pass_pct","cases"], ascending=[True,False,False], kind="mergesort")
    by_ps["pass_pct"] = by_ps["pass_pct"].round(1)

    st.markdown(f"#### Pass % by Portfolio × Scheme — {latest_month.strftime('%b-%y')}")
    st.dataframe(
        by_ps.rename(columns={"pass_pct":"pass_%"}),
        hide_index=True,
        use_container_width=True
    )

    # ---------------- Reasons for fail — Jan-2025..latest ----------------
    fails = df.loc[~df["is_pass"]].copy()
    fails["Reason"] = _label_reason_openai(fails["CaseComment"].astype(str).tolist())

    reason_counts = fails.groupby("Reason", as_index=False).size().rename(columns={"size":"count"})
    reason_counts = reason_counts.sort_values("count", ascending=False)
    total_fail = int(reason_counts["count"].sum()) if not reason_counts.empty else 0
    reason_counts["percent"] = np.where(total_fail>0, (reason_counts["count"]/total_fail*100.0), 0.0)
    reason_counts["cum_percent"] = reason_counts["percent"].cumsum()

    # Pareto top 80 table
    top80 = reason_counts.loc[reason_counts["cum_percent"]<=80.0].copy()
    if top80.empty and not reason_counts.empty:
        # always show at least 1
        top80 = reason_counts.head(1).copy()

    left,right = st.columns([1,1])
    with left:
        st.markdown(f"#### Reasons for Fail — Jan-25 to {latest_month.strftime('%b-%y')} (counts)")
        # vertical bars with clean look
        if not reason_counts.empty:
            fig2, ax2 = plt.subplots(figsize=(7,2.5), dpi=140)
            ax2.bar(reason_counts["Reason"], reason_counts["count"], color="#9ecae1")
            ax2.set_xticklabels(reason_counts["Reason"].tolist(), rotation=90, ha="center", color=DARK_GREY)
            _axis_clean(ax2)
            st.pyplot(fig2, use_container_width=True)
        else:
            st.info("No failed items in the selected period.")

    with right:
        st.markdown(f"#### Reason breakdown (top 80%) — Jan-25 to {latest_month.strftime('%b-%y')}")
        if not top80.empty:
            tbl = top80.copy()
            tbl["percent"] = tbl["percent"].round(1)
            tbl["cum_percent"] = tbl["cum_percent"].round(1)
            st.dataframe(tbl.rename(columns={"cum_percent":"cum_%"}), hide_index=True, use_container_width=True)
        else:
            st.info("No reasons to display.")

    # Footer note
    st.caption(f"Source: {os.path.basename(fpath)}  •  Months with no cases are shown as 0% in the MoM trend.")
