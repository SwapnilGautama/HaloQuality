# -*- coding: utf-8 -*-
# questions/first_pass_accuracy.py
from __future__ import annotations

import os
import re
from glob import glob
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------------------
# Optional OpenAI (classification works fine without it).
# --------------------------------------------------------------------------------------
_OPENAI_READY = False
try:
    import openai  # type: ignore
    if os.getenv("OPENAI_API_KEY"):
        openai.api_key = os.getenv("OPENAI_API_KEY")
        _OPENAI_READY = True
except Exception:
    _OPENAI_READY = False

# --------------------------------------------------------------------------------------
# Display / Pareto tuning (overridable via env without code changes)
# --------------------------------------------------------------------------------------
PARETO_THRESH = float(os.getenv("FPA_PARETO", "0.90"))  # keep ~90% by default
MIN_HEAD = int(os.getenv("FPA_MIN_CATEGORIES", "6"))    # show at least 6 categories
MAX_HEAD = int(os.getenv("FPA_MAX_CATEGORIES", "10"))   # but no more than 10

# --------------------------------------------------------------------------------------
# Data loading helpers
# --------------------------------------------------------------------------------------
def _root(store: Dict) -> str:
    return store.get("root", ".")

def _find_workbook(root: str) -> Optional[str]:
    patterns = [
        os.path.join(root, "data", "first_pass_accuracy", "FirstPassAccuracy_*.xlsx"),
        os.path.join(root, "data", "first_pass_accuracy", "FirstPassAccuracy*.xlsx"),
        os.path.join(root, "data", "first_pass_accuracy", "*FirstPass*.xlsx"),
    ]
    cand: List[str] = []
    for p in patterns: cand.extend(glob(p))
    if not cand: return None
    cand.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return cand[0]

def _is_pass(x) -> bool:
    if x is None: return False
    s = str(x).strip().lower()
    return s.startswith("pass") or s in {"y", "yes", "correct", "ok"}

def _preprocess(df: pd.DataFrame) -> pd.DataFrame:
    colmap = {
        "Activity Date": "date", "activity date": "date",
        "Review Result": "result", "review result": "result",
        "Portfolio": "portfolio", "portfolio": "portfolio",
        "Scheme": "scheme", "scheme": "scheme",
        "Case Comment": "comment", "case comment": "comment",
    }
    df = df.rename(columns={c: colmap.get(c, c) for c in df.columns})
    for w in ["date", "result", "portfolio", "scheme", "comment"]:
        if w not in df.columns: df[w] = np.nan
    df["date"] = pd.to_datetime(df["date"], errors="coerce", dayfirst=True)
    df["_m"] = df["date"].dt.to_period("M")
    for c in ["result","portfolio","scheme","comment"]:
        df[c] = df[c].astype(str).fillna("")
    return df

# --------------------------------------------------------------------------------------
# Rule-based classifier (works offline). OpenAI layer is optional.
# --------------------------------------------------------------------------------------
_RULES: Dict[str, List[str]] = {
    "Communication / update": [
        "update","awaiting update","follow up","chase","chasing","no response","no reply",
        "not responded","email not received","phone not answered","letter sent","awaiting reply",
        "clarification","query","enquiry","documentation sent","reminder","communication",
        "incorrect/unclear instruction","ambiguous instruction",
    ],
    "Data entry / setup": [
        "data entry","keying","keystroke","setup","set up","configured","coding","captured wrong",
        "entered wrong","spelling","mismatch","wrong details","ni number","national insurance",
        "dob mismatch","date of birth","address mismatch","update address","postcode","postal code",
        "member record","record not found","search issue",
    ],
    "Bank / payment": [
        "payment","paid","bank","bacs","chaps","disinvest","disinvestment","cheque","refund",
        "overpayment","underpayment","account","sort code",
    ],
    "Trustee / AVC": [
        "trustee","avc","other 3rd party","third party","employer approval","waiting trustee",
        "trustee approval","adviser","ifa","actuary",
    ],
    "Postal / dispatch": [
        "post","postal","royal mail","dispatch","despatch","scanning","mailroom","envelope",
        "returned mail",
    ],
    "Manual calculation": [
        "manual calc","manual calculation","calc error","calculation error","manual check",
        "2nd review","second review","recalculated","rework",
    ],
    "System": [
        "system","portal","workflow","it","bug","error code","timeout","script","automation",
        "apta","aptia","aptia standard timescale",
    ],
    "Waiting on member/TPA": [
        "waiting on member","awaiting member","member not responded","late notice","late notification",
        "waiting on tpa","third party info","awaiting info",
    ],
    "Documents / ID missing": [
        "id","identity","proof","document missing","missing document","certified","photo id",
        "passport","driving licence","birth certificate",
    ],
    "Scheme rules / interpretation": [
        "scheme rules","rules interpretation","factor","actuarial","drop in value","change in factor",
    ],
}

def _label_reason_rules(text: str) -> str:
    s = re.sub(r"[^a-z0-9\s/]+", " ", str(text).lower())
    for label, kws in _RULES.items():
        for kw in kws:
            if kw in s: return label
    return "Other"

def _ai_label_many(texts: List[str]) -> List[str]:
    if not _OPENAI_READY:
        return ["Other"] * len(texts)
    sys_prompt = (
        "You categorise pension case 'fail' comments into one of these labels:\n"
        f"{', '.join(_RULES.keys())}\n"
        "Pick the single best label only. If unclear, answer 'Other'."
    )
    out: List[str] = []
    for t in texts:
        try:
            resp = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role":"system","content":sys_prompt},
                          {"role":"user","content":t[:1500]}],
                max_tokens=8, temperature=0.0,
            )
            label = resp.choices[0].message["content"].strip()
            out.append(label if label in _RULES or label=="Other" else "Other")
        except Exception:
            out.append("Other")
    return out

# --------------------------------------------------------------------------------------
# Analytics
# --------------------------------------------------------------------------------------
def _pass_mom(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("_m", dropna=True)["result"]
    num = g.apply(lambda s: sum(_is_pass(x) for x in s))
    den = g.size()
    pct = (num / den * 100.0).fillna(0.0).rename("pass_%").reset_index()
    return pct.sort_values("_m").reset_index(drop=True)

def _pass_by_portfolio_scheme_latest(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Period]:
    latest = df["_m"].max()
    sub = df[df["_m"] == latest].copy()
    if sub.empty:
        return pd.DataFrame(columns=["portfolio","scheme","cases","pass_%"]), latest
    g = sub.groupby(["portfolio","scheme"], dropna=False)
    cases = g.size().rename("cases")
    passed = g["result"].apply(lambda s: sum(_is_pass(x) for x in s)).rename("passed")
    out = pd.concat([cases, passed], axis=1).reset_index()
    out["pass_%"] = (out["passed"]/out["cases"]*100.0).round(1)
    out = out.drop(columns=["passed"]).sort_values(["portfolio","scheme"]).reset_index(drop=True)
    return out, latest

def _reasons_latest(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Period]:
    latest = df["_m"].max()
    if pd.isna(latest):
        return pd.DataFrame(), latest
    fails = df[(df["_m"]==latest) & (~df["result"].apply(_is_pass))]
    if fails.empty or "comment" not in fails.columns:
        return pd.DataFrame(columns=["reason","count","percent","cum_percent"]), latest

    texts = fails["comment"].astype(str).fillna("").tolist()

    ai_labels = _ai_label_many(texts)
    labels = [lab if lab in _RULES or lab=="Other" else _label_reason_rules(t)
              for t, lab in zip(texts, ai_labels)]
    labels = [lbl if lbl!="Other" else _label_reason_rules(t) for lbl,t in zip(labels, texts)]

    s = pd.Series(labels).value_counts().rename_axis("reason").reset_index(name="count")
    s = s.sort_values("count", ascending=False).reset_index(drop=True)
    total = int(s["count"].sum()) or 1

    other_pre = int(s.loc[s["reason"]=="Other","count"].sum()) if "Other" in s["reason"].values else 0
    s_no_other = s[s["reason"]!="Other"].copy()
    s_no_other["percent"] = (s_no_other["count"]*100.0/total)
    s_no_other = s_no_other.sort_values("count", ascending=False).reset_index(drop=True)
    s_no_other["cum_percent"] = s_no_other["percent"].cumsum()

    head_pareto = s_no_other[s_no_other["cum_percent"] <= (PARETO_THRESH*100.0)]
    head_count = max(MIN_HEAD, len(head_pareto))
    head_count = min(head_count, MAX_HEAD, len(s_no_other))

    head = s_no_other.iloc[:head_count].copy()
    tail = s_no_other.iloc[head_count:].copy()
    other_total = int(tail["count"].sum()) + other_pre

    out = head[["reason","count"]].copy()
    if other_total > 0:
        out = pd.concat([out, pd.DataFrame([{"reason":"Other","count":other_total}])], ignore_index=True)

    out["percent"] = (out["count"]*100.0/total)
    out = out.sort_values("count", ascending=False).reset_index(drop=True)
    out["cum_percent"] = out["percent"].cumsum()
    out["percent"] = out["percent"].round(1)
    out["cum_percent"] = out["cum_percent"].round(1)
    return out, latest

# --------------------------------------------------------------------------------------
# Charts
# --------------------------------------------------------------------------------------
def _style_ax(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)
    return ax

def _plot_mom(df_mom: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8.5, 3.2))
    _style_ax(ax)
    x = pd.PeriodIndex(df_mom["_m"]).to_timestamp()
    y = df_mom["pass_%"].fillna(0.0).values
    ax.plot(x, y, marker="o", linewidth=2)
    ax.set_ylim(0, max(100, (int(np.nanmax(y)//10)+1)*10))
    for xi, yi in zip(x, y):
        ax.text(xi, yi + 1.5, f"{yi:.0f}%", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel(""); ax.set_xlabel("")
    return fig

def _plot_pareto(counts: pd.DataFrame, title: str):
    fig, ax = plt.subplots(figsize=(8.5, 4.0))
    _style_ax(ax)
    bars = ax.bar(counts["reason"], counts["count"])
    ax.set_title(title)
    ax.set_ylabel(""); ax.set_xlabel("")
    ax.set_ylim(0, max(10, int(counts["count"].max()*1.15)))
    ax.tick_params(axis="x", rotation=90)
    for b in bars:
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+ (0.01*ax.get_ylim()[1]),
                f"{int(b.get_height())}", ha="center", va="bottom", fontsize=9)
    return fig

# --------------------------------------------------------------------------------------
# Entry point expected by the app
# --------------------------------------------------------------------------------------
def run(store: Dict, params: Dict, q: str):
    wb = _find_workbook(_root(store))
    if not wb:
        st.error("Could not find a FirstPassAccuracy workbook (FirstPassAccuracy*.xlsx) in data/first_pass_accuracy.")
        return
    try:
        df_raw = pd.read_excel(wb)
    except Exception as e:
        st.error(f"Could not open workbook: {os.path.basename(wb)}\n\n{e}")
        return

    df = _preprocess(df_raw)

    mom = _pass_mom(df)
    if not mom.empty:
        t_start = mom["_m"].min().strftime("%b-%y")
        t_end = mom["_m"].max().strftime("%b-%y")
        st.subheader(f"First-Pass Accuracy — {t_start}–{t_end}")
        st.pyplot(_plot_mom(mom), use_container_width=True)
    else:
        st.info("No rows available to compute month-on-month pass %.")

    table, latest = _pass_by_portfolio_scheme_latest(df)
    if not table.empty:
        st.subheader(f"Pass % by Portfolio × Scheme — {latest.strftime('%b-%y')}")
        st.dataframe(table, use_container_width=True, hide_index=True)
    else:
        st.info("No latest-month rows found for portfolio × scheme table.")

    counts, latest2 = _reasons_latest(df)
    st.subheader(f"Reasons for Fail — {latest2.strftime('%b-%y') if pd.notna(latest2) else ''}")
    if counts.empty:
        st.info("No fail records found for the latest month.")
        return

    c1, c2 = st.columns([3, 2])
    with c1:
        st.pyplot(_plot_pareto(counts, "Fail reasons — Pareto (top 90% + Other)"), use_container_width=True)
    with c2:
        st.caption("Reason breakdown (top set + Other) — latest month")
        st.dataframe(counts, use_container_width=True, hide_index=True)
