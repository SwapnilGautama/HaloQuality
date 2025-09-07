# questions/first_pass_accuracy.py
from __future__ import annotations
import os, glob, re
from io import BytesIO
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# Optional OpenAI assist
_OPENAI_READY = False
try:
    import openai  # type: ignore
    if os.getenv("OPENAI_API_KEY"):
        openai.api_key = os.getenv("OPENAI_API_KEY")
        _OPENAI_READY = True
except Exception:
    _OPENAI_READY = False

# ------------------------------
# Fast loaders (cached)
# ------------------------------
@st.cache_data(show_spinner=False)
def _load_fpa() -> pd.DataFrame:
    """Load FPA data quickly (selected columns), from repo folder or uploaded path."""
    cols = ["Activity Date", "Review Result", "Portfolio", "Scheme", "Case Comment"]
    frames = []

    # 1) project data folder
    for path in glob.glob("data/first_pass_accuracy/*.xlsx"):
        try:
            frames.append(pd.read_excel(path, usecols=cols))
        except Exception:
            pass

    # 2) uploaded dev file as fallback
    alt_path = "/mnt/data/FirstPassAccuracy_Aug'25.xlsx"
    if not frames and os.path.exists(alt_path):
        try:
            frames.append(pd.read_excel(alt_path, usecols=cols))
        except Exception:
            pass

    if not frames:
        return pd.DataFrame(columns=cols)

    df = pd.concat(frames, ignore_index=True)
    # Normalize
    df.rename(columns={
        "Activity Date": "activity_date",
        "Review Result": "review_result",
        "Portfolio": "portfolio",
        "Scheme": "scheme",
        "Case Comment": "case_comment",
    }, inplace=True)
    df["activity_date"] = pd.to_datetime(df["activity_date"], errors="coerce")
    df.dropna(subset=["activity_date"], inplace=True)
    df["review_result"] = df["review_result"].astype(str).str.strip().str.lower()
    df["portfolio"] = df["portfolio"].astype(str).str.strip()
    df["scheme"] = df["scheme"].astype(str).str.strip()
    df["case_comment"] = df["case_comment"].astype(str).fillna("").str.strip()
    return df

# ------------------------------
# Helpers: pass flags & month index
# ------------------------------
def _is_pass(s: pd.Series) -> pd.Series:
    # treat anything beginning with 'pass' as pass; else fail
    return s.str.startswith("pass")

def _month_span_2025_jan_jun() -> pd.PeriodIndex:
    return pd.period_range("2025-01", "2025-06", freq="M")

# ------------------------------
# Keyword-based deeper reason mapping (robust fallback)
# ------------------------------
_RULES = [
    ("Bank/Payment issue", r"bank|bacs|payment|direct debit|refund|remittance"),
    ("Postal delay", r"post|postal|royal mail|mail delay"),
    ("Manual calculation", r"manual calc|manual calculation|calc error|calculation"),
    ("Data entry error", r"data entry|keying|typo|mis-key|miskey|mis-keyed"),
    ("Case not created", r"case not created|case missing|no case|uncreated"),
    ("Second review", r"second review|2nd review|review 2"),
    ("Trustee", r"trustee"),
    ("AVC", r"\bavc\b|additional voluntary"),
    ("Pension set up", r"setup|set up|pension set up|initialise|initialization"),
    ("Scheme Rules", r"scheme rule|rule breach|rules|scheme rules"),
    ("Timescale", r"timescale|sla|service level|delay requirement|deadline"),
    ("Holding Letter", r"holding letter"),
    ("Incorrect/Incomplete information", r"incorrect|incomplete|missing info|lack of clarity|clarity|form incomplete"),
    ("System", r"system|it issue|tech(nical)?|glitch"),
    ("Overpayment", r"overpayment"),
]

def _map_reason_keyword(txt: str) -> str:
    t = txt.lower()
    for label, pat in _RULES:
        if re.search(pat, t):
            return label
    return "Other"

# Optional OpenAI-assisted classification for reason (batched)
def _map_reasons_openai(texts: list[str]) -> Dict[str, str]:
    """Return {original_text: label}. Only used when API key is present; otherwise empty."""
    if not _OPENAI_READY or not texts:
        return {}
    system = (
        "You are classifying complaint review failure reasons into high-level buckets. "
        "Pick ONE label from this set (case-sensitive): "
        "[Bank/Payment issue, Postal delay, Manual calculation, Data entry error, Case not created, "
        "Second review, Trustee, AVC, Pension set up, Scheme Rules, Timescale, Holding Letter, "
        "Incorrect/Incomplete information, System, Overpayment, Other]. "
        "Return ONLY the label."
    )
    out: Dict[str, str] = {}
    # Simple mini-batching to stay snappy
    BATCH = 40
    for i in range(0, len(texts), BATCH):
        chunk = texts[i:i+BATCH]
        try:
            # One-shot: include short guidance + items as bullet points
            prompt = "Classify each bullet point into ONE label:\n" + "\n".join(f"- {t}" for t in chunk)
            rsp = openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role":"system","content":system},
                          {"role":"user","content":prompt}],
                temperature=0.0,
            )
            answer = rsp.choices[0].message.content.strip()
            # Heuristic parsing: read back line-per-line labels
            labels = [ln.strip("-• ").strip() for ln in answer.splitlines() if ln.strip()]
            # Align by index; if mismatch, fall back to keyword
            for j, txt in enumerate(chunk):
                if j < len(labels) and labels[j] in [r[0] for r in _RULES] + ["Other"]:
                    out[txt] = labels[j]
                else:
                    out[txt] = _map_reason_keyword(txt)
        except Exception:
            for txt in chunk:
                out[txt] = _map_reason_keyword(txt)
    return out

# ------------------------------
# Precompute & cache processed outputs for speed
# ------------------------------
@st.cache_data(show_spinner=False)
def _precompute(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      monthly_pass (PeriodIndex Jan-Jun 2025 with pass%), 
      june_table (pass% by portfolio x scheme for June), 
      reasons (reason counts for fails in June)
    """
    if df.empty:
        months = _month_span_2025_jan_jun()
        monthly_pass = pd.DataFrame({"month": months, "pass_pct": 0.0}).set_index("month")
        return monthly_pass, pd.DataFrame(), pd.DataFrame()

    df["is_pass"] = _is_pass(df["review_result"])
    df["month"] = df["activity_date"].dt.to_period("M")
    # Keep Jan-Jun 2025
    wanted = _month_span_2025_jan_jun()
    df = df[df["month"].isin(wanted)]

    # Monthly pass% (0s for missing)
    monthly = df.groupby("month", as_index=True)["is_pass"].mean().mul(100)
    monthly = monthly.reindex(wanted, fill_value=0.0).to_frame("pass_pct")

    # June pass% by portfolio x scheme
    june = df[df["month"] == pd.Period("2025-06", freq="M")].copy()
    if june.empty:
        june_table = pd.DataFrame(columns=["portfolio","scheme","cases","pass_pct"])
        reasons = pd.DataFrame(columns=["reason","count","percent","cum_percent"])
        return monthly, june_table, reasons

    grp = june.groupby(["portfolio","scheme"])
    june_table = grp["is_pass"].agg(["count","mean"]).reset_index()
    june_table.rename(columns={"count":"cases","mean":"pass_pct"}, inplace=True)
    june_table["pass_pct"] = (june_table["pass_pct"] * 100).round(1)
    june_table = june_table.sort_values(["portfolio","scheme"])

    # Reasons (fails only, June)
    fails = june.loc[~june["is_pass"], ["case_comment"]].copy()
    if fails.empty:
        reasons = pd.DataFrame(columns=["reason","count","percent","cum_percent"])
        return monthly, june_table, reasons

    uniq = fails["case_comment"].dropna().astype(str).str.strip().unique().tolist()

    labels_map: Dict[str, str] = {}
    # 1) OpenAI assist (optional)
    if _OPENAI_READY:
        try:
            labels_map = _map_reasons_openai(uniq)
        except Exception:
            labels_map = {}
    # 2) Keyword fallback
    for t in uniq:
        if t not in labels_map:
            labels_map[t] = _map_reason_keyword(t)

    fails["reason"] = fails["case_comment"].map(labels_map).fillna("Other")
    rc = fails.groupby("reason").size().reset_index(name="count").sort_values("count", ascending=False)
    rc["percent"] = (rc["count"] / rc["count"].sum() * 100).round(1)
    rc["cum_percent"] = rc["percent"].cumsum().round(1)

    return monthly, june_table, rc

# ------------------------------
# Drawing helpers (charts with no grid & no y-axis)
# ------------------------------
def _style_axes(ax):
    ax.grid(False)
    ax.set_ylabel("")
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    # keep soft grey x axis
    ax.spines["bottom"].set_color("#e0e0e0")
    ax.spines["bottom"].set_linewidth(1.2)

def _plot_monthly_pass(monthly: pd.DataFrame):
    months = monthly.index.astype(str)  # '2025-01'...
    # Pretty x labels MMM-YY
    xlabels = pd.PeriodIndex(months, freq="M").to_timestamp().strftime("%b-%y")

    fig, ax = plt.subplots(figsize=(7, 2.4))
    ax.plot(range(len(months)), monthly["pass_pct"].values, marker="o", linewidth=2.5)
    for i, v in enumerate(monthly["pass_pct"].values):
        ax.text(i, v, f"{v:.1f}%", ha="center", va="bottom", fontsize=9)
    _style_axes(ax)
    ax.set_xticks(range(len(months)))
    ax.set_xticklabels(xlabels, rotation=0)
    st.pyplot(fig, use_container_width=True)

def _plot_reasons(rc: pd.DataFrame):
    if rc.empty:
        st.info("No failed cases in June 2025.")
        return
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.bar(rc["reason"], rc["count"], width=0.6)
    _style_axes(ax)
    ax.set_xticklabels(rc["reason"], rotation=90)
    st.pyplot(fig, use_container_width=True)

# ------------------------------
# Page run()
# ------------------------------
def run(store, params: Dict, user_q: str):
    st.markdown("### First-Pass Accuracy — Jan–Jun 2025", unsafe_allow_html=False)

    df = _load_fpa()
    monthly, june_table, rc = _precompute(df)

    # Summary row
    rows = len(df)
    passed = int(df["review_result"].str.startswith("pass").sum())
    overall = (passed / rows * 100) if rows else 0.0
    st.caption(f"Rows={rows:,} · Passed={passed:,} · Overall={overall:.1f}%")

    left, right = st.columns([1.1, 1.2])
    with left:
        st.write("##### Pass % — MoM")
        _plot_monthly_pass(monthly)

    with right:
        st.write("##### Pass % by Portfolio × Scheme — Jun-25")
        if june_table.empty:
            st.info("No June 2025 data.")
        else:
            jt = june_table.rename(columns={"pass_pct":"pass_%"})
            st.dataframe(
                jt[["portfolio","scheme","cases","pass_%"]],
                use_container_width=True,
                hide_index=True
            )

    st.write("---")
    c1, c2 = st.columns([1.1, 1.2])
    with c1:
        st.write("##### Reasons for Fail — Jun 2025 (counts)")
        _plot_reasons(rc)
    with c2:
        st.write("##### Reason breakdown (top 80%) — Jun 2025")
        if rc.empty:
            st.info("No failed cases.")
        else:
            top = rc.copy()
            st.dataframe(
                top[["reason","count","percent","cum_percent"]],
                use_container_width=True,
                hide_index=True
            )
