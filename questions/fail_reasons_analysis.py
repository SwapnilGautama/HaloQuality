from __future__ import annotations

import os
from typing import Dict, Any

import pandas as pd
import streamlit as st
import plotly.express as px

# Prefer your shared labeller if present (OpenAI optional).
try:
    from core.reason_labeller import classify_comments, OPENAI_READY as _OPENAI_READY
except Exception:
    _OPENAI_READY = False

    def classify_comments(series: pd.Series, use_openai: bool | None = None, sample_cap: int = 220) -> pd.Series:
        """
        Lightweight fallback keyword model (keeps the module working if core.reason_labeller
        hasn't been added yet). You can remove this once core.reason_labeller exists.
        """
        import re
        CATS = {
            "Communication / update": [r"update", r"chase", r"follow[- ]?up", r"await", r"waiting",
                                       r"no response", r"respond", r"email", r"call", r"letter", r"inform"],
            "Data entry / setup":    [r"data (entry|issue|error)", r"setup", r"record", r"index", r"scan",
                                       r"document", r"upload", r"capture", r"input"],
            "Bank / payment":        [r"payment", r"bank", r"bacs", r"cheque", r"refund", r"overpayment",
                                       r"underpayment", r"remit"],
            "Trustee / AVC":         [r"trustee", r"\bavc\b", r"additional contribution"],
            "Postal / dispatch":     [r"post(al)?", r"dispatch", r"mail", r"courier", r"deliver"],
            "Manual calculation":    [r"manual (calc|calculation)", r"hand[- ]?calc", r"complex"],
            "System":                [r"system (issue|error|down)", r"access", r"permission", r"bug", r"crash"],
            "Waiting on member/TPA": [r"waiting on (member|tpa|third party)"],
        }
        RX = {k: re.compile("|".join(v), re.I) for k, v in CATS.items()}

        def _hit(x: str) -> str:
            x = str(x or "")
            for k, r in RX.items():
                if r.search(x):
                    return k
            return "Other"

        return series.astype(str).map(_hit)

def _load_fpa(root: str) -> pd.DataFrame:
    """
    Loads FirstPassAccuracy workbook without assuming exact filename,
    keeping your previous behaviour.
    """
    # look in data/first_pass_accuracy then data/
    search_dirs = [os.path.join(root, "data", "first_pass_accuracy"), os.path.join(root, "data"), root]
    excel_path = None
    for d in search_dirs:
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            if f.lower().startswith("firstpassaccuracy") and f.lower().endswith(".xlsx"):
                excel_path = os.path.join(d, f)
    if not excel_path:
        st.error("Could not find a FirstPassAccuracy workbook (FirstPassAccuracy*.xlsx).")
        return pd.DataFrame()

    try:
        df = pd.read_excel(excel_path)
    except Exception as e:
        st.error(f"Failed to read workbook: {excel_path}\n{e}")
        return pd.DataFrame()

    # Normalize core columns: Activity Date, Review Result, Portfolio, Scheme, Case Comment
    cols = {c.lower().strip(): c for c in df.columns}
    def pick(name: str) -> str:
        n = name.lower().strip()
        return cols.get(n, name)
    act = pick("Activity Date")
    res = pick("Review Result")
    port = pick("Portfolio")
    scheme = pick("Scheme")
    comm = pick("Case Comment")

    for c in [act, res, port, scheme, comm]:
        if c in df.columns and c == act:
            df[c] = pd.to_datetime(df[c], errors="coerce")
        elif c in df.columns:
            df[c] = df[c].astype(str)

    keep = [c for c in [act, res, port, scheme, comm] if c in df.columns]
    return df[keep].rename(columns={act: "Activity Date",
                                    res: "Review Result",
                                    port: "Portfolio",
                                    scheme: "Scheme",
                                    comm: "Case Comment"}).copy()

def _month_floor(s: pd.Series) -> pd.Series:
    return (s.values.astype("datetime64[M]")).astype("datetime64[ns]")

def _pareto_top80(counts: pd.Series) -> pd.DataFrame:
    if counts.empty:
        return pd.DataFrame(columns=["reason", "count", "percent", "cum_percent"])
    df = counts.reset_index()
    df.columns = ["reason", "count"]
    df = df.sort_values("count", ascending=False).reset_index(drop=True)
    total = df["count"].sum()
    df["percent"] = (df["count"] / total) * 100
    df["cum_percent"] = df["percent"].cumsum()

    head = df[df["cum_percent"] <= 80.0]
    tail = df[df["cum_percent"] > 80.0]
    if not tail.empty:
        other = pd.DataFrame([["Other", tail["count"].sum(),
                               (tail["count"].sum() / total) * 100, 100.0]],
                             columns=["reason", "count", "percent", "cum_percent"])
        out = pd.concat([head, other], ignore_index=True)
    else:
        out = df.copy()
    return out

def run(store: Dict[str, Any], params: Dict[str, Any], q: str) -> None:
    """
    Q3: Fail Reasons Analysis (FRA). Self-contained, does not alter Q1 or Q2.
    """
    root = store.get("root", os.getcwd())
    df = _load_fpa(root)
    if df.empty:
        return

    # Month selection
    df["_month"] = _month_floor(df["Activity Date"])
    months = df["_month"].dropna().sort_values().unique()
    if len(months) == 0:
        st.info("No dated rows in the workbook.")
        return

    # UI – month select (default: latest)
    sel = st.selectbox("Choose month", options=list(months), index=len(months)-1,
                       format_func=lambda d: pd.to_datetime(d).strftime("%b-%y"))

    work = df[df["_month"].eq(sel)].copy()
    fails = work[work["Review Result"].str.lower().eq("fail")].copy()

    st.subheader(f"Fail reasons — {pd.to_datetime(sel).strftime('%b-%y')}")

    if not _OPENAI_READY:
        st.info("OpenAI labelling inactive or not configured. Using keyword model; results may include 'Other'.")

    if fails.empty:
        st.info("No failed cases for the chosen month.")
        return

    # Classify
    reasons = classify_comments(fails["Case Comment"])
    counts = reasons.value_counts(dropna=False)

    # Pareto
    pareto = _pareto_top80(counts)

    # Side by side
    c1, c2 = st.columns([3, 2], gap="large")
    with c1:
        fig = px.bar(
            pareto, x="reason", y="count", text="count",
            labels={"reason": "", "count": ""},
            title="Fail reasons — Pareto (top 80% + Other)"
        )
        fig.update_traces(textposition="outside")
        fig.update_yaxes(showgrid=False, visible=False)
        fig.update_xaxes(showgrid=False)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.dataframe(
            pareto[["reason", "count", "percent", "cum_percent"]],
            use_container_width=True,
            height=380
        )

    # Small audit list (optional)
    with st.expander("Show a small sample of classified comments"):
        sample = fails[["Case Comment"]].copy()
        sample["reason"] = reasons.values
        st.dataframe(sample.sample(min(40, len(sample))), use_container_width=True, height=320)
