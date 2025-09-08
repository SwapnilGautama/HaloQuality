# questions/fail_reasons_analysis.py
from __future__ import annotations

import os
import re
from glob import glob
from pathlib import Path
from typing import Dict, Optional, Tuple, List

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt


# --------------------------- helpers: resilient data access ---------------------------

def _first_df(df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    return df if isinstance(df, pd.DataFrame) and not df.empty else None


def _get_df_from_store(store: Dict) -> Optional[pd.DataFrame]:
    """
    Try the obvious keys we already use elsewhere. This keeps Q1/Q2 isolated.
    """
    for k in ("fpa", "fpa_df", "df_fpa", "first_pass_accuracy", "fpa_data"):
        if k in store and isinstance(store[k], pd.DataFrame):
            return _first_df(store[k])
    return None


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


def _rename_columns(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    Normalize headers to:
      - activity_date
      - review_result
      - case_comment
    Accept several common variations and rename in-place.
    """
    if df is None or df.empty:
        return None

    col_map = {_norm(c): c for c in df.columns}

    # candidates we will look for
    activity_candidates = ["activity_date", "activitydate", "date", "review_date", "activity_dt"]
    result_candidates = ["review_result", "result", "status", "decision"]
    comment_candidates = ["case_comment", "comment", "comments", "reason", "notes", "detail", "details"]

    def _pick(cands: List[str]) -> Optional[str]:
        for cand in cands:
            if cand in col_map:
                return col_map[cand]
        return None

    c_activity = _pick(activity_candidates)
    c_result   = _pick(result_candidates)
    c_comment  = _pick(comment_candidates)

    if not all([c_activity, c_result, c_comment]):
        return None  # missing essentials; let caller handle error

    df = df.rename(
        columns={
            c_activity: "activity_date",
            c_result: "review_result",
            c_comment: "case_comment",
        }
    )

    # Basic parsing
    df["activity_date"] = pd.to_datetime(df["activity_date"], errors="coerce")
    # normalize result text
    df["review_result"] = df["review_result"].astype(str).str.strip()

    # normalize comment (object, fillna)
    df["case_comment"] = df["case_comment"].astype(str).fillna("").str.strip()

    # Drop rows with no date
    df = df.dropna(subset=["activity_date"])

    return df


def _load_latest_excel(base: str = "data/first_pass_accuracy") -> Optional[pd.DataFrame]:
    """
    Load the most-recently modified Excel file under data/first_pass_accuracy.
    """
    candidates = []
    for pattern in ("*.xlsx", "**/*.xlsx"):
        candidates.extend(glob(str(Path(base) / pattern), recursive=True))

    if not candidates:
        return None

    # Pick newest
    latest = max(candidates, key=lambda p: os.path.getmtime(p))

    try:
        # heuristic: pick first sheet that contains any of the expected columns
        xls = pd.ExcelFile(latest, engine="openpyxl")
        chosen_df = None
        for sheet in xls.sheet_names:
            tmp = pd.read_excel(xls, sheet_name=sheet)
            if _rename_columns(tmp) is not None:
                chosen_df = _rename_columns(tmp)
                break
        return chosen_df
    except Exception:
        return None


def _get_fpa_df(store: Dict) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """
    Try store first; else load newest Excel from repo.
    Returns (df, error_message_if_any)
    """
    df = _get_df_from_store(store)
    if df is None:
        df = _load_latest_excel()

    if df is None:
        return None, "Could not find a First-Pass Accuracy dataset in the store or under data/first_pass_accuracy."

    df = _rename_columns(df)
    if df is None:
        return None, "Could not find the required columns (Activity Date, Review Result, Case Comment) in the dataset."

    return df, None


# --------------------------- fail reason classification (keyword model) ---------------

# broader, more actionable categories
_REASON_KEYWORDS = {
    "Communication / update": [
        r"\bemail\b", r"\bchase\b", r"\bfollow\s*up\b", r"\bupdate\b", r"\bcall(ed)?\b",
        r"\bcontact(ed)?\b", r"\bawaiting response\b", r"\bno reply\b", r"\brespons(e|es) pending\b"
    ],
    "Data entry / setup": [
        r"\bdata\s*entry\b", r"\bsetup\b", r"\bset[-\s]*up\b", r"\bform(s)? incomplete\b",
        r"\bmissing field(s)?\b", r"\bincorrect detail(s)?\b", r"\bkey(ed)? wrong\b",
    ],
    "Bank / payment": [
        r"\bpayment\b", r"\bbacs\b", r"\bbank\b", r"\btransfer\b", r"\bcheque\b", r"\brefund\b"
    ],
    "Trustee / AVC": [
        r"\btrustee\b", r"\bavc\b", r"\baditional voluntary\b", r"\btrust(ee)? approval\b"
    ],
    "Postal / dispatch": [
        r"\bpost(al)?\b", r"\bdispatch\b", r"\bmail(ed)?\b", r"\breturned mail\b", r"\baddressed\b"
    ],
    "Manual calculation": [
        r"\bmanual calc(ulation)?\b", r"\bmanually\b", r"\bcalc(ulation)? required\b"
    ],
    "System": [
        r"\bsystem\b", r"\bit issue\b", r"\boutage\b", r"\bbug\b", r"\btech(nical)?\b", r"\bportal\b"
    ],
    "Waiting on member/TPA": [
        r"\bwaiting on\b", r"\bawaiting\b", r"\bmember\b", r"\btpa\b", r"\b3(rd)?\s*party\b",
    ],
    "Missing documents": [
        r"\bmissing doc(ument)?s?\b", r"\bid(proof)?\b", r"\bcertificate\b", r"\bpoa\b", r"\bevidence\b"
    ],
    "Address / identity": [
        r"\baddress\b", r"\bchange of address\b", r"\bidentity\b", r"\bname change\b"
    ],
    "Eligibility / service": [
        r"\beligibilit(y|ies)\b", r"\bqualif(y|ied)\b", r"\bservice\b", r"\bvest(ed|ing)\b"
    ],
    "Calculation / rules": [
        r"\brule(s)?\b", r"\bcalc(ulation)?\b", r"\bpolicy\b", r"\binterpretation\b"
    ],
    "Escalation / approval": [
        r"\bescalat(e|ion)\b", r"\bmanager\b", r"\bapprov(al|e)\b"
    ],
    "Backlog / workload": [
        r"\bbacklog\b", r"\bworkload\b", r"\bcapacity\b", r"\bqueue\b"
    ],
    "Third-party delay": [
        r"\bprovider\b", r"\binsurer\b", r"\bbroker\b", r"\bexternal\b", r"\bthird[-\s]*party\b"
    ],
    "Reporting / MI": [
        r"\breport(ing)?\b", r"\bmi\b", r"\bmetric(s)?\b", r"\bstat(s|istics)\b"
    ],
}

# pre-compile for speed
_COMPILED = {k: re.compile("|".join(v), flags=re.I) for k, v in _REASON_KEYWORDS.items()}


def _label_reason(text: str) -> str:
    t = str(text or "")
    if not t:
        return "Other"
    for cat, rx in _COMPILED.items():
        if rx.search(t):
            return cat
    return "Other"


# --------------------------- pareto builder ------------------------------------------

def _pareto_top80(counts: pd.Series) -> Tuple[pd.DataFrame, plt.Figure]:
    """
    counts: index=reason, value=count
    returns: (pareto_df, fig)
    """
    if counts.empty:
        return pd.DataFrame(columns=["reason", "count", "percent", "cum_percent"]), plt.figure()

    df = counts.reset_index()
    df.columns = ["reason", "count"]
    df = df.sort_values("count", ascending=False, ignore_index=True)
    total = df["count"].sum()

    df["percent"] = (df["count"] / total * 100).round(1)
    df["cum_percent"] = df["percent"].cumsum().round(1)

    # keep those contributing to top 80%, club rest into "Other"
    keep = df[df["cum_percent"] <= 80.0].index.tolist()
    if keep:
        kept = df.loc[keep]
        rest = df.loc[~df.index.isin(keep)]
        other_count = int(rest["count"].sum())
        if other_count > 0:
            df_top = pd.concat([kept, pd.DataFrame([{"reason": "Other", "count": other_count}])],
                               ignore_index=True)
        else:
            df_top = kept.copy()
    else:
        # if nothing under 80, just show biggest category as is and rest as Other
        biggest = df.head(1)
        other_count = int(df.iloc[1:]["count"].sum())
        if other_count > 0:
            df_top = pd.concat([biggest, pd.DataFrame([{"reason": "Other", "count": other_count}])],
                               ignore_index=True)
        else:
            df_top = biggest.copy()

    # recompute percents on the shown set
    total2 = df_top["count"].sum()
    df_top["percent"] = (df_top["count"] / total2 * 100).round(1)
    df_top["cum_percent"] = df_top["percent"].cumsum().round(1)

    # plot
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(df_top["reason"], df_top["count"])
    # formatting: no grid, hide y-axis, annotate bars
    ax.grid(False)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.set_ylabel("")  # remove y label
    ax.set_xlabel("")  # remove x label
    ax.tick_params(axis="y", left=False, labelleft=False)
    ax.tick_params(axis="x", rotation=90)

    for b in bars:
        ax.bar_label([b], labels=[f"{int(b.get_height())}"], padding=3)

    fig.tight_layout()
    return df_top, fig


# --------------------------- main render ---------------------------------------------

def run(store: Dict, params: Dict, q: str) -> None:
    """
    Streamlit renderer: Reasons for Fail — Pareto (top 80% + Other)
    """
    st.subheader("Reasons for Fail — Analysis")

    df, err = _get_fpa_df(store)
    if err:
        st.info(err)
        return

    # consider only fails (anything that's not clearly pass)
    # adjust this predicate to your exact review result vocabulary
    rr = df["review_result"].astype(str).str.strip().str.lower()
    is_pass = rr.str.contains(r"\bpass(ed)?\b", na=False)
    fails = df.loc[~is_pass].copy()

    if fails.empty:
        st.info("No failed rows available in the selected dataset.")
        return

    # month selector (YYYY-MM) default -> latest available from activity_date
    fails["month"] = fails["activity_date"].dt.to_period("M").astype(str)
    months = sorted(fails["month"].dropna().unique().tolist())
    default_month = months[-1] if months else None

    sel_month = st.selectbox("Select month", months, index=(months.index(default_month) if default_month in months else 0))

    dfm = fails.loc[fails["month"] == sel_month].copy()
    st.markdown(f"### Reasons for Fail — {pd.Period(sel_month).strftime('%b-%y') if sel_month else ''}")

    if dfm.empty:
        st.info(f"No failed rows found for {sel_month}.")
        return

    # label reasons
    dfm["reason"] = dfm["case_comment"].map(_label_reason)

    # counts
    counts = dfm["reason"].value_counts()

    pareto_df, fig = _pareto_top80(counts)

    left, right = st.columns([2, 1])
    with left:
        st.pyplot(fig, clear_figure=True)
    with right:
        st.dataframe(pareto_df, use_container_width=True)
