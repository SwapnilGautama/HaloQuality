# -*- coding: utf-8 -*-
# questions/complaints_june_by_portfolio.py
from __future__ import annotations

import os
import re
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# -------------------------------
# Optional OpenAI (AI labelling)
# -------------------------------
_OPENAI_READY = False
try:
    import openai  # type: ignore
    if os.getenv("OPENAI_API_KEY"):
        openai.api_key = os.getenv("OPENAI_API_KEY")
        _OPENAI_READY = True
except Exception:
    _OPENAI_READY = False


# -------------------------------
# small helpers (styling)
# -------------------------------

_DARK_BLUE = "#0b3d91"
_DARK_GREY = "#333333"
_SOFT_GREY = "#DDDDDD"

def _header(title: str) -> None:
    st.markdown(
        f"<h3 style='color:{_DARK_BLUE};margin:0 0 .35rem 0; font-weight:700;'>{title}</h3>",
        unsafe_allow_html=True,
    )

def _style_table(df: pd.DataFrame) -> "pd.io.formats.style.Styler":
    return (
        df.style
        .set_table_styles(
            [
                {"selector": "th", "props": [("color", _DARK_BLUE), ("font-weight", "700")]},
                {"selector": "tbody td", "props": [("color", _DARK_GREY)]},
            ]
        )
        .set_properties(**{"color": _DARK_GREY})
        .format(precision=3)
    )


def _find_first_col(df: pd.DataFrame, candidates: List[str]) -> str | None:
    cols = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in cols:
            return cols[c.lower()]
    return None


def _norm_month_from_series(s: pd.Series) -> pd.Series:
    if pd.api.types.is_period_dtype(s):
        return s.astype(str)
    dt = pd.to_datetime(s, errors="coerce", dayfirst=True, utc=False)
    return dt.dt.to_period("M").astype(str)


def _add_total_row(df: pd.DataFrame, sum_cols: List[str], label_col: str, label="Total") -> pd.DataFrame:
    total = {c: df[c].sum() if c in sum_cols else None for c in df.columns}
    total[label_col] = label
    out = pd.concat([pd.DataFrame([total]), df], ignore_index=True)
    if all(c in out.columns for c in ["cases", "complaints"]):
        with np.errstate(divide="ignore", invalid="ignore"):
            per = out["complaints"] * 1000 / out["cases"]
        out["per_1000"] = per.replace([np.inf, -np.inf], np.nan)
    return out


# -------------------------------
# RCA mappings (keyword fallback)
# -------------------------------

def _preclean(text: str) -> str:
    t = (text or "").lower()
    t = re.sub(r"[_/\\\-]+", " ", t)
    t = re.sub(r"[^a-z0-9 %]", " ", t)
    replacements = {
        r"\bbacs\b": " bank payment ",
        r"\bchaps\b": " bank payment ",
        r"\bdd\b": " direct debit ",
        r"\bsoa\b": " statement of account ",
        r"\btpa\b": " third party ",
        r"\bifa\b": " third party ",
        r"\bpoa\b": " power of attorney ",
        r"\baddr\b": " address ",
        r"\bni\b": " national insurance ",
        r"\bsort\b": " sort code ",
        r"\bacc\b": " account ",
        r"\brecalc(ulate|ulation|)\b": " recalculation ",
        r"\bcalc(ulate|ulation|)\b": " calculation ",
        r"\bresp\b": " response ",
        r"\bsig(n|)\b": " sign ",
        r"\bid\b": " identification ",
        r"\bkyc\b": " identification ",
        r"\bcof\b": " change of fund ",
    }
    for pat, repl in replacements.items():
        t = re.sub(pat, repl, t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _classify(text: str, patterns: Dict[str, List[str]], default: str) -> str:
    t = _preclean(text)
    for label, pats in patterns.items():
        for p in pats:
            if re.search(p, t):
                return label
    return default


def _rca1_keyword(text: str) -> str:
    patterns = {
        "Delay": [
            r"\bdelay(ed|s|ing)?\b", r"\btimes? ?scale\b", r"\bsla\b", r"\bbacklog\b",
            r"\bqueue\b", r"\boverdue\b", r"\bawait(ing)?\b", r"\bchase(r|s|d|)?\b",
        ],
        "Procedure": [
            r"\bscheme rules?\b", r"\bprocedure\b", r"\bprocess\b",
            r"\btemplate\b", r"\bform\b", r"\bconsent form\b",
            r"\bdocument(ation)? (missing|not provided|not received)\b",
            r"\bevidence (missing|not provided|not received)\b",
        ],
        "Communication": [
            r"\bcommunicat(e|ion|ions)\b", r"\bemail\b", r"\bletter\b",
            r"\bphone|call\b", r"\bupdate\b", r"\bno (reply|response)\b",
            r"\bunclear\b", r"\bmis-?communicat",
        ],
        "System": [
            r"\bsystem\b", r"\bportal\b", r"\bworkflow\b", r"\bautomation\b",
            r"\bbug\b", r"\btechnical\b", r"\bit\b",
        ],
        "Incorrect/Incomplete information": [
            r"\bincorrect\b", r"\bwrong\b", r"\bincomplete\b", r"\berror\b",
            r"\bmis-?key(ed)?\b", r"\btypo\b", r"\bmisallocat(ed|ion)\b",
            r"\baddress\b", r"\bbank\b", r"\bdob\b", r"\bnational insurance\b", r"\bsort code\b", r"\baccount\b",
        ],
    }
    return _classify(text, patterns, default="Other")


def _rca2_keyword(text: str) -> str:
    patterns = {
        "Manual calculation": [r"\bmanual calculat", r"\bre-?calculat", r"\brecalculation\b"],
        "Data entry error": [
            r"\bmis-?key", r"\bkey(ing|ed) error\b", r"\btypo\b", r"\bwrong (amount|date|address|bank|ni|nino)\b",
            r"\bincorrect (amount|date|address|bank|ni|nino)\b", r"\bduplicate( case|)\b",
        ],
        "Documentation missing": [
            r"\bdocument(ation)? (missing|not provided|not received)\b",
            r"\bevidence (missing|not provided|not received)\b",
            r"\b(ids?|passport|birth|marriage|death) (cert|certificate)\b",
            r"\bproof (of )?(address|id|identity)\b",
        ],
        "Template/Form issue": [
            r"\bwrong form\b", r"\bincorrect form\b", r"\btemplate\b",
            r"\bform (not|in) (complete|completed|signed)\b", r"\bmissing signature\b", r"\bunsigned\b",
            r"\bcheckbox|tick box\b",
        ],
        "Waiting on member/TPA": [
            r"\bawait(ing)?\b", r"\bchase(r|s|d|ing)?\b", r"\bno (reply|response)\b",
            r"\bthird party\b", r"\bifa\b", r"\binsurer\b",
        ],
        "Bank/Payment issue": [
            r"\bbank\b", r"\b(bank )?payment\b", r"\brefund\b", r"\breturned\b",
            r"\bbounced\b", r"\bchaps\b", r"\bbacs\b", r"\bcheque\b", r"\baccount\b", r"\bsort code\b",
        ],
        "Overpayment": [r"\boverpayment\b"],
        "Address/Contact incorrect": [
            r"\baddress (wrong|incorrect|incomplete|change|updated?)\b",
            r"\bcontact (details|number|email) (wrong|incorrect|missing|change|updated?)\b",
        ],
        "Pension set up": [r"\b(pension|record) set ?up\b", r"\bsetup\b"],
        "Postal delay": [r"\bpost(al)?\b", r"\bmail\b"],
        "AVC": [r"\bavc\b"],
        "Case not created": [r"\bcase not created\b", r"\bnot created\b", r"\bnot raised\b"],
        "2nd review / QA": [r"\b(second|2nd) review\b", r"\bqa\b"],
        "Trustee": [r"\btrustee\b"],
        "Death benefits payout": [r"\bdeath benefit", r"\bbeneficiar(y|ies)\b"],
        "Drop in value / factor change": [r"\bfactor change\b", r"\bdrop in value\b", r"\bmarket\b", r"\bunit price\b"],
        "Scheme rules": [r"\bscheme rules?\b", r"\blegislation\b"],
        "Communication unclear": [r"\black of clarity\b", r"\bnot clear\b", r"\bconfus"],
    }
    return _classify(text, patterns, default="Other")


# -------------------------------
# Optional: AI labelling helpers
# -------------------------------

_RCA1_ALLOWED = [
    "Delay",
    "Procedure",
    "Communication",
    "System",
    "Incorrect/Incomplete information",
    "Other",
]

_RCA2_ALLOWED = [
    "Manual calculation",
    "Documentation missing",
    "Template/Form issue",
    "Data entry error",
    "Waiting on member/TPA",
    "Bank/Payment issue",
    "Address/Contact incorrect",
    "Pension set up",
    "Postal delay",
    "AVC",
    "Case not created",
    "2nd review / QA",
    "Trustee",
    "Death benefits payout",
    "Overpayment",
    "Drop in value / factor change",
    "Scheme rules",
    "Communication unclear",
    "Other",
]


def _ai_label_batch(texts: List[str]) -> List[Tuple[str, str]]:
    if not _OPENAI_READY:
        return [(_rca1_keyword(t), _rca2_keyword(t)) for t in texts]
    try:
        prompt = (
            "You are classifying complaint root causes. "
            "For each 'Brief Description – RCA done by admin', return a JSON array of objects "
            "with keys 'rca1' and 'rca2'.\n\n"
            f"RCA1 must be one of: {', '.join(_RCA1_ALLOWED)}.\n"
            f"RCA2 must be one of: {', '.join(_RCA2_ALLOWED)}.\n"
            "If unsure, use 'Other'.\n\n"
            "Descriptions:\n"
            + "\n".join([f"- {t}" for t in texts])
        )
        resp = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Return only valid JSON."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
        )
        content = resp["choices"][0]["message"]["content"]
        out = []
        try:
            data = pd.read_json(content)
            if isinstance(data, pd.DataFrame):
                for _, row in data.iterrows():
                    r1 = row.get("rca1", "Other")
                    r2 = row.get("rca2", "Other")
                    r1 = r1 if r1 in _RCA1_ALLOWED else "Other"
                    r2 = r2 if r2 in _RCA2_ALLOWED else "Other"
                    out.append((r1, r2))
        except Exception:
            out = []
        if len(out) != len(texts):
            return [(_rca1_keyword(t), _rca2_keyword(t)) for t in texts]
        return out
    except Exception:
        return [(_rca1_keyword(t), _rca2_keyword(t)) for t in texts]


# -------------------------------
# field detection
# -------------------------------

def _detect_cases_fields(cases: pd.DataFrame):
    id_col = _find_first_col(cases, ["Case ID", "CaseId", "ID"])
    port_col = _find_first_col(cases, ["Portfolio", "portfolio"])
    date_col = _find_first_col(
        cases, ["Create Date (cases)", "Create Date", "Create date", "Start Date", "StartDate", "Created On", "CreateDt"]
    )
    return id_col, port_col, date_col


def _detect_complaints_fields(comp: pd.DataFrame):
    id_col = _find_first_col(comp, ["Original Process Affected Case ID", "Case ID", "Parent Case ID"])
    port_col = _find_first_col(comp, ["Portfolio", "portfolio"])
    date_col = _find_first_col(
        comp, [
            "Date Complaint Received - DD/MM/YY",
            "Date Complaint Received",
            "Complaint Date",
            "Received Date",
            "Month",
        ],
    )
    return id_col, port_col, date_col


def _build_month_column(df: pd.DataFrame, raw_col: str, assume_year: int | None = None) -> pd.Series:
    s = df[raw_col]
    m = _norm_month_from_series(s)
    if m.notna().sum() >= max(1, int(0.1 * len(m))):
        return m
    if assume_year is not None:
        try:
            coerced = pd.to_datetime(s.astype(str) + f" {assume_year}", format="%B %Y", errors="coerce")
        except Exception:
            coerced = pd.to_datetime(s.astype(str) + f" {assume_year}", errors="coerce")
        return coerced.dt.to_period("M").astype(str)
    return pd.to_datetime(s, errors="coerce", dayfirst=True).dt.to_period("M").astype(str)


# -------------------------------
# core computations
# -------------------------------

def _portfolio_table_for_june(cases: pd.DataFrame, comp: pd.DataFrame) -> pd.DataFrame:
    id_c, port_c, date_c = _detect_cases_fields(cases)
    id_k, port_k, date_k = _detect_complaints_fields(comp)

    missing = []
    if port_c is None: missing.append("Portfolio (cases)")
    if date_c is None: missing.append("Create Date (cases)")
    if port_k is None: missing.append("Portfolio (complaints)")
    if date_k is None: missing.append("Complaint date (complaints)")
    if missing:
        st.warning(f"Missing columns: {missing}")
        return pd.DataFrame(columns=["portfolio", "cases", "complaints", "per_1000"])

    cases = cases.copy()
    comp = comp.copy()

    cases["_month"] = _build_month_column(cases, date_c)
    comp["_month"] = _build_month_column(comp, date_k, assume_year=2025 if date_k.lower() == "month" else None)

    cases_jun = cases.loc[cases["_month"] == "2025-06"].groupby(port_c, dropna=False, as_index=False).size()
    cases_jun.rename(columns={"size": "cases", port_c: "portfolio"}, inplace=True)

    comp_jun = comp.loc[comp["_month"] == "2025-06"].groupby(port_k, dropna=False, as_index=False).size()
    comp_jun.rename(columns={"size": "complaints", port_k: "portfolio"}, inplace=True)

    out = pd.merge(cases_jun, comp_jun, how="left", on="portfolio")
    out["complaints"] = out["complaints"].fillna(0).astype(int)
    with np.errstate(divide="ignore", invalid="ignore"):
        out["per_1000"] = (out["complaints"] * 1000 / out["cases"]).replace([np.inf, -np.inf], np.nan)
    out["per_1000"] = out["per_1000"].round(3)

    out = out.sort_values(["complaints", "portfolio"], ascending=[False, True], kind="stable").reset_index(drop=True)
    out = _add_total_row(out, sum_cols=["cases", "complaints"], label_col="portfolio", label="Total")
    return out


def _mom_series(cases: pd.DataFrame, comp: pd.DataFrame) -> pd.DataFrame:
    _, _, date_c = _detect_cases_fields(cases)
    _, _, date_k = _detect_complaints_fields(comp)
    if date_c is None or date_k is None:
        return pd.DataFrame(columns=["month", "per_1000"])

    cases = cases.copy(); comp = comp.copy()
    cases["_month"] = _build_month_column(cases, date_c)
    comp["_month"] = _build_month_column(comp, date_k, assume_year=2025 if date_k.lower() == "month" else None)

    want = [f"2025-{i:02d}" for i in range(1, 7)]
    cases_m = cases.loc[cases["_month"].isin(want)].groupby("_month").size().reindex(want, fill_value=0)
    comp_m = comp.loc[comp["_month"].isin(want)].groupby("_month").size().reindex(want, fill_value=0)
    per_1000 = (comp_m * 1000 / cases_m.replace(0, np.nan)).fillna(0.0).round(2)
    return pd.DataFrame({"month": want, "per_1000": per_1000.values})


def _rca_tables_for_june(comp: pd.DataFrame, use_ai: bool) -> Tuple[pd.DataFrame, pd.DataFrame]:
    _, _, date_k = _detect_complaints_fields(comp)
    desc_col = _find_first_col(
        comp,
        ["Brief Description - RCA done by admin", "Brief Description", "RCA", "Admin Description", "Root Cause"],
    )
    if date_k is None or desc_col is None:
        return pd.DataFrame(columns=["RCA2", "count", "percent", "cum_percent"]), pd.DataFrame(columns=["RCA1", "count"])

    comp = comp.copy()
    comp["_month"] = _build_month_column(comp, date_k, assume_year=2025 if date_k.lower() == "month" else None)
    june = comp.loc[comp["_month"] == "2025-06", [desc_col]].fillna("")
    texts = june[desc_col].astype(str).tolist()

    if use_ai and _OPENAI_READY and len(texts) > 0:
        r1_labels, r2_labels = [], []
        batch = 80
        for i in range(0, len(texts), batch):
            chunk = texts[i:i+batch]
            pairs = _ai_label_batch(chunk)
            r1_labels.extend([p[0] for p in pairs])
            r2_labels.extend([p[1] for p in pairs])
    else:
        r1_labels = [_rca1_keyword(t) for t in texts]
        r2_labels = [_rca2_keyword(t) for t in texts]

    r1 = pd.Series(r1_labels).value_counts(dropna=False).rename_axis("RCA1").reset_index(name="count")

    r2_counts = pd.Series(r2_labels).value_counts(dropna=False).rename_axis("RCA2").reset_index(name="count")
    r2_counts["order"] = np.where(r2_counts["RCA2"].eq("Other"), 1, 0)
    r2_counts = r2_counts.sort_values(["order", "count"], ascending=[True, False]).drop(columns="order").reset_index(drop=True)

    total = max(1, r2_counts["count"].sum())
    r2_counts["percent"] = (r2_counts["count"] * 100 / total).round(2)
    r2_counts["cum_percent"] = r2_counts["percent"].cumsum()
    r2 = r2_counts.loc[r2_counts["cum_percent"] <= 80.0].reset_index(drop=True)

    return r2, r1


# -------------------------------
# plotting
# -------------------------------

def _plot_mom_line(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(6.0, 3.2))
    ax.plot(df["month"], df["per_1000"], marker="o", linewidth=2.5, color="#9ecae1")
    for x, y in zip(df["month"], df["per_1000"]):
        ax.text(x, y + 0.03, f"{y:.2f}", ha="center", va="bottom", fontsize=9, color=_DARK_GREY)
    ax.set_title("Complaints per 1,000 — MoM (Jan–Jun ’25)", pad=8, color=_DARK_BLUE)
    ax.set_ylim(bottom=0)
    # soften axes
    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_color(_SOFT_GREY)
    ax.spines["bottom"].set_linewidth(1.25)
    ax.get_yaxis().set_visible(False)
    ax.tick_params(axis="x", colors=_DARK_GREY)
    ax.set_xlabel(""); ax.set_ylabel(""); ax.grid(False)
    st.pyplot(fig)


def _plot_rca1_pareto(df: pd.DataFrame):
    """
    Pareto chart: bars (descending counts) + cumulative percentage line.
    Labels rotated vertical for readability.
    """
    data = df.copy()
    data = data.sort_values("count", ascending=False).reset_index(drop=True)
    total = float(max(1, data["count"].sum()))
    data["percent"] = data["count"] * 100.0 / total
    data["cum_percent"] = data["percent"].cumsum()

    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    bar_colors = ["#9ecae1", "#a1d99b", "#fdd0a2", "#c7c7c7", "#fdae6b", "#d9d9d9", "#bcbddc", "#ccebc5"]
    ax.bar(data["RCA1"], data["count"], color=bar_colors[: len(data)])

    # annotate bar counts
    for i, y in enumerate(data["count"].tolist()):
        ax.text(i, y + max(1, y * 0.02), f"{int(y)}", ha="center", va="bottom", fontsize=9, color=_DARK_GREY)

    ax.set_title("RCA1 — June 2025 (Pareto)", pad=8, color=_DARK_BLUE)

    # right-hand cumulative % line
    ax2 = ax.twinx()
    ax2.plot(data["RCA1"], data["cum_percent"], color=_DARK_BLUE, marker="o", linewidth=2)
    for i, y in enumerate(data["cum_percent"].tolist()):
        ax2.text(i, y + 1, f"{y:.0f}%", ha="center", va="bottom", fontsize=9, color=_DARK_GREY)

    # styling: no left y-axis, soft bottom spine, vertical x labels
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.grid(False)
    ax.get_yaxis().set_visible(False)
    ax.spines["bottom"].set_color(_SOFT_GREY)
    ax.spines["bottom"].set_linewidth(1.25)
    ax.tick_params(axis="x", colors=_DARK_GREY)
    plt.setp(ax.get_xticklabels(), rotation=90, ha="center")

    # right y-axis = 0..100%
    ax2.set_ylim(0, 100)
    ax2.set_ylabel("")  # clean
    ax2.tick_params(axis="y", colors=_DARK_GREY)
    ax2.grid(False)

    st.pyplot(fig)


# -------------------------------
# streamlit UI / entrypoint
# -------------------------------

def run(store: Dict, params: Dict, user_text: str = "") -> Tuple[str, pd.DataFrame]:
    """
    Row 1: Portfolio table + MoM line
    Row 2: RCA1 Pareto (left) + RCA2 Top-80 table (right)
    """
    # Hide host “Parsed filters” expander & trailing info alerts
    st.markdown(
        """
        <style>
        div[data-testid="stExpander"] {display: none;}
        div[data-testid="stAlert"] {display: none;}
        </style>
        """,
        unsafe_allow_html=True,
    )

    cases: pd.DataFrame = store.get("cases", pd.DataFrame()).copy()
    comp: pd.DataFrame = store.get("complaints", pd.DataFrame()).copy()

    # ----- Row 1 -----
    table = _portfolio_table_for_june(cases, comp)
    mom = _mom_series(cases, comp)

    c1, c2 = st.columns((1.2, 1.0), gap="large")
    with c1:
        _header("Complaint analysis — Jun 2025 (by portfolio)")
        if table.empty:
            st.info("No rows returned for the current filters.")
        else:
            st.dataframe(_style_table(table), use_container_width=True)
    with c2:
        if not mom.empty:
            _plot_mom_line(mom)

    # ----- Row 2 (RCA1 Pareto LEFT, RCA2 table RIGHT) -----
    use_ai = bool(os.getenv("OPENAI_API_KEY"))
    rca2, rca1 = _rca_tables_for_june(comp, use_ai=use_ai)

    c3, c4 = st.columns((1.05, 1.0), gap="large")
    with c3:
        if not rca1.empty:
            _plot_rca1_pareto(rca1)
    with c4:
        _header("RCA2 — Top 80% (June 2025)")
        if rca2.empty:
            st.info("No June-2025 complaints with usable RCA text.")
        else:
            st.dataframe(_style_table(rca2.rename(columns={"RCA2": "RCA2"})), use_container_width=True)

    # prevent host app duplicate table
    return ("", pd.DataFrame())
