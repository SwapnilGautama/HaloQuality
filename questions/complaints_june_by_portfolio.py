# questions/complaints_june_by_portfolio.py
from __future__ import annotations

import os
import re
from io import BytesIO
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# Optional OpenAI for AI labelling
_OPENAI_READY = False
try:
    import openai  # type: ignore
    if os.getenv("OPENAI_API_KEY"):
        openai.api_key = os.getenv("OPENAI_API_KEY")
        _OPENAI_READY = True
except Exception:
    _OPENAI_READY = False

# Optional python-pptx for export
_PPT_READY = False
try:
    from pptx import Presentation  # type: ignore
    from pptx.util import Inches, Pt
    from pptx.enum.text import PP_ALIGN
    from pptx.dml.color import RGBColor
    _PPT_READY = True
except Exception:
    _PPT_READY = False


# -------------------------------
# Theming helpers
# -------------------------------

_DARK_BLUE = "#0b3d91"
_DARK_GREY = "#333333"
_SOFT_GREY = "#DDDDDD"
_PASTEL_LINE = "#88cde8"
_PARETO_LINE = "#6ab6e1"

def _header(title: str) -> None:
    st.markdown(
        f"<h3 style='color:{_DARK_BLUE};margin:0 0 .35rem 0; font-weight:700;'>{title}</h3>",
        unsafe_allow_html=True,
    )

def _hide_index(sty: "pd.io.formats.style.Styler") -> "pd.io.formats.style.Styler":
    try:
        return sty.hide(axis="index")
    except Exception:
        try:
            return sty.hide_index()
        except Exception:
            return sty

def _style_table(
    df: pd.DataFrame,
    formats: Dict[str, str] | None = None,
) -> "pd.io.formats.style.Styler":
    sty = (
        df.style
        .set_table_styles(
            [
                {"selector": "th", "props": [("color", _DARK_BLUE), ("font-weight", "700")]},
                {"selector": "tbody td", "props": [("color", _DARK_GREY)]},
            ]
        )
        .set_properties(**{"color": _DARK_GREY})
    )
    sty = _hide_index(sty)
    if formats:
        sty = sty.format(formats)
    else:
        sty = sty.format(precision=3)
    return sty


# -------------------------------
# Column detection / month helpers
# -------------------------------

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

@st.cache_data(show_spinner=False)
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

def _add_total_row(df: pd.DataFrame, sum_cols: List[str], label_col: str, label="Total") -> pd.DataFrame:
    total = {c: df[c].sum() if c in sum_cols else None for c in df.columns}
    total[label_col] = label
    out = pd.concat([pd.DataFrame([total]), df], ignore_index=True)
    if all(c in out.columns for c in ["cases", "complaints"]):
        with np.errstate(divide="ignore", invalid="ignore"):
            per = out["complaints"] * 1000 / out["cases"]
        out["per_1000"] = per.replace([np.inf, -np.inf], np.nan)
    return out

def _months_jan_to_aug_2025() -> List[str]:
    return [f"2025-{i:02d}" for i in range(1, 9)]

# detect latest available month (by cases; robust to varied columns)
@st.cache_data(show_spinner=False)
def _latest_month_2025(cases: pd.DataFrame, comp: pd.DataFrame) -> Tuple[str | None, str]:
    c_date = _find_first_col(
        cases,
        ["Create Date (cases)", "Create Date", "Create date", "Start Date", "StartDate", "Created On", "CreateDt"],
    )
    k_date = _find_first_col(
        comp,
        ["Date Complaint Received - DD/MM/YY", "Date Complaint Received", "Complaint Date", "Received Date", "Month"],
    )
    if c_date is None and k_date is None:
        return None, "Latest"
    latest = None
    if c_date is not None:
        cm = pd.to_datetime(cases[c_date], errors="coerce", dayfirst=True).dt.to_period("M")
        cm = cm[cm.dt.year == 2025]
        if not cm.dropna().empty:
            latest = cm.max()
    if latest is None and k_date is not None:
        # fall back to complaints
        if k_date.lower() == "month":
            km = pd.to_datetime(comp[k_date].astype(str) + " 2025", errors="coerce").dt.to_period("M")
        else:
            km = pd.to_datetime(comp[k_date], errors="coerce", dayfirst=True).dt.to_period("M")
        km = km[km.dt.year == 2025]
        if not km.dropna().empty:
            latest = km.max()
    if latest is None:
        return None, "Latest"
    return str(latest), pd.Period(latest).to_timestamp().strftime("%b %Y")


# -------------------------------
# RCA rules + AI support
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

_RCA1_ALLOWED = [
    "Delay", "Procedure", "Communication", "System",
    "Incorrect/Incomplete information", "Other",
]
_RCA2_ALLOWED = [
    "Manual calculation", "Documentation missing", "Template/Form issue", "Data entry error",
    "Waiting on member/TPA", "Bank/Payment issue", "Address/Contact incorrect",
    "Pension set up", "Postal delay", "AVC", "Case not created", "2nd review / QA",
    "Trustee", "Death benefits payout", "Overpayment", "Drop in value / factor change",
    "Scheme rules", "Communication unclear", "Other",
]

RCA2_TO_RCA1_MAP = {
    "Manual calculation": "Delay",
    "Waiting on member/TPA": "Delay",
    "Postal delay": "Delay",
    "Case not created": "Delay",
    "2nd review / QA": "Delay",
    "Pension set up": "Delay",
    "Trustee": "Delay",
    "AVC": "Delay",
    "Overpayment": "Delay",
    "Death benefits payout": "Delay",
    "Bank/Payment issue": "Delay",

    "Documentation missing": "Procedure",
    "Template/Form issue": "Procedure",
    "Scheme rules": "Procedure",

    "Communication unclear": "Communication",

    "Data entry error": "Incorrect/Incomplete information",
    "Address/Contact incorrect": "Incorrect/Incomplete information",
    "Drop in value / factor change": "Incorrect/Incomplete information",
}

DELAY_RCA2 = {
    "Manual calculation", "Waiting on member/TPA", "Postal delay",
    "Case not created", "2nd review / QA", "Pension set up",
    "Trustee", "AVC", "Overpayment", "Death benefits payout",
    "Bank/Payment issue",
}
DELAY_EXTERNAL = {"Waiting on member/TPA", "Bank/Payment issue", "Postal delay", "Trustee", "AVC"}
DELAY_APTIA = {"Manual calculation", "Case not created", "2nd review / QA", "Pension set up", "Overpayment", "Data entry error"}

@st.cache_data(show_spinner=False)
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
            "Prefer specific labels over 'Other'.\n\n"
            "Descriptions:\n" + "\n".join([f"- {t}" for t in texts])
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
    desc_col = _find_first_col(
        comp,
        ["Brief Description - RCA done by admin", "Brief Description", "RCA", "Admin Description", "Root Cause"],
    )
    return id_col, port_col, date_col, desc_col


# -------------------------------
# Core computations — CACHED
# -------------------------------

@st.cache_data(show_spinner=False)
def _portfolio_table_for_month(cases: pd.DataFrame, comp: pd.DataFrame, month_str: str) -> pd.DataFrame:
    id_c, port_c, date_c = _detect_cases_fields(cases)
    _, port_k, date_k, _ = _detect_complaints_fields(comp)
    if any(x is None for x in [port_c, date_c, port_k, date_k]) or not month_str:
        return pd.DataFrame(columns=["portfolio", "cases", "complaints", "per_1000"])

    cases = cases.copy(); comp = comp.copy()
    cases["_month"] = _build_month_column(cases, date_c)
    comp["_month"] = _build_month_column(comp, date_k, assume_year=2025 if date_k.lower() == "month" else None)

    cases_m = cases.loc[cases["_month"] == month_str].groupby(port_c, dropna=False, as_index=False).size()
    cases_m.rename(columns={"size": "cases", port_c: "portfolio"}, inplace=True)

    comp_m = comp.loc[comp["_month"] == month_str].groupby(port_k, dropna=False, as_index=False).size()
    comp_m.rename(columns={"size": "complaints", port_k: "portfolio"}, inplace=True)

    out = pd.merge(cases_m, comp_m, how="left", on="portfolio")
    out["complaints"] = out["complaints"].fillna(0).astype(int)
    with np.errstate(divide="ignore", invalid="ignore"):
        out["per_1000"] = (out["complaints"] * 1000 / out["cases"]).replace([np.inf, -np.inf], np.nan)
    out["per_1000"] = out["per_1000"].round(1)

    out = out.sort_values(["complaints", "portfolio"], ascending=[False, True], kind="stable").reset_index(drop=True)
    out = _add_total_row(out, sum_cols=["cases", "complaints"], label_col="portfolio", label="Total")
    out["per_1000"] = out["per_1000"].round(1)
    return out

@st.cache_data(show_spinner=False)
def _mom_series(cases: pd.DataFrame, comp: pd.DataFrame) -> pd.DataFrame:
    _, _, date_c = _detect_cases_fields(cases)
    _, _, date_k, _ = _detect_complaints_fields(comp)
    if date_c is None or date_k is None:
        return pd.DataFrame(columns=["month", "per_1000"])

    cases = cases.copy(); comp = comp.copy()
    cases["_month"] = _build_month_column(cases, date_c)
    comp["_month"] = _build_month_column(comp, date_k, assume_year=2025 if date_k.lower() == "month" else None)

    want = _months_jan_to_aug_2025()
    cases_m = cases.loc[cases["_month"].isin(want)].groupby("_month").size().reindex(want, fill_value=0)
    comp_m = comp.loc[comp["_month"].isin(want)].groupby("_month").size().reindex(want, fill_value=0)
    per_1000 = (comp_m * 1000 / cases_m.replace(0, np.nan)).fillna(0.0).round(1)
    pretty = [pd.Period(m).to_timestamp().strftime("%b-%y") for m in want]
    return pd.DataFrame({"month": pretty, "per_1000": per_1000.values})

def _repair_rca1_from_rca2(rca1: List[str], rca2: List[str]) -> List[str]:
    out = []
    for a, b in zip(rca1, rca2):
        if a == "Other" and b in RCA2_TO_RCA1_MAP:
            out.append(RCA2_TO_RCA1_MAP[b])
        else:
            out.append(a)
    return out

@st.cache_data(show_spinner=False)
def _rca_tables_for_june(comp: pd.DataFrame, use_ai: bool) -> Tuple[pd.DataFrame, pd.DataFrame]:
    _, _, date_k, desc_col = _detect_complaints_fields(comp)
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

    r1_labels = _repair_rca1_from_rca2(r1_labels, r2_labels)

    r1 = pd.Series(r1_labels).value_counts(dropna=False).rename_axis("RCA1").reset_index(name="count")

    r2_counts = pd.Series(r2_labels).value_counts(dropna=False).rename_axis("RCA2").reset_index(name="count")
    r2_counts["order"] = np.where(r2_counts["RCA2"].eq("Other"), 1, 0)
    r2_counts = r2_counts.sort_values(["order", "count"], ascending=[True, False]).drop(columns="order").reset_index(drop=True)
    total = max(1, r2_counts["count"].sum())
    r2_counts["percent"] = (r2_counts["count"] * 100 / total)
    r2_counts["cum_percent"] = r2_counts["percent"].cumsum()
    r2_counts["percent"] = r2_counts["percent"].round(1)
    r2_counts["cum_percent"] = r2_counts["cum_percent"].round(1)
    r2 = r2_counts.loc[r2_counts["cum_percent"] <= 80.0].reset_index(drop=True)

    return r2, r1

@st.cache_data(show_spinner=False)
def _rca2_table_by_portfolio_for_june(
    comp: pd.DataFrame,
    use_ai: bool,
    portfolios: List[str] | None = None,
    rca1_keep: List[str] | None = None,
) -> pd.DataFrame:
    id_k, port_k, date_k, desc_col = _detect_complaints_fields(comp)
    if any(x is None for x in [port_k, date_k, desc_col]):
        return pd.DataFrame(columns=["Portfolio", "RCA2", "count"])

    df = comp.copy()
    df["_month"] = _build_month_column(df, date_k, assume_year=2025 if date_k.lower() == "month" else None)

    df = df.loc[df["_month"] == "2025-06", [port_k, desc_col]].dropna(subset=[desc_col])
    if df.empty:
        return pd.DataFrame(columns=["Portfolio", "RCA2", "count"])

    if portfolios and len(portfolios) > 0:
        df = df.loc[df[port_k].astype(str).isin([str(p) for p in portfolios])]
        if df.empty:
            return pd.DataFrame(columns=["Portfolio", "RCA2", "count"])

    texts = df[desc_col].astype(str).tolist()
    if use_ai and _OPENAI_READY and len(texts) > 0:
        pairs = []
        batch = 80
        for i in range(0, len(texts), batch):
            pairs.extend(_ai_label_batch(texts[i:i+batch]))
        r1_labels = [p[0] for p in pairs]
        r2_labels = [p[1] for p in pairs]
    else:
        r1_labels = [_rca1_keyword(t) for t in texts]
        r2_labels = [_rca2_keyword(t) for t in texts]
    r1_labels = _repair_rca1_from_rca2(r1_labels, r2_labels)

    if rca1_keep and len(rca1_keep) > 0:
        mask = [r in set(rca1_keep) for r in r1_labels]
        if not any(mask):
            return pd.DataFrame(columns=["Portfolio", "RCA2", "count"])
        df = df.loc[mask].copy()
        r2_labels = [r2 for r2, keep in zip(r2_labels, mask) if keep]

    df["RCA2"] = r2_labels
    df["Portfolio"] = df[port_k].astype(str).fillna("")

    tab = df.groupby(["Portfolio", "RCA2"], dropna=False, as_index=False).size().rename(columns={"size": "count"})
    tab = tab.sort_values(["count", "Portfolio", "RCA2"], ascending=[False, True, True], kind="stable").reset_index(drop=True)
    return tab


# -------------------------------
# Plotting (overall)
# -------------------------------

def _mom_line_fig(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(6.0, 3.2))
    ax.plot(df["month"], df["per_1000"], marker="o", linewidth=2.5, color="#9ecae1")
    for x, y in zip(df["month"], df["per_1000"]):
        ax.text(x, y + 0.03, f"{y:.1f}", ha="center", va="bottom", fontsize=9, color=_DARK_GREY)
    ax.set_title("Complaints per 1,000 — MoM (Jan–Aug ’25)", pad=8, color=_DARK_BLUE)
    ax.set_ylim(bottom=0)
    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_color(_SOFT_GREY)
    ax.spines["bottom"].set_linewidth(1.25)
    ax.get_yaxis().set_visible(False)
    ax.tick_params(axis="x", colors=_DARK_GREY)
    ax.set_xlabel(""); ax.set_ylabel(""); ax.grid(False)
    return fig

def _plot_mom_line(df: pd.DataFrame):
    st.pyplot(_mom_line_fig(df))

def _pareto_fig(df: pd.DataFrame):
    data = df.copy().sort_values("count", ascending=False).reset_index(drop=True)
    total = float(max(1, data["count"].sum()))
    data["percent"] = data["count"] * 100.0 / total
    data["cum_percent"] = data["percent"].cumsum()

    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    bar_colors = ["#9ecae1", "#a1d99b", "#bdbdbd", "#fdd0a2", "#fdae6b", "#c7c7c7", "#bcbddc", "#ccebc5"]
    ax.bar(data["RCA1"], data["count"], color=bar_colors[: len(data)])
    for i, y in enumerate(data["count"].tolist()):
        ax.text(i, y + max(1, y * 0.02), f"{int(y)}", ha="center", va="bottom", fontsize=9, color=_DARK_GREY)
    ax.set_title("RCA1 — June 2025 (Pareto)", pad=8, color=_DARK_BLUE)

    ax2 = ax.twinx()
    ax2.plot(
        data["RCA1"],
        data["cum_percent"],
        color=_PARETO_LINE,
        marker="o",
        linewidth=2.5,
        solid_capstyle="round",
        solid_joinstyle="round",
    )
    for i, y in enumerate(data["cum_percent"].tolist()):
        ax2.text(i, y + 1, f"{y:.0f}%", ha="center", va="bottom", fontsize=9, color=_DARK_GREY)

    for sp in ["top", "right", "left"]:
        ax.spines[sp].set_visible(False)
    ax.grid(False)
    ax.get_yaxis().set_visible(False)
    ax.spines["bottom"].set_color(_SOFT_GREY)
    ax.spines["bottom"].set_linewidth(1.25)
    ax.tick_params(axis="x", colors=_DARK_GREY)
    plt.setp(ax.get_xticklabels(), rotation=90, ha="center")

    for sp in ["top", "right", "left", "bottom"]:
        ax2.spines[sp].set_visible(False)
    ax2.set_ylim(0, 100)
    ax2.set_ylabel("")
    ax2.tick_params(axis="y", length=0)
    ax2.get_yaxis().set_visible(False)
    ax2.grid(False)

    return fig

def _plot_rca1_pareto(df: pd.DataFrame):
    st.pyplot(_pareto_fig(df))


# -------------------------------
# Portfolio-tab computations & plots — CACHED
# -------------------------------

@st.cache_data(show_spinner=False)
def _portfolio_list(cases: pd.DataFrame, comp: pd.DataFrame) -> List[str]:
    _, port_c, _ = _detect_cases_fields(cases)
    _, port_k, _, _ = _detect_complaints_fields(comp)
    ports = set()
    if port_c and port_c in cases.columns:
        ports |= set(cases[port_c].dropna().astype(str).unique().tolist())
    if port_k and port_k in comp.columns:
        ports |= set(comp[port_k].dropna().astype(str).unique().tolist())
    # Keep only specific tabs in required order if present
    desired = ["Chichester", "London", "Northwest", "Scotland"]
    return [p for p in desired if p in ports]

@st.cache_data(show_spinner=False)
def _portfolio_mom_series(cases: pd.DataFrame, comp: pd.DataFrame, portfolio: str) -> pd.DataFrame:
    _, port_c, date_c = _detect_cases_fields(cases)
    _, port_k, date_k, _ = _detect_complaints_fields(comp)
    if any(x is None for x in [port_c, date_c, port_k, date_k]):
        return pd.DataFrame(columns=["month", "per_1000"])

    cases = cases.copy(); comp = comp.copy()
    cases["_month"] = _build_month_column(cases, date_c)
    comp["_month"] = _build_month_column(comp, date_k, assume_year=2025 if date_k.lower() == "month" else None)

    months = _months_jan_to_aug_2025()
    cs = cases.loc[(cases[port_c] == portfolio) & (cases["_month"].isin(months))].groupby("_month").size().reindex(months, fill_value=0)
    cp = comp.loc[(comp[port_k] == portfolio) & (comp["_month"].isin(months))].groupby("_month").size().reindex(months, fill_value=0)
    per = (cp * 1000 / cs.replace(0, np.nan)).fillna(0.0).values
    pretty = [pd.Period(m).to_timestamp().strftime("%b-%y") for m in months]
    return pd.DataFrame({"month": pretty, "per_1000": np.round(per, 1)})

def _fig_portfolio_trend(df: pd.DataFrame, title: str):
    fig, ax = plt.subplots(figsize=(6.8, 3.2))
    ax.plot(df["month"], df["per_1000"], color="#d95f02", linewidth=2.5, marker="o")
    for x, y in zip(df["month"], df["per_1000"]):
        ax.text(x, y + 0.15, f"{y:.1f}", ha="center", va="bottom", fontsize=9, color=_DARK_GREY)
    idx = np.arange(len(df))
    z = np.polyfit(idx, df["per_1000"].astype(float), 1)
    ax.plot(df["month"], z[0]*idx + z[1], linestyle=":", linewidth=1.8, color="#7f7f7f")
    ax.set_title(title, color=_DARK_BLUE)
    ax.set_ylim(bottom=0)
    for sp in ["left", "right", "top"]:
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color(_SOFT_GREY)
    ax.spines["bottom"].set_linewidth(1.25)
    ax.get_yaxis().set_visible(False)
    ax.tick_params(axis="x", rotation=0, colors=_DARK_GREY)
    ax.set_xlabel(""); ax.set_ylabel(""); ax.grid(False)
    return fig

@st.cache_data(show_spinner=False)
def _rca_labels_for_subset(df: pd.DataFrame, use_ai: bool) -> Tuple[List[str], List[str]]:
    texts = df.astype(str).fillna("").tolist()
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
    r1_labels = _repair_rca1_from_rca2(r1_labels, r2_labels)
    return r1_labels, r2_labels

@st.cache_data(show_spinner=False)
def _reason_trend_df(comp: pd.DataFrame, portfolio: str, use_ai: bool) -> pd.DataFrame:
    _, port_k, date_k, desc_col = _detect_complaints_fields(comp)
    if any(x is None for x in [port_k, date_k, desc_col]):
        return pd.DataFrame()
    comp = comp.copy()
    comp["_month"] = _build_month_column(comp, date_k, assume_year=2025 if date_k.lower() == "month" else None)

    months = _months_jan_to_aug_2025()
    lab_months = [pd.Period(m).to_timestamp().strftime("%b’%y") for m in months]

    out = { "RCA1": ["Delay","Procedure","Communication","System","Incorrect/Incomplete information"] }
    for m, label in zip(months, lab_months):
        subset = comp.loc[(comp[port_k] == portfolio) & (comp["_month"] == m), [desc_col]]
        if subset.empty:
            out[label] = [0,0,0,0,0]
            continue
        r1, r2 = _rca_labels_for_subset(subset[desc_col], use_ai=use_ai)
        s = pd.Series(r1).value_counts()
        total = max(1, s.sum())
        vals = [
            round(100*s.get(k, 0)/total) for k in ["Delay","Procedure","Communication","System","Incorrect/Incomplete information"]
        ]
        out[label] = vals
    return pd.DataFrame(out)

def _fig_reason_trend(df: pd.DataFrame):
    if df.empty:
        return None
    fig, ax = plt.subplots(figsize=(6.6, 3.2))
    x = np.arange(len(df["RCA1"]))
    width = 0.26
    colors = ["#74c476", "#a1d99b", "#9ecae1"]
    labels = [c for c in df.columns if c != "RCA1"]
    for i, col in enumerate(labels):
        ax.bar(x + (i-1)*width, df[col].values, width=width, label=col, color=colors[i % len(colors)])
    ax.set_xticks(x)
    ax.set_xticklabels(df["RCA1"], rotation=0, color=_DARK_GREY)
    ax.set_ylim(0, 100)
    ax.legend(frameon=False, loc="upper right")
    ax.set_title("Reason Trend (Jan–Aug ’25) — % split", color=_DARK_BLUE)
    for sp in ["left", "right", "top"]:
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color(_SOFT_GREY)
    ax.spines["bottom"].set_linewidth(1.25)
    ax.get_yaxis().set_visible(False)
    ax.set_ylabel(""); ax.set_xlabel("")
    return fig

@st.cache_data(show_spinner=False)
def _delay_split_df(comp: pd.DataFrame, portfolio: str, use_ai: bool) -> pd.DataFrame:
    _, port_k, date_k, desc_col = _detect_complaints_fields(comp)
    if any(x is None for x in [port_k, date_k, desc_col]):
        return pd.DataFrame()
    comp = comp.copy()
    comp["_month"] = _build_month_column(comp, date_k, assume_year=2025 if date_k.lower() == "month" else None)

    months = _months_jan_to_aug_2025()
    labels = [pd.Period(m).to_timestamp().strftime("%b’%y") for m in months]

    rows = []
    for m, lab in zip(months, labels):
        subset = comp.loc[(comp[port_k] == portfolio) & (comp["_month"] == m), [desc_col]]
        if subset.empty:
            rows.append((lab, 0.0, 0.0))
            continue
        r1, r2 = _rca_labels_for_subset(subset[desc_col], use_ai=use_ai)
        delay_mask = [a == "Delay" or b in DELAY_RCA2 for a, b in zip(r1, r2)]
        if not any(delay_mask):
            rows.append((lab, 0.0, 0.0)); continue
        r2_delay = [r2[i] for i, keep in enumerate(delay_mask) if keep]
        total = len(r2_delay)
        ext = sum(1 for v in r2_delay if v in DELAY_EXTERNAL)
        apt = sum(1 for v in r2_delay if v in DELAY_APTIA)
        ext_pct = round(100*ext/total, 0) if total else 0.0
        apt_pct = round(100*apt/total, 0) if total else 0.0
        rows.append((lab, ext_pct, apt_pct))
    return pd.DataFrame(rows, columns=["month", "External delay %", "Aptia delay %"])

def _fig_delay_split(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(6.6, 3.0))
    x = np.arange(len(df["month"]))
    width = 0.35
    ax.bar(x - width/2, df["External delay %"], width=width, color="#74c476", label="External Delay")
    ax.bar(x + width/2, df["Aptia delay %"], width=width, color="#9ecae1", label="Aptia Delay")
    for i, (e, a) in enumerate(zip(df["External delay %"], df["Aptia delay %"])):
        ax.text(i - width/2, e + 1, f"{e:.0f}%", ha="center", va="bottom", fontsize=9, color=_DARK_GREY)
        ax.text(i + width/2, a + 1, f"{a:.0f}%", ha="center", va="bottom", fontsize=9, color=_DARK_GREY)
    ax.set_xticks(x); ax.set_xticklabels(df["month"], color=_DARK_GREY)
    ax.set_ylim(0, 100)
    ax.legend(frameon=False, loc="upper right")
    ax.set_title("Delay split — External vs Aptia (Jan–Aug ’25)", color=_DARK_BLUE)
    for sp in ["left", "right", "top"]:
        ax.spines[sp].setVisible = False
    ax.spines["bottom"].set_color(_SOFT_GREY); ax.spines["bottom"].set_linewidth(1.25)
    ax.get_yaxis().set_visible(False)
    ax.set_xlabel(""); ax.set_ylabel(""); ax.grid(False)
    return fig

@st.cache_data(show_spinner=False)
def _table_delay_80(comp: pd.DataFrame, portfolio: str, use_ai: bool) -> pd.DataFrame:
    _, port_k, date_k, desc_col = _detect_complaints_fields(comp)
    if any(x is None for x in [port_k, date_k, desc_col]):
        return pd.DataFrame(columns=["Delay 80% Reason", "Percentage contribution"])
    comp = comp.copy()
    comp["_month"] = _build_month_column(comp, date_k, assume_year=2025 if date_k.lower() == "month" else None)
    subset = comp.loc[(comp[port_k] == portfolio) & (comp["_month"] == "2025-06"), [desc_col]]
    if subset.empty:
        return pd.DataFrame(columns=["Delay 80% Reason", "Percentage contribution"])
    _, r2 = _rca_labels_for_subset(subset[desc_col], use_ai=use_ai)
    r2 = [v for v in r2 if v in DELAY_RCA2]
    if not r2:
        return pd.DataFrame(columns=["Delay 80% Reason", "Percentage contribution"])
    s = pd.Series(r2).value_counts().rename_axis("Delay 80% Reason").reset_index(name="count")
    total = max(1, s["count"].sum())
    s["Percentage contribution"] = (s["count"]*100/total).round(0)
    s["cum"] = s["Percentage contribution"].cumsum()
    s = s.loc[s["cum"] <= 80].drop(columns="cum")
    return s[["Delay 80% Reason", "Percentage contribution"]]

@st.cache_data(show_spinner=False)
def _table_procedure_80(comp: pd.DataFrame, portfolio: str, use_ai: bool) -> pd.DataFrame:
    _, port_k, date_k, desc_col = _detect_complaints_fields(comp)
    if any(x is None for x in [port_k, date_k, desc_col]):
        return pd.DataFrame(columns=["Procedure", "Percentage contribution"])
    comp = comp.copy()
    comp["_month"] = _build_month_column(comp, date_k, assume_year=2025 if date_k.lower() == "month" else None)
    subset = comp.loc[(comp[port_k] == portfolio) & (comp["_month"] == "2025-06"), [desc_col]]
    if subset.empty:
        return pd.DataFrame(columns=["Procedure", "Percentage contribution"])
    r1, r2 = _rca_labels_for_subset(subset[desc_col], use_ai=use_ai)
    proc = [r2[i] for i in range(len(r2)) if r1[i] == "Procedure"]
    if not proc:
        return pd.DataFrame(columns=["Procedure", "Percentage contribution"])
    s = pd.Series(proc).value_counts().rename_axis("Procedure").reset_index(name="count")
    total = max(1, s["count"].sum())
    s["Percentage contribution"] = (s["count"]*100/total).round(0)
    s["cum"] = s["Percentage contribution"].cumsum()
    s = s.loc[s["cum"] <= 80].drop(columns="cum")
    return s[["Procedure", "Percentage contribution"]]


# -------------------------------
# PowerPoint export (Overall)
# -------------------------------

def _add_df_table_to_slide(slide, df: pd.DataFrame, left_in: float, top_in: float, width_in: float):
    rows, cols = df.shape[0] + 1, df.shape[1]
    table = slide.shapes.add_table(rows, cols, Inches(left_in), Inches(top_in), Inches(width_in), Inches(0.8 + 0.3*rows)).table
    for j, col in enumerate(df.columns):
        cell = table.cell(0, j)
        cell.text = str(col)
        cell.text_frame.paragraphs[0].font.bold = True
        cell.text_frame.paragraphs[0].font.size = Pt(12)
        cell.text_frame.paragraphs[0].font.color.rgb = RGBColor(11, 61, 145)
    for i, (_, row) in enumerate(df.iterrows(), start=1):
        for j, col in enumerate(df.columns):
            val = row[col]
            if isinstance(val, float):
                if col == "per_1000" or col.endswith("percent"):
                    text = f"{val:.1f}"
                else:
                    s = f"{val:.3f}"
                    text = s.rstrip('0').rstrip('.') if '.' in s else f"{val:.0f}"
            else:
                text = str(val)
            cell = table.cell(i, j)
            cell.text = text
            p = cell.text_frame.paragraphs[0]
            p.font.size = Pt(11)
            p.font.color.rgb = RGBColor(51, 51, 51)
            p.alignment = PP_ALIGN.LEFT
    return table

def _build_ppt(table_df: pd.DataFrame, mom_df: pd.DataFrame, rca1_df: pd.DataFrame, rca2_df: pd.DataFrame) -> bytes:
    prs = Presentation()
    title_slide = prs.slides.add_slide(prs.slide_layouts[5])
    title = title_slide.shapes.add_textbox(Inches(0.5), Inches(0.3), Inches(9), Inches(0.8)).text_frame
    title.text = "Complaint analysis — Jun 2025"
    title.paragraphs[0].font.color.rgb = RGBColor(11, 61, 145)
    title.paragraphs[0].font.size = Pt(28)

    _add_df_table_to_slide(title_slide, table_df, left_in=0.5, top_in=1.2, width_in=5.0)
    fig_mom = _mom_line_fig(mom_df)
    buf_mom = BytesIO(); fig_mom.savefig(buf_mom, format="png", dpi=220, bbox_inches="tight"); plt.close(fig_mom)
    title_slide.shapes.add_picture(buf_mom, Inches(6.0), Inches(1.0), width=Inches(4.5))

    slide2 = prs.slides.add_slide(prs.slide_layouts[5])
    t2 = slide2.shapes.add_textbox(Inches(0.5), Inches(0.3), Inches(9), Inches(0.8)).text_frame
    t2.text = "June reasons — RCA"
    t2.paragraphs[0].font.color.rgb = RGBColor(11, 61, 145); t2.paragraphs[0].font.size = Pt(24)

    fig_pareto = _pareto_fig(rca1_df)
    buf_pareto = BytesIO(); fig_pareto.savefig(buf_pareto, format="png", dpi=220, bbox_inches="tight"); plt.close(fig_pareto)
    slide2.shapes.add_picture(buf_pareto, Inches(0.5), Inches(1.1), width=Inches(5.6))
    _add_df_table_to_slide(slide2, rca2_df, left_in=6.4, top_in=1.1, width_in=4.2)

    out = BytesIO(); prs.save(out); out.seek(0)
    return out.read()


# -------------------------------
# NEW: Insights helpers (purely additive; other tabs unchanged)
# -------------------------------

def _slope_ppm(values: List[float]) -> float:
    """Return slope in 'per_1000 points per month'."""
    if len(values) < 2:
        return 0.0
    x = np.arange(len(values))
    y = np.array(values, dtype=float)
    m, b = np.polyfit(x, y, 1)
    return float(np.round(m, 2))

def _slope_text(m: float) -> str:
    if abs(m) < 0.1:
        return f"flat over Jan–Aug ’25 (slope {m:+.2f} per-1k/month)"
    direction = "rising" if m > 0 else "falling"
    return f"{direction} (slope {m:+.2f} per-1k/month)"

@st.cache_data(show_spinner=False)
def _delay_split_overall(comp: pd.DataFrame, use_ai: bool) -> Tuple[float, float]:
    """Return average External vs Aptia delay split across Jan–Aug ’25 (overall)."""
    _, port_k, date_k, desc_col = _detect_complaints_fields(comp)
    if any(x is None for x in [date_k, desc_col]):
        return (0.0, 0.0)
    df = comp.copy()
    df["_month"] = _build_month_column(df, date_k, assume_year=2025 if date_k.lower() == "month" else None)
    months = _months_jan_to_aug_2025()

    ext_total = apt_total = all_delay = 0
    for m in months:
        subset = df.loc[df["_month"] == m, [desc_col]]
        if subset.empty:
            continue
        r1, r2 = _rca_labels_for_subset(subset[desc_col], use_ai=use_ai)
        for a, b in zip(r1, r2):
            if a == "Delay" or b in DELAY_RCA2:
                all_delay += 1
                if b in DELAY_EXTERNAL:
                    ext_total += 1
                if b in DELAY_APTIA:
                    apt_total += 1
    if all_delay == 0:
        return (0.0, 0.0)
    return (round(100*ext_total/all_delay, 0), round(100*apt_total/all_delay, 0))

def _render_insights(cases: pd.DataFrame, comp: pd.DataFrame, use_ai: bool, portfolios: List[str]) -> None:
    st.markdown("### What’s happening and why")

    # At a glance
    latest_month_str, latest_label = _latest_month_2025(cases, comp)
    if not latest_month_str:
        latest_month_str, latest_label = "2025-06", "Jun 2025"
    latest_tab = _portfolio_table_for_month(cases, comp, latest_month_str)
    mom = _mom_series(cases, comp)

    overall_latest = None
    delta_text = "—"
    if not latest_tab.empty:
        overall_latest = latest_tab.loc[latest_tab["portfolio"] == "Total", "per_1000"].values
        overall_latest = float(overall_latest[0]) if len(overall_latest) else None

        # delta vs previous month
        prev_month = pd.Period(latest_month_str).asfreq("M") - 1
        prev_tab = _portfolio_table_for_month(cases, comp, str(prev_month))
        if not prev_tab.empty:
            prev_total = prev_tab.loc[prev_tab["portfolio"] == "Total", "per_1000"].values
            if len(prev_total):
                d = float(np.round(overall_latest - float(prev_total[0]), 1))
                delta_text = f"{d:+.1f}"

    st.markdown(
        f"""
- **Overall complaints per 1,000** in **{latest_label}**: **{overall_latest if overall_latest is not None else 'n/a'}** (Δ vs prev: **{delta_text}**).
- **Trend**: {_slope_text(_slope_ppm(mom['per_1000'].tolist())) if not mom.empty else 'insufficient months to assess.'}
        """
    )

    # Drivers (June RCA)
    rca2_80, rca1_all = _rca_tables_for_june(comp, use_ai=use_ai)
    if not rca1_all.empty:
        top_rca1 = rca1_all.sort_values("count", ascending=False).head(3)
        bullets = "  \n".join([f"  • **{r}**" for r in top_rca1["RCA1"].tolist()])
        st.markdown("**Key drivers (June ’25):**")
        st.markdown(bullets)

    if not rca2_80.empty:
        st.markdown("_Focus reasons contributing ~80% of volume (RCA2, June ’25):_")
        st.dataframe(_style_table(rca2_80, formats={"count": "{:,.0f}", "percent": "{:.1f}", "cum_percent": "{:.1f}"}), use_container_width=True)

    # Delay split overall
    ext_pct, apt_pct = _delay_split_overall(comp, use_ai=use_ai)
    st.markdown(f"**Delay split (Jan–Aug ’25)** — External: **{ext_pct:.0f}%**, Aptia: **{apt_pct:.0f}%**.")

    # Watchlist portfolios (latest month per_1000)
    if not latest_tab.empty:
        watch = latest_tab.loc[latest_tab["portfolio"] != "Total"].copy()
        watch = watch.sort_values("per_1000", ascending=False).head(5)
        if not watch.empty:
            st.markdown("**Watchlist — highest per-1k in latest month**")
            st.dataframe(
                _style_table(watch[["portfolio", "per_1000", "complaints", "cases"]],
                             formats={"per_1000": "{:.1f}", "complaints": "{:,.0f}", "cases": "{:,.0f}"}),
                use_container_width=True,
            )

    # How to read
    st.markdown(
        """
**How to read this**
- *per-1k* = complaints per 1,000 cases.  
- Trend uses a simple line fit over Jan–Aug ’25 (*slope = monthly change in per-1k*).  
- “Focus reasons” list the smallest set of RCA2 themes that explains ~80% of June complaints.
        """
    )


# -------------------------------
# Streamlit entrypoint
# -------------------------------

def run(store: Dict, params: Dict, user_text: str = "") -> Tuple[str, pd.DataFrame]:
    """
    Tabs:
      - Insights (new)
      - Overall
      - Chichester, London, Northwest, Scotland (only these, if present)
    """
    # Hide sidebar / toolbar / parsed filters AND any alert (blue) boxes
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"], section[data-testid="stSidebar"] {display: none !important;}
        .stApp div[role="complementary"] {display:none !important;}
        [data-testid="stSidebarNav"] {display: none !important;}
        [data-testid="stToolbar"] {display: none !important;}
        div[data-testid="stExpander"] {display: none !important;}
        div[role="alert"] { display: none !important; }
        section[data-testid="stMain"] {padding-left: 1rem; padding-right: 1rem;}
        </style>
        """,
        unsafe_allow_html=True,
    )

    cases: pd.DataFrame = store.get("cases", pd.DataFrame()).copy()
    comp: pd.DataFrame = store.get("complaints", pd.DataFrame()).copy()
    use_ai = bool(os.getenv("OPENAI_API_KEY"))

    # latest month (for dynamic title & table)
    latest_month_str, latest_label = _latest_month_2025(cases, comp)

    portfolios = _portfolio_list(cases, comp)
    tabs = st.tabs(["Insights", "Overall"] + portfolios)

    # ----------------- Insights (new) -----------------
    with tabs[0]:
        _render_insights(cases, comp, use_ai=use_ai, portfolios=portfolios)

    # ----------------- Overall tab (unchanged logic) -----------------
    with tabs[1]:
        if latest_month_str:
            table = _portfolio_table_for_month(cases, comp, latest_month_str)
        else:
            table = _portfolio_table_for_month(cases, comp, "2025-06")

        mom = _mom_series(cases, comp)

        c1, c2 = st.columns((1.2, 1.0), gap="large")
        with c1:
            _header(f"Complaint analysis — {latest_label} (by portfolio)")
            if table.empty:
                st.info("No rows returned for the current filters.")
            else:
                st.dataframe(
                    _style_table(
                        table,
                        formats={"per_1000": "{:.1f}", "cases": "{:,.0f}", "complaints": "{:,.0f}"},
                    ),
                    use_container_width=True,
                )
        with c2:
            if not mom.empty:
                _plot_mom_line(mom)

        # ROW 2 (unchanged)
        _, port_k, _, _ = _detect_complaints_fields(comp)
        rca1_options = _RCA1_ALLOWED
        ports_options = portfolios if portfolios else []

        f1, f2 = st.columns((1.0, 1.0))
        with f1:
            sel_ports = st.multiselect(
                "Portfolio (local filter for RCA visuals)",
                options=ports_options,
                default=ports_options,
                key="compl_rca_row2_ports",
            )
        with f2:
            sel_rca1 = st.multiselect(
                "Complaint Reason — RCA1 (local filter)",
                options=rca1_options,
                default=rca1_options,
                key="compl_rca_row2_rca1",
            )

        comp_local = comp.copy()
        if port_k and sel_ports:
            comp_local = comp_local[comp_local[port_k].astype(str).isin([str(p) for p in sel_ports])]

        rca2_filtered, rca1_filtered = _rca_tables_for_june(comp_local, use_ai=use_ai)
        if not rca1_filtered.empty and sel_rca1:
            rca1_filtered = rca1_filtered[rca1_filtered["RCA1"].isin(sel_rca1)]

        c3, c4 = st.columns((1.05, 1.0), gap="large")
        with c3:
            if not rca1_filtered.empty:
                _plot_rca1_pareto(rca1_filtered)
            else:
                _header("RCA1 — June 2025 (Pareto)")
                st.info("No data for the selected filters.")

        with c4:
            _header("RCA2 — June 2025 (by Portfolio)")
            rca2_by_port = _rca2_table_by_portfolio_for_june(
                comp=comp,
                use_ai=use_ai,
                portfolios=sel_ports if sel_ports else ports_options,
                rca1_keep=sel_rca1 if sel_rca1 else rca1_options,
            )
            if rca2_by_port.empty:
                st.info("No June-2025 complaints for the selected filters.")
            else:
                st.dataframe(
                    _style_table(
                        rca2_by_port[["Portfolio", "RCA2", "count"]],
                        formats={"count": "{:,.0f}"},
                    ),
                    use_container_width=True,
                )

        # PPT (unchanged)
        rca2_all, rca1_all = _rca_tables_for_june(comp, use_ai=use_ai)
        if _PPT_READY and not table.empty and not mom.empty and not rca1_all.empty and not rca2_all.empty:
            ppt_bytes = _build_ppt(table, mom, rca1_all, rca2_all)
            st.download_button(
                "Download PPT",
                data=ppt_bytes,
                file_name="Complaint_Analysis_Jun2025.pptx",
                mime="application/vnd.openxmlformats-officedocument.presentationml.presentation",
                type="primary",
            )
        elif not _PPT_READY:
            st.caption("Install `python-pptx` to enable PPT download.")

    # ----------------- Portfolio tabs (unchanged) -----------------
    for i, portfolio in enumerate(portfolios, start=2):
        with tabs[i]:
            st.markdown(f"<h2 style='color:{_DARK_BLUE};margin:.3rem 0 1rem 0;'>{portfolio} — complaints analysis</h2>", unsafe_allow_html=True)

            t1, t2 = st.columns((1.1, 1.0), gap="large")
            with t1:
                df_trend = _portfolio_mom_series(cases, comp, portfolio)
                if not df_trend.empty:
                    st.pyplot(_fig_portfolio_trend(df_trend, f"{portfolio} trend Jan–Aug ’25"))
            with t2:
                df_reason_trend = _reason_trend_df(comp, portfolio, use_ai=use_ai)
                if not df_reason_trend.empty:
                    st.pyplot(_fig_reason_trend(df_reason_trend))

            b1, b2 = st.columns((1.1, 1.0), gap="large")
            with b1:
                df_delay = _delay_split_df(comp, portfolio, use_ai=use_ai)
                if not df_delay.empty:
                    st.pyplot(_fig_delay_split(df_delay))

            with b2:
                left, right = st.columns(2)
                with left:
                    _header("Delay — June (Top 80%)")
                    tdelay = _table_delay_80(comp, portfolio, use_ai=use_ai)
                    if not tdelay.empty:
                        st.dataframe(
                            _style_table(tdelay, formats={"Percentage contribution": "{:.0f}%"}),
                            use_container_width=True,
                        )
                with right:
                    _header("Procedure — June (Top 80%)")
                    tproc = _table_procedure_80(comp, portfolio, use_ai=use_ai)
                    if not tproc.empty:
                        st.dataframe(
                            _style_table(tproc, formats={"Percentage contribution": "{:.0f}%"}),
                            use_container_width=True,
                        )

    return ("", pd.DataFrame())
