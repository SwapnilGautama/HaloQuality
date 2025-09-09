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

def _months_13() -> List[str]:
    base = pd.period_range("2024-06", "2025-06", freq="M")
    return [str(p) for p in base]


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
# Core computations (overall)
# -------------------------------

def _portfolio_table_for_june(cases: pd.DataFrame, comp: pd.DataFrame) -> pd.DataFrame:
    id_c, port_c, date_c = _detect_cases_fields(cases)
    _, port_k, date_k, _ = _detect_complaints_fields(comp)

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
    comp["_month"] = _build_month_column(comp, date_k, assume_year=2025 if date_k.lower() == "month
