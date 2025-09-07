# -*- coding: utf-8 -*-
# questions/complaints_june_by_portfolio.py
from __future__ import annotations

from typing import Dict, Optional, Tuple, List
import re
import calendar
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

# -------------------------------------------------------------------
# helpers
# -------------------------------------------------------------------

def _find_col(df: pd.DataFrame, needles: List[str]) -> Optional[str]:
    """Return the first column whose lowercase name contains ANY of the
    substrings in `needles` (also lowercase)."""
    low = {c.lower(): c for c in df.columns}
    for lc, orig in low.items():
        if any(n in lc for n in needles):
            return orig
    return None


def _month_key(year: int, month: int) -> str:
    return f"{year}-{month:02d}"


def _to_month(df: pd.DataFrame, col: str) -> pd.Series:
    """Coerce to pandas Period (month) then to string 'YYYY-MM'."""
    x = pd.to_datetime(df[col], errors="coerce").dt.to_period("M").astype(str)
    return x


def _coerce_months_from_text(s: pd.Series, default_year: int) -> pd.Series:
    """
    Handle cases like 'June', 'Jun', etc., assuming `default_year`.
    Returns 'YYYY-MM' strings or NaN.
    """
    def parse_one(v):
        if pd.isna(v):
            return np.nan
        txt = str(v).strip()
        # try direct date
        dt = pd.to_datetime(txt, errors="coerce", dayfirst=True)
        if not pd.isna(dt):
            return str(dt.to_period("M"))
        # try month names
        m = re.match(r"^(jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)", txt, re.I)
        if m:
            month = m.group(1).lower()[:3]
            month_num = {
                "jan":1,"feb":2,"mar":3,"apr":4,"may":5,"jun":6,
                "jul":7,"aug":8,"sep":9,"oct":10,"nov":11,"dec":12
            }[month]
            return f"{default_year}-{month_num:02d}"
        return np.nan
    return s.map(parse_one)


def _soft_axes(ax):
    """Hide frame & y ticks for minimalist visuals (pastel style)."""
    for spine in ["top", "right", "left", "bottom"]:
        ax.spines[spine].set_visible(False)
    ax.grid(False)
    ax.tick_params(axis="y", left=False, labelleft=False)


def _annotate_line(ax, xs, ys):
    for x, y in zip(xs, ys):
        ax.text(x, y, f"{y:.2f}", ha="center", va="bottom", fontsize=9)


# -------------------------------------------------------------------
# RCA mapping
# -------------------------------------------------------------------

# Higher level — RCA1 (broad umbrellas, kept as-is)
RCA1_MAP: List[Tuple[str, str]] = [
    (r"\bdelay|late|slow|backlog|waiting|manual calc|postal|2nd review|not created|not checked\b", "Delay"),
    (r"\bprocedure|process|timescale|scheme rule|factor change|setup\b", "Procedure"),
    (r"\bcommunicat|letter|email|call|clarity\b", "Communication"),
    (r"\bsystem|portal|workflow|it error|glitch\b", "System"),
    (r"\bincorrect|incomplete|wrong|missing|evidence|document|doc|form\b", "Incorrect/Incomplete information"),
]

# Deeper level — RCA2 (built from "Brief Description - RCA done by admin")
# You can extend this easily; left side is a pattern, right side is the category label.
RCA2_MAP: List[Tuple[str, str]] = [
    (r"\bscheme rule", "Scheme Rules"),
    (r"\bstandard time|timescale|sla\b", "Aptia standard Timescale"),
    (r"\bdrop in value|factor change", "Drop in value / factor change"),
    (r"\bdeath benefit|death\s*benefits\s*payout", "Death benefits payout"),
    (r"\boverpayment\b", "Overpayment"),
    (r"\bpension increase\b", "Pension Increase"),
    (r"\btransfer doc|transfer\s*document", "Transfer Documentation"),
    (r"\bmanual calc|manual calculation", "Manual calculation"),
    (r"\bpostal delay|post\b", "Postal Delay"),
    (r"\brequirement not checked|not checked|validation", "Requirement not checked"),
    (r"\bcase not created|missing case\b", "Case not created"),
    (r"\b2nd review|second review\b", "2nd Review"),
    (r"\btrustee\b", "Trustee"),
    (r"\bavc\b", "Delay – AVC"),
    (r"\bcommunicat|lack of clarity|unclear\b", "Communication — lack of clarity"),
    (r"\bdoc(ument)? missing|missing (doc|document|evidence|form)", "Documentation Missing"),
    (r"\bsetup|set up\b", "Pension set up"),
]

def _classify(text: str, mapping: List[Tuple[str, str]], default_label="Other") -> str:
    if not isinstance(text, str):
        return default_label
    t = text.lower()
    for pat, label in mapping:
        if re.search(pat, t):
            return label
    return default_label


# -------------------------------------------------------------------
# main compute
# -------------------------------------------------------------------

def _build_tables_and_charts(store, month_key: str = "2025-06"):
    """Do all the work; returns the DF we display in Row 1 (for app to cache/return)."""
    # ---- read store ----
    cases = store.get("cases", pd.DataFrame()).copy()
    complaints = store.get("complaints", pd.DataFrame()).copy()

    if cases.empty or complaints.empty:
        st.warning("No cases or complaints data.")
        return pd.DataFrame()

    # ---- locate columns robustly ----
    # portfolio
    port_c_cases = _find_col(cases, ["portfolio"])
    port_c_compl = _find_col(complaints, ["portfolio"])
    if port_c_cases is None:
        st.error("Missing 'Portfolio' in cases.")
        return pd.DataFrame()
    if port_c_compl is None:
        # It's ok if complaints has no portfolio — join later still works but table looks better with cases' portfolios
        port_c_compl = port_c_cases

    # case id (for sanity checks if needed)
    case_id_col = _find_col(cases, ["case id", "id"])

    # create date for cases
    create_cases = _find_col(cases, ["create date"])
    if create_cases is None:
        # fallback: start date
        create_cases = _find_col(cases, ["start date"])
    if create_cases is None:
        st.error("Missing Create date in cases.")
        return pd.DataFrame()

    # complaint date/month in complaints (Month field is often present)
    comp_date_col = None
    for cand in [["date complaint received", "dd/mm/yy"], ["complaint date", "month"], ["month"]]:
        comp_date_col = _find_col(complaints, cand)
        if comp_date_col:
            break
    # brief description field for RCA2 text analysis
    rca_text_col = _find_col(complaints, ["brief description", "rca done by admin"])

    # ---- month keys ----
    # Cases use real dates
    cases["_month"] = _to_month(cases, create_cases)
    # Complaints can be actual dates or just "June"
    if comp_date_col is None:
        complaints["_month"] = np.nan
    else:
        # Try datetime first; if fails, allow month words and coerce to 2025
        tmp = pd.to_datetime(complaints[comp_date_col], errors="coerce")
        if tmp.notna().any():
            complaints["_month"] = tmp.dt.to_period("M").astype(str)
        else:
            complaints["_month"] = _coerce_months_from_text(complaints[comp_date_col], 2025)

    # ---- June (Row 1 table per portfolio) ----
    cases_jun = cases.loc[cases["_month"] == month_key].groupby(port_c_cases).size().rename("cases")
    compl_jun = complaints.loc[complaints["_month"] == month_key].groupby(port_c_compl).size().rename("complaints")

    # Keep the portfolio index from CASES (so we don't pick up 'None' from complaints)
    df1 = (
        pd.DataFrame(cases_jun)
        .join(compl_jun, how="left")
        .fillna({"complaints": 0})
        .astype({"complaints": int})
        .reset_index()
        .rename(columns={port_c_cases: "portfolio"})
    )

    # per 1000
    df1["per_1000"] = np.where(df1["cases"] > 0, df1["complaints"] / df1["cases"] * 1000, np.nan)

    # Add TOTAL row at the top
    total_row = {
        "portfolio": "Total",
        "cases": int(df1["cases"].sum()),
        "complaints": int(df1["complaints"].sum()),
        "per_1000": (df1["complaints"].sum() / df1["cases"].sum() * 1000) if df1["cases"].sum() > 0 else np.nan,
    }
    df1 = pd.concat([pd.DataFrame([total_row]), df1], ignore_index=True)

    # ---- MoM chart (Jan–Jun 2025) overall per 1000 ----
    months = [ _month_key(2025, m) for m in range(1, 7) ]  # Jan..Jun 2025
    cases_m = cases.groupby("_month").size().reindex(months).fillna(0).astype(int)
    compl_m = complaints.groupby("_month").size().reindex(months).fillna(0).astype(int)
    per1000_m = np.where(cases_m > 0, compl_m / cases_m * 1000, 0.0)

    # ---- draw Row 1 ----
    left, right = st.columns([1.2, 1.2])

    with left:
        st.subheader("Complaint analysis — Jun 2025 (by portfolio)")
        st.caption(f"Total: cases={total_row['cases']:,}, complaints={total_row['complaints']:,}, per_1000={total_row['per_1000']:.2f}")
        st.dataframe(
            df1[["portfolio", "cases", "complaints", "per_1000"]],
            use_container_width=True
        )

    with right:
        fig, ax = plt.subplots(figsize=(6.5, 3.0))
        xs = list(range(len(months)))
        ax.plot(xs, per1000_m, marker="o", linewidth=2.5)
        _soft_axes(ax)
        # x labels Jan..Jun
        ax.set_xticks(xs, [calendar.month_abbr[int(m.split("-")[1])] for m in months], fontsize=10)
        _annotate_line(ax, xs, per1000_m)
        ax.set_title("Complaints per 1,000 — MoM (Jan–Jun ’25)", pad=10)
        st.pyplot(fig, clear_figure=True)

    # ---- Row 2: RCA2 (left table) + RCA1 (right chart) for June 2025 ----
    st.markdown("---")
    left2, right2 = st.columns([1.2, 1.2])

    # RCA2 on June data only, using deep text mapping
    with left2:
        st.subheader("RCA2 — Top 80% (June 2025)")
        if rca_text_col is None:
            st.info("No 'Brief Description - RCA done by admin' field in complaints.")
            rca2_df = pd.DataFrame(columns=["RCA2", "count", "percent", "cum_percent"])
        else:
            june_mask = complaints["_month"].eq(month_key)
            rcatext = complaints.loc[june_mask, rca_text_col].astype(str)
            mapped = rcatext.map(lambda x: _classify(x, RCA2_MAP, default_label="Other"))
            rca2 = mapped.value_counts(dropna=False).reset_index()
            rca2.columns = ["RCA2", "count"]
            rca2["percent"] = rca2["count"] / rca2["count"].sum() * 100
            rca2 = rca2.sort_values("count", ascending=False, ignore_index=True)
            rca2["cum_percent"] = rca2["percent"].cumsum()
            # top 80%
            rca2_df = rca2.loc[rca2["cum_percent"] <= 80]
            if rca2_df.empty and not rca2.empty:
                rca2_df = rca2.head(5)
        st.dataframe(rca2_df, use_container_width=True)

    # RCA1 on June data (broad umbrellas; unchanged)
    with right2:
        st.subheader("RCA1 — June 2025")
        # compute from comments + fallback to 'Other'
        if rca_text_col is None:
            st.info("No 'Brief Description - RCA done by admin' field for RCA1.")
        else:
            june_mask = complaints["_month"].eq(month_key)
            rcatext = complaints.loc[june_mask, rca_text_col].astype(str)
            def map_rca1(x: str) -> str:
                for pat, lab in RCA1_MAP:
                    if re.search(pat, x.lower()):
                        return lab
                return "Other"
            rca1 = rcatext.map(map_rca1).value_counts().reindex(
                ["Incorrect/Incomplete information", "Delay", "Procedure", "Communication", "System", "Other"],
                fill_value=0
            )
            # pastel bar
            fig2, ax2 = plt.subplots(figsize=(6.5, 3.2))
            bars = ax2.bar(rca1.index, rca1.values)
            _soft_axes(ax2)
            ax2.set_xticklabels(rca1.index, rotation=20, ha="right")
            ax2.set_title("June reasons (RCA1)")
            # labels on bars
            for b in bars:
                ax2.text(b.get_x() + b.get_width()/2, b.get_height(), f"{int(b.get_height())}",
                         ha="center", va="bottom", fontsize=9)
            st.pyplot(fig2, clear_figure=True)

    # Return Row 1 table for app’s caching/logging
    return df1


# -------------------------------------------------------------------
# public entrypoint expected by app
# -------------------------------------------------------------------

def run(store: Dict[str, pd.DataFrame], params: Dict, user_text: Optional[str] = None):
    """
    Streamlit renders here. Always return (title, df) so the caller
    never gets a None return (avoids 'cannot unpack non-iterable NoneType').
    """
    st.header("Complaint analysis — June 2025 (by portfolio)")

    # We keep June 2025 as default; the router already parses month but we
    # deliberately DO NOT change what’s working in Row 1.
    month_key = "2025-06"

    df_out = _build_tables_and_charts(store, month_key=month_key)
    title = "complaints_june_by_portfolio"
    if df_out is None or df_out.empty:
        # Still return a harmless empty frame to keep caller happy
        return title, pd.DataFrame(columns=["portfolio", "cases", "complaints", "per_1000"])
    return title, df_out
