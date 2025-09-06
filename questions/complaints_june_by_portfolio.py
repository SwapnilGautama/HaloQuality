# questions/complaints_june_by_portfolio.py
from __future__ import annotations

import re
from typing import Any, Dict, Optional, Tuple, List

import pandas as pd

# Optional UI/plot libs – code guards if they aren't available
try:
    import streamlit as st  # type: ignore
except Exception:
    st = None  # graceful fallback

try:
    import altair as alt  # type: ignore
except Exception:
    alt = None  # graceful fallback


# ---------------- helpers ----------------

def _find_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    """Return the first existing column (case/space-insensitive) from candidates."""
    if df is None or df.empty:
        return None
    norm = {c.lower().strip(): c for c in df.columns}
    nospace = {k.replace(" ", ""): v for k, v in norm.items()}
    for cand in candidates:
        key = cand.lower().strip()
        if key in norm:
            return norm[key]
        k2 = key.replace(" ", "")
        if k2 in nospace:
            return nospace[k2]
    return None


def _month_key_from_datetime(series: pd.Series) -> pd.Series:
    """Convert datetimes to YYYY-MM strings."""
    s = pd.to_datetime(series, errors="coerce", dayfirst=True)
    return s.dt.to_period("M").astype(str)


def _month_key_from_month_name(series: pd.Series, year: int) -> pd.Series:
    """Convert 'June' -> 'YYYY-06' using a fixed year."""
    s = series.astype(str).str.strip()
    dt = pd.to_datetime("1 " + s + f" {year}", errors="coerce", dayfirst=True)
    return dt.dt.to_period("M").astype(str)


def _parse_month_from_params_or_text(params: Dict[str, Any], user_text: Optional[str]) -> Tuple[str, int]:
    """
    Decide the target month key and year.

    Priority:
      1) params['month'] in 'YYYY-MM' or 'Mon YYYY'
      2) user_text: 'June 2025' or 'June'
      3) default '2025-06'
    """
    # 1) explicit param
    if params and isinstance(params.get("month"), str):
        m = params["month"].strip()
        if re.match(r"^\d{4}-\d{2}$", m):  # 'YYYY-MM'
            return m, int(m[:4])
        m2 = re.match(r"^([A-Za-z]{3,})\s+(\d{4})$", m)  # 'Month YYYY'
        if m2:
            yr = int(m2.group(2))
            mk = pd.to_datetime(f"1 {m2.group(1)} {yr}", errors="coerce").to_period("M").astype(str)
            return mk, yr

    # 2) try user text
    if user_text:
        mt = re.search(r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\b(?:\s+(\d{4}))?", user_text, re.I)
        if mt:
            mon = mt.group(1)
            yr = int(mt.group(2)) if mt.group(2) else 2025
            mk = pd.to_datetime(f"1 {mon} {yr}", errors="coerce").to_period("M").astype(str)
            return mk, yr

    # 3) default: June 2025
    return "2025-06", 2025


def _clean_portfolio(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.replace(r"\s+", " ", regex=True).str.title()


def _month_range_ending_at(target_month_key: str, months: int = 13) -> List[str]:
    """Return a list of YYYY-MM covering the previous (months) ending at target (inclusive)."""
    end = pd.Period(target_month_key, freq="M")
    rng = pd.period_range(end=end, periods=months, freq="M")
    return [p.astype(str) for p in rng]


# --- RCA labelling (simple rules; tweak as needed) ---

_CAT_PATTERNS = {
    "Delay": [
        r"\bdelay(ed|s)?\b", r"\bsl[ao]\b", r"\bbacklog\b", r"\bawait(ing|ed)\b", r"\bturnaround\b", r"\btimescale\b"
    ],
    "Procedure": [
        r"\bprocedure\b", r"\bprocess\b", r"\bstandard\b", r"\bpolicy\b", r"\bcompliance\b"
    ],
    "Communication": [
        r"\bemail\b", r"\bletter\b", r"\bphone\b", r"\bcall\b", r"\bcontact\b", r"\bclarity\b", r"\binform(ation|ed)\b"
    ],
    "System": [
        r"\bsystem\b", r"\bit\b", r"\bportal\b", r"\btool\b", r"\btech(nical)?\b", r"\bbug\b", r"\berror\b"
    ],
    "Incorrect/Incomplete information": [
        r"\bincorrect\b", r"\bwrong\b", r"\bmis-?match\b", r"\bmissing\b", r"\bincomplete\b", r"\bnot provided\b"
    ],
}

_SUBREASON_PATTERNS = {
    "Delay - Manual calculation": [r"\bmanual\b", r"\bcalc(ulation|ulate)\b"],
    "Delay - Other 3rd Party": [r"\bthird\b", r"\b3rd\b", r"\bpartner\b", r"\bprovider\b"],
    "Delay - Pension set up": [r"\bpension\b", r"\bset ?up\b", r"\benrol(l)?ment\b"],
    "Delay - Postal Delay": [r"\bpost(al)?\b", r"\bmail\b", r"\broyal\s*mail\b"],
    "Delay - AVC": [r"\bavc\b"],
    "Delay - Requirement not checked": [r"\bcheck(ed|ing)?\b", r"\bverification\b", r"\bmissing\b"],
    "Delay - Case not created": [r"\bcase\b.*\bnot\b.*\bcreat(ed|e)\b"],
    "Delay - 2nd Review": [r"\bsecond\b.*\breview\b", r"\b2(nd)?\s*review\b"],
    "Delay - Trustee": [r"\btrustee\b"],
}

def _label_category(text: str) -> str:
    t = (text or "").lower()
    for cat, pats in _CAT_PATTERNS.items():
        for p in pats:
            if re.search(p, t):
                return cat
    return "Other"

def _label_subreason(text: str) -> str:
    t = (text or "").lower()
    for name, pats in _SUBREASON_PATTERNS.items():
        for p in pats:
            if re.search(p, t):
                return name
    return "Other"


# ---------------- main entry ----------------

def run(store: Dict[str, Any], params: Dict[str, Any], user_text: Optional[str] = None):
    """
    Complaint analysis for a single month, by portfolio.
    Join keys: (month_key, Portfolio)
      • Cases month_key: from 'Create Date' (fallbacks allowed)
      • Complaints month_key: from 'Date Complaint Received - DD/MM/YY', else 'Month'+assumed year
    Returns (title, subtitle), dataframe — same as the previous working version.
    Also renders charts/tables into Streamlit when available.
    """
    cases: pd.DataFrame = (store.get("cases") or pd.DataFrame()).copy()
    complaints: pd.DataFrame = (store.get("complaints") or pd.DataFrame()).copy()

    if cases.empty and complaints.empty:
        return "No data loaded.", pd.DataFrame()

    # target month and assumed year (if complaints only have Month names)
    target_month_key, assumed_year = _parse_month_from_params_or_text(params, user_text)

    # ---- CASES ----
    port_col_cases = _find_col(cases, ["Portfolio"])
    if not port_col_cases:
        return "Missing 'Portfolio' in cases.", pd.DataFrame()

    date_col_cases = _find_col(cases, ["Create Date", "Create Dt", "CreateDate", "Start Date", "Start Dt", "StartDate"])
    if not date_col_cases:
        return "Missing a usable date column in cases (e.g., 'Create Date').", pd.DataFrame()

    cases["_month_key"] = _month_key_from_datetime(cases[date_col_cases])
    cases["_portfolio"] = _clean_portfolio(cases[port_col_cases])

    cases_target = cases.loc[cases["_month_key"] == target_month_key].copy()
    cases_by_port = (
        cases_target.groupby("_portfolio", dropna=False)
        .size()
        .reset_index(name="cases")
    )

    # ---- COMPLAINTS ----
    port_col_comp = _find_col(complaints, ["Portfolio"])
    if not port_col_comp:
        return "Missing 'Portfolio' in complaints.", pd.DataFrame()

    comp_date_col = _find_col(complaints, ["Date Complaint Received - DD/MM/YY"])
    if comp_date_col:
        complaints["_month_key"] = _month_key_from_datetime(complaints[comp_date_col])
    else:
        month_name_col = _find_col(complaints, ["Month", "Report Month", "Complaint Month"])
        if not month_name_col:
            return ("Missing date in complaints. Provide 'Date Complaint Received - DD/MM/YY' or 'Month'.",
                    pd.DataFrame())
        complaints["_month_key"] = _month_key_from_month_name(complaints[month_name_col], assumed_year)

    complaints["_portfolio"] = _clean_portfolio(complaints[port_col_comp])
    comp_target = complaints.loc[complaints["_month_key"] == target_month_key].copy()

    comps_by_port = (
        comp_target.groupby("_portfolio", dropna=False)
        .size()
        .reset_index(name="complaints")
    )

    # ---- Join (portfolio) and compute per_1000 ----
    out = pd.merge(cases_by_port, comps_by_port, how="outer", on="_portfolio").fillna(0)
    out["cases"] = out["cases"].astype(int, errors="ignore")
    out["complaints"] = out["complaints"].astype(int, errors="ignore")
    out["per_1000"] = (out["complaints"] / out["cases"].where(out["cases"] != 0, pd.NA)) * 1000
    out["per_1000"] = out["per_1000"].round(2)
    out = out.rename(columns={"_portfolio": "portfolio"}).sort_values(
        ["per_1000", "portfolio"], ascending=[False, True], na_position="last"
    )
    tot_cases = int(out["cases"].sum())
    tot_comps = int(out["complaints"].sum())
    tot_per_1000 = round((tot_comps / tot_cases) * 1000, 2) if tot_cases else 0.0

    # ---- Build MoM: last 13 months, fill 0s ----
    months = _month_range_ending_at(target_month_key, months=13)
    # cases monthly totals
    cases_month = (
        cases.groupby("_month_key").size().rename("cases").reindex(months).fillna(0).astype(int).reset_index()
        if not cases.empty else pd.DataFrame({"index": months, "cases": 0})
    )
    cases_month = cases_month.rename(columns={"index": "month_key"})

    # complaints monthly totals (use whichever mapping we already created)
    # If Date Complaint Received was missing, we already created month_key via Month+year (only for assumed year).
    if comp_date_col:
        comp_month_all = complaints.copy()
    else:
        # We only know June 2025 reliably; assume 0 elsewhere if Month names don't exist for other months.
        comp_month_all = complaints.copy()

    comp_by_month = (
        comp_month_all.groupby("_month_key").size().rename("complaints")
        .reindex(months).fillna(0).astype(int).reset_index()
    )
    comp_by_month = comp_by_month.rename(columns={"index": "month_key"})

    mom = pd.merge(cases_month, comp_by_month, on="month_key", how="outer").fillna(0)
    mom["per_1000"] = (mom["complaints"] / mom["cases"].where(mom["cases"] != 0, pd.NA)) * 1000
    mom["per_1000"] = mom["per_1000"].fillna(0).round(2)
    mom["label"] = pd.PeriodIndex(mom["month_key"], freq="M").strftime("%b '%y")

    # ---- RCA categorisation (June) ----
    reason_col = _find_col(complaints, ["Brief Description - RCA done by admin", "Brief Description", "RCA"])
    june_reasons = pd.DataFrame()
    reason_trend = pd.DataFrame()
    if reason_col:
        # label categories for every complaint (only use rows we have month mapping for)
        complaints["_rca_cat"] = complaints[reason_col].astype(str).map(_label_category)
        complaints["_rca_sub"] = complaints[reason_col].astype(str).map(_label_subreason)

        # 3-month trend (Apr/May/Jun of the same year)
        yr = int(target_month_key[:4])
        last3 = [pd.to_datetime(f"1 {m} {yr}").to_period("M").astype(str) for m in ["Apr", "May", "Jun"]]
        tmp = (
            complaints.loc[complaints["_month_key"].isin(last3)]
            .groupby(["_month_key", "_rca_cat"])
            .size()
            .rename("count")
            .reset_index()
        )
        # percent share per month
        if not tmp.empty:
            tmp["month_label"] = pd.PeriodIndex(tmp["_month_key"], freq="M").strftime("%b %y")
            totals = tmp.groupby("_month_key")["count"].transform("sum")
            tmp["share"] = (tmp["count"] / totals).fillna(0) * 100
            reason_trend = tmp

        # June 80% table (by category and by subreason)
        june_only = complaints.loc[complaints["_month_key"] == target_month_key]
        if not june_only.empty:
            cat_counts = (
                june_only.groupby("_rca_cat").size().rename("count").reset_index().sort_values("count", ascending=False)
            )
            if not cat_counts.empty:
                cat_counts["pct"] = (cat_counts["count"] / cat_counts["count"].sum()) * 100
                cat_counts["cum_pct"] = cat_counts["pct"].cumsum()
                june_reasons = cat_counts

    # ---- Render (optional) ----
    if st is not None:
        # Header
        st.subheader(f"Complaint analysis — {pd.Period(target_month_key).strftime('%b %Y')} (by portfolio)")
        st.caption(f"Total: cases={tot_cases:,}, complaints={tot_comps:,}, per_1000={tot_per_1000}")

        # Table
        st.dataframe(out, use_container_width=True)

        # MoM trend
        if alt is not None and not mom.empty:
            base = alt.Chart(mom).transform_calculate(
                tooltip_title="'Complaints per 1,000'"
            )

            line = base.mark_line(interpolate="monotone", strokeWidth=3, color="#86c5da").encode(
                x=alt.X("label:N", title="Month"),
                y=alt.Y("per_1000:Q", title="Complaints per 1,000"),
                tooltip=[alt.Tooltip("label", title="Month"), alt.Tooltip("per_1000", title="Per 1,000")]
            )

            st.subheader("Complaints per 1,000 — trend (past 12 months)")
            st.altair_chart(line.properties(height=260), use_container_width=True)

        # Reason trend (Apr/May/Jun)
        if alt is not None and not reason_trend.empty:
            st.subheader("Reason trend (Apr–Jun)")
            bars = alt.Chart(reason_trend).mark_bar().encode(
                x=alt.X("month_label:N", title=None),
                y=alt.Y("share:Q", title="Percentage"),
                color=alt.Color("_rca_cat:N", title="Category",
                                scale=alt.Scale(scheme="pastel1")),
                tooltip=[
                    alt.Tooltip("month_label", title="Month"),
                    alt.Tooltip("_rca_cat", title="Category"),
                    alt.Tooltip("share", title="%")
                ]
            )
            st.altair_chart(bars.properties(height=280), use_container_width=True)

        # June 80% reasons
        if not june_reasons.empty:
            st.subheader("June reasons — 80% coverage")
            # keep rows until cumulative percentage reaches 80%
            jr = june_reasons.copy()
            jr80 = jr.loc[jr["cum_pct"] <= 80]
            if jr80.empty:  # if first row already >80%, still show it
                jr80 = jr.head(1)
            jr80 = jr80.rename(columns={"_rca_cat": "Category", "count": "Complaints", "pct": "Percent", "cum_pct": "Cumulative %"})
            jr80["Percent"] = jr80["Percent"].round(1)
            jr80["Cumulative %"] = jr80["Cumulative %"].round(1)
            st.dataframe(jr80, use_container_width=True)

    # ---- Final return (preserve prior interface) ----
    title = f"Complaint analysis — {pd.Period(target_month_key).strftime('%b %Y')} (by portfolio)"
    subtitle = f"Total: cases={tot_cases:,}, complaints={tot_comps:,}, per_1000={tot_per_1000}"
    return (title, subtitle), out
