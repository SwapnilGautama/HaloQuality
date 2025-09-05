# question_engine/nl_router.py
from __future__ import annotations

import math
import pandas as pd
import streamlit as st
from typing import Dict, Tuple, Optional

from .parser import parse

# ------------- helpers --------------------------------------------------------

def _month_between(s: str, e: str) -> pd.DatetimeIndex:
    s2 = pd.to_datetime(s + "-01")
    e2 = pd.to_datetime(e + "-01")
    return pd.date_range(s2, e2, freq="MS")

def _ensure_month(df: pd.DataFrame, date_col: str, out_col: str = "month") -> pd.DataFrame:
    out = df.copy()
    if out_col not in out.columns:
        out[out_col] = pd.to_datetime(out[date_col]).dt.strftime("%Y-%m")
    return out

def _filters_summary(portfolio: Optional[str], process: Optional[str], months: Optional[Tuple[str, str]]):
    bits = []
    if portfolio: bits.append(f"Portfolio: **{portfolio}**")
    if process: bits.append(f"Process: **{process}**")
    if months: bits.append(f"Months: **{months[0]} → {months[1]}**")
    return " | ".join(bits) or "_no filters_"

# ------------- intent handlers -----------------------------------------------

def _h_complaints_per_1000(store, pr, months_window: Optional[int] = 3):
    cases = store["cases"]
    complaints = store["complaints"]

    # canonical columns
    if "Portfolio_std" not in cases.columns:
        st.error("Cases is missing 'Portfolio_std'.")
        return

    cases = _ensure_month(cases, "Create Date" if "Create Date" in cases.columns else "Create_Date")
    complaints = _ensure_month(complaints, "Report_Date" if "Report_Date" in complaints.columns else "Report Date")

    df_cases = cases.copy()
    df_comps = complaints.copy()

    # optional filters
    if pr.portfolio:
        df_cases = df_cases[df_cases["Portfolio_std"].str.casefold() == pr.portfolio.casefold()]
        df_comps = df_comps[df_comps["Portfolio_std"].str.casefold() == pr.portfolio.casefold()]

    # process can be either ProcessName (cases) or Parent_Case_Type (complaints)
    if pr.process:
        if "ProcessName" in df_cases.columns:
            df_cases = df_cases[df_cases["ProcessName"].str.casefold() == pr.process.casefold()]
        if "Parent_Case_Type" in df_comps.columns:
            df_comps = df_comps[df_comps["Parent_Case_Type"].str.casefold() == pr.process.casefold()]

    # month filter
    if pr.months:
        rng = {m.strftime("%Y-%m") for m in _month_between(*pr.months)}
        df_cases = df_cases[df_cases["month"].isin(rng)]
        df_comps = df_comps[df_comps["month"].isin(rng)]
    elif months_window:
        # last N months present in cases/complaints overlap
        inter = sorted(set(df_cases["month"]).intersection(set(df_comps["month"])))
        inter = inter[-months_window:]
        df_cases = df_cases[df_cases["month"].isin(inter)]
        df_comps = df_comps[df_comps["month"].isin(inter)]

    # unique case ids per month
    case_count = (
        df_cases.drop_duplicates(["Case ID" if "Case ID" in df_cases.columns else "CaseID"])
        .groupby("month")
        .size()
        .rename("unique_cases")
        .reset_index()
    )

    comp_count = df_comps.groupby("month").size().rename("complaints").reset_index()

    joined = pd.merge(case_count, comp_count, on="month", how="inner")
    if joined.empty:
        st.info("No overlapping data for the selected filters.")
        return

    joined["complaints_per_1000"] = (joined["complaints"] / joined["unique_cases"]) * 1000

    st.subheader("Complaints per 1,000 cases (MoM)")
    st.caption(_filters_summary(pr.portfolio, pr.process, pr.months))
    st.line_chart(joined.set_index("month")[["complaints_per_1000"]])

    with st.expander("Data", expanded=False):
        st.dataframe(joined.sort_values("month"))

def _h_rca1_by_portfolio(store, pr):
    df = store["complaints"].copy()
    df = _ensure_month(df, "Report_Date" if "Report_Date" in df.columns else "Report Date")

    if pr.process and "Parent_Case_Type" in df.columns:
        df = df[df["Parent_Case_Type"].str.casefold() == pr.process.casefold()]
    if pr.months:
        rng = {m.strftime("%Y-%m") for m in _month_between(*pr.months)}
        df = df[df["month"].isin(rng)]

    if "RCA1" not in df.columns:
        st.error("Complaints is missing 'RCA1'.")
        return

    piv = (
        df.pivot_table(
            index="Portfolio_std",
            columns="RCA1",
            values="Parent_Case_Type",  # any column; we only need counts
            aggfunc="count",
            fill_value=0,
        )
        .sort_index()
    )

    if piv.empty:
        st.info("No data for the selected filters.")
        return

    st.subheader("RCA1 mix by portfolio")
    st.caption(_filters_summary(pr.portfolio, pr.process, pr.months))
    st.bar_chart(piv / piv.sum(axis=1).replace({0: math.nan}))

    with st.expander("Data", expanded=False):
        st.dataframe((piv.T / piv.sum(axis=1)).T.round(3))

def _h_unique_cases(store, pr):
    cases = store["cases"].copy()
    cases = _ensure_month(cases, "Create Date" if "Create Date" in cases.columns else "Create_Date")

    if pr.portfolio:
        cases = cases[cases["Portfolio_std"].str.casefold() == pr.portfolio.casefold()]
    if pr.process and "ProcessName" in cases.columns:
        cases = cases[cases["ProcessName"].str.casefold() == pr.process.casefold()]
    if pr.months:
        rng = {m.strftime("%Y-%m") for m in _month_between(*pr.months)}
        cases = cases[cases["month"].isin(rng)]

    key = ["month"]
    if "ProcessName" in cases.columns:
        key.append("ProcessName")
    if "Portfolio_std" in cases.columns:
        key.append("Portfolio_std")

    out = (
        cases.drop_duplicates(["Case ID" if "Case ID" in cases.columns else "CaseID"])
        .groupby(key)
        .size()
        .rename("unique_cases")
        .reset_index()
        .sort_values(key)
    )

    st.subheader("Unique cases")
    st.caption(_filters_summary(pr.portfolio, pr.process, pr.months))
    st.dataframe(out, use_container_width=True)

def _h_fpa_fail_drivers(store, pr):
    fpa = store.get("fpa")
    if fpa is None or fpa.empty:
        st.info("No FPA data loaded.")
        return

    fpa = _ensure_month(fpa, "Review_Date" if "Review_Date" in fpa.columns else fpa.columns[0])

    # filter to fails
    rr_col = "Review_Result" if "Review_Result" in fpa.columns else "review_result"
    if rr_col not in fpa.columns:
        st.error("FPA file is missing 'Review_Result' column.")
        return
    fpa = fpa[fpa[rr_col].str.lower().eq("fail")]

    if pr.portfolio and "Portfolio_std" in fpa.columns:
        fpa = fpa[fpa["Portfolio_std"].str.casefold() == pr.portfolio.casefold()]
    if pr.process and "ProcessName" in fpa.columns:
        fpa = fpa[fpa["ProcessName"].str.casefold() == pr.process.casefold()]
    if pr.months:
        rng = {m.strftime("%Y-%m") for m in _month_between(*pr.months)}
        fpa = fpa[fpa["month"].isin(rng)]

    label_col = None
    for c in ["FPA_Label", "Label", "RCA1"]:  # prefer your new labeller output
        if c in fpa.columns:
            label_col = c
            break

    if label_col:
        top = (
            fpa.groupby(label_col).size().rename("fail_count").sort_values(ascending=False).head(20).reset_index()
        )
        st.subheader("Top drivers (FPA fail labels)")
        st.caption(_filters_summary(pr.portfolio, pr.process, pr.months))
        st.bar_chart(top.set_index(label_col)["fail_count"])
        with st.expander("Data", expanded=False):
            st.dataframe(top)
    else:
        # crude textual top-words extraction if no labels exist
        text_col = None
        for c in ["Case_Comment", "Comments", "Case comment", "comment"]:
            if c in fpa.columns:
                text_col = c
                break
        if not text_col:
            st.info("FPA fail drivers: no label or comment field found.")
            return

        import collections, re as _re

        cnt = collections.Counter()
        for t in fpa[text_col].dropna().astype(str).tolist():
            tokens = [w for w in _re.findall(r"[a-zA-Z']{3,}", t.lower()) if w not in {"the","and","for","with","from","that"}]
            cnt.update(tokens)

        top = pd.DataFrame(cnt.most_common(25), columns=["token", "count"])
        st.subheader("Top tokens in FPA fail comments")
        st.caption(_filters_summary(pr.portfolio, pr.process, pr.months))
        st.bar_chart(top.set_index("token")["count"])
        with st.expander("Data", expanded=False):
            st.dataframe(top)

# ------------- public entrypoint ---------------------------------------------

def run_nl(store: Dict[str, pd.DataFrame]):
    """
    Minimal chat-style NL interface. `store` is the dict returned by core.data_store.load_store().
    """
    st.subheader("Halo Quality — Chat")
    st.caption("Ask me about **cases**, **complaints** (incl. RCA), or **first-pass accuracy**.")

    # tips
    with st.sidebar:
        st.markdown("### Data status")
        try:
            c_rows = len(store.get("cases", pd.DataFrame()))
            comp_rows = len(store.get("complaints", pd.DataFrame()))
            fpa_rows = len(store.get("fpa", pd.DataFrame()))
        except Exception:
            c_rows = comp_rows = fpa_rows = 0
        st.write(f"Cases rows: **{c_rows:,}**")
        st.write(f"Complaints rows: **{comp_rows:,}**")
        st.write(f"FPA rows: **{fpa_rows:,}**")
        st.divider()
        st.caption("Tip: Ask things like:")
        st.markdown("- complaints per 1000 by process for portfolio London Jun 2025 to Aug 2025")
        st.markdown('- show rca1 by portfolio for process "Member Enquiry" last 3 months')
        st.markdown("- unique cases by process and portfolio Apr 2025 to Jun 2025")
        st.markdown("- show the biggest drivers of case fails")

    portfolios = sorted(set(
        pd.concat([store.get("cases", pd.DataFrame()), store.get("complaints", pd.DataFrame())])
        .get("Portfolio_std", pd.Series(dtype=str))
        .dropna()
        .unique()
        .tolist()
    ))

    # derive "processes" from both sources
    processes = set()
    if "cases" in store and "ProcessName" in store["cases"].columns:
        processes.update(store["cases"]["ProcessName"].dropna().unique().tolist())
    if "complaints" in store and "Parent_Case_Type" in store["complaints"].columns:
        processes.update(store["complaints"]["Parent_Case_Type"].dropna().unique().tolist())
    processes = sorted(processes)

    q = st.chat_input("Type your question (e.g., 'complaints per 1000 by process last 3 months')")
    if not q:
        return

    st.chat_message("user").write(q)
    with st.chat_message("assistant"):
        pr = parse(q, portfolios=portfolios, processes=processes)

        # dispatch
        if pr.intent == "complaints_per_1000":
            _h_complaints_per_1000(store, pr)
        elif pr.intent == "rca1_by_portfolio":
            _h_rca1_by_portfolio(store, pr)
        elif pr.intent == "unique_cases":
            _h_unique_cases(store, pr)
        elif pr.intent == "fpa_fail_drivers":
            _h_fpa_fail_drivers(store, pr)
        else:
            st.info("I couldn't recognise that intent. Try: 'complaints per 1000', 'rca1 by portfolio', 'unique cases', or 'drivers of case fails'.")
