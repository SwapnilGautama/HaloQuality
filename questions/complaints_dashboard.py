import re
import pandas as pd
import streamlit as st

# -----------------------------
# Helpers (mirror the existing keys logic in the store)
# -----------------------------
def _month_key(dt_series: pd.Series) -> pd.Series:
    dt = pd.to_datetime(dt_series, errors="coerce")
    return dt.dt.to_period("M").dt.to_timestamp()

def _month_label(key_series: pd.Series) -> pd.Series:
    return key_series.dt.strftime("%b %y")

def _norm_text(x: str) -> str:
    if pd.isna(x):
        return None
    s = str(x).strip().lower()
    s = s.replace("–", "-").replace("—", "-").replace("&", "and")
    s = re.sub(r"^(member|employer|employee)\s*-\s*", "", s)
    s = re.sub(r"[^a-z0-9]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s or None

def _pick_text_col(df: pd.DataFrame, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    # soft contains
    for c in df.columns:
        lc = str(c).lower()
        if any(tok in lc for tok in ["comment","description","details","notes","summary"]):
            return c
    return None

def _pick_rca_col(df: pd.DataFrame):
    strict = {"rca1","rca 1","root cause 1","root cause","primary cause","primary category"}
    for c in df.columns:
        if str(c).strip().lower() in strict:
            return c
    for c in df.columns:
        lc = str(c).lower()
        if "rca" in lc or ("root" in lc and "cause" in lc) or ("primary" in lc and "category" in lc):
            return c
    return None

# Editable reason patterns (only used if no RCA1 is available)
_REASON_PATTERNS = [
    ("Delay Manual calculation",  [r"\bmanual calc", r"\bmanual (?:process|work)", r"calc(?:ulation)? delay"]),
    ("Aptia standard Timescale",  [r"\btimescale", r"\bsl[a-]?", r"\bturnaround", r"\btarget time"]),
    ("Delay – 3rd Party",         [r"\bthird[- ]party", r"\b3rd ?party", r"\btrustee\b", r"\binsurer"]),
    ("Delay – Postal",            [r"\bpost(al)? delay", r"\bletter.*delay", r"\bmail delay"]),
    ("Pension set up / AVC",      [r"\bpension set ?up", r"\bavc\b", r"additional voluntary"]),
    ("Requirement not checked",   [r"not checked", r"missing check", r"verification.? missing"]),
    ("Case not created",          [r"case not created", r"\bno case", r"not opened"]),
    ("2nd Review / QA",           [r"second review", r"\b2nd review", r"\bqa\b"]),
    ("Trustee / Scheme Rules",    [r"scheme rules", r"\brule\b", r"\btrustee"]),
    ("Aptia standard template",   [r"standard template", r"standard form", r"template"]),
    ("Drop in value / factor",    [r"drop in value", r"factor change", r"revaluation"]),
    ("Overpayment",               [r"overpay", r"over payment"]),
    ("Death benefits payout",     [r"death benefit", r"bereave"]),
    ("Pension increase",          [r"pension increase", r"increase applied"]),
    ("Transfer documentation",    [r"transfer doc", r"transfer.*document"]),
]

def _compile_reason_map():
    return [(name, [re.compile(pat, re.I) for pat in pats]) for name, pats in _REASON_PATTERNS]

def _label_reason(text: str, compiled):
    if not isinstance(text, str) or not text.strip():
        return "Other"
    for label, regs in compiled:
        if any(r.search(text) for r in regs):
            return label
    return "Other"

# -----------------------------
# Core computations
# -----------------------------
def _per_1000_by_portfolio_month(cases, complaints, start_key=None, end_key=None):
    wk_cases = cases.copy()
    wk_compl = complaints.copy()
    if start_key is not None:
        wk_cases = wk_cases[wk_cases["month_key"] >= start_key]
        wk_compl = wk_compl[wk_compl["month_key"] >= start_key]
    if end_key is not None:
        wk_cases = wk_cases[wk_cases["month_key"] <= end_key]
        wk_compl = wk_compl[wk_compl["month_key"] <= end_key]

    case_g = (wk_cases
        .groupby(["port_key","month_key"], dropna=False).size()
        .rename("cases").reset_index())
    comp_g = (wk_compl
        .groupby(["port_key","month_key"], dropna=False).size()
        .rename("complaints").reset_index())

    df = case_g.merge(comp_g, on=["port_key","month_key"], how="left")
    df["complaints"] = df["complaints"].fillna(0).astype(int)
    df["per_1000"] = (df["complaints"] / df["cases"]).fillna(0) * 1000
    df["month"] = _month_label(df["month_key"])
    return df

def _reasons_for_month(complaints, target_month_key):
    # prefer RCA1 if present, else derive from comments
    rca_col = _pick_rca_col(complaints)
    month_df = complaints[complaints["month_key"] == target_month_key].copy()
    if month_df.empty:
        return pd.DataFrame(columns=["Reason","count","pct"])

    if rca_col:
        month_df["Reason"] = month_df[rca_col].fillna("Other").astype(str).str.strip()
    else:
        text_col = _pick_text_col(month_df, ["Comments","Comment","Description","Details"])
        compiled = _compile_reason_map()
        month_df["Reason"] = month_df[text_col].apply(lambda t: _label_reason(t, compiled)) if text_col else "Other"

    counts = (month_df.groupby("Reason", dropna=False).size()
              .rename("count").reset_index().sort_values("count", ascending=False))
    total = counts["count"].sum() or 1
    counts["pct"] = (counts["count"] / total * 100).round(1)
    counts["cum_pct"] = counts["pct"].cumsum()
    counts["in_80"] = counts["cum_pct"] <= 80.0
    return counts

# -----------------------------
# Public entry point
# -----------------------------
def run(store, params, user_text=None):
    st.subheader("Complaints analysis (dashboard)")

    cases = store["cases"]
    complaints = store["complaints"]

    start_key = params.get("start_month_key")
    end_key   = params.get("end_month_key")
    # if the user asked a single month (e.g., “Jun 2025”), use end_key as “selected month”
    selected_key = params.get("selected_month_key") or end_key

    # 1) rate per 1,000 by portfolio × month
    by_port_m = _per_1000_by_portfolio_month(cases, complaints, start_key, end_key)
    if by_port_m.empty:
        st.info("No overlapping data for the selected window.")
        return

    piv = (by_port_m
           .pivot_table(index="port_key", columns="month", values="per_1000", aggfunc="sum")
           .fillna(0)
           .sort_index())
    st.markdown("**Complaints per 1,000 cases — by portfolio**")
    st.dataframe(piv.round(1))

    # 2) Reasons for the selected month
    if selected_key is not None:
        rtbl = _reasons_for_month(complaints, selected_key)
        st.markdown("---")
        st.markdown(f"**Reasons — {_month_label(pd.Series([selected_key])).iloc[0]}**")
        if rtbl.empty:
            st.info("No complaints in the selected month.")
        else:
            st.dataframe(
                rtbl[["Reason","count","pct","in_80"]]
                .rename(columns={"pct": "% contribution", "in_80":"in 80% band"}),
                hide_index=True
            )
