# questions/nps_by_portfolio.py
from __future__ import annotations

from typing import Dict, Any, Optional, Iterable, List, Tuple
from pathlib import Path
import re
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# ----------------- helpers -----------------
def _find_col(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    if df is None or df.empty:
        return None
    m = {str(c).strip().lower(): c for c in df.columns}
    for c in candidates:
        lc = c.strip().lower()
        if lc in m:
            return m[lc]
    # loose fallback: partial match
    for c in df.columns:
        cl = str(c).strip().lower()
        for n in candidates:
            if n.strip().lower() in cl:
                return c
    return None

def _soft_pastels(n: int) -> list:
    base = ["#A3C4F3","#CDE7BE","#F6C1C1","#FFD6A5","#BDB2FF","#FFAFCC","#BEE1E6","#E2ECE9"]
    return [base[i % len(base)] for i in range(n)]

def _norm_portfolio(x: str) -> str:
    if not isinstance(x, str): return "Unknown"
    t = x.strip().title()
    t = t.replace("Baes-Leatherhead", "Leatherhead - Baes").replace("Leatherhead  - Baes", "Leatherhead - Baes")
    return t

# simple offline lexical sentiment
_POS = {"good","great","excellent","amazing","helpful","fast","quick","responsive","easy","clear",
        "friendly","polite","supportive","smooth","love","efficient","prompt","awesome","happy"}
_NEG = {"bad","poor","terrible","slow","delay","delayed","waiting","confusing","unclear","hard",
        "rude","unhelpful","expensive","issue","problem","bug","error","crash","difficult","worst"}

def _lex_sentiment(text: str) -> float:
    if not isinstance(text, str) or not text.strip():
        return 0.0
    toks = re.findall(r"\b\w+\b", text.lower())
    if not toks:
        return 0.0
    score = sum(1 for t in toks if t in _POS) - sum(1 for t in toks if t in _NEG)
    return max(-1.0, min(1.0, score / max(len(toks), 4)))

# ----------------- AI key-theme extraction + effect-size helpers -----------------
_STOP = {
    "the","a","an","and","or","for","to","of","in","on","at","by","with","from","as","is","are",
    "was","were","be","been","it","this","that","these","those","we","you","they","i","he","she",
    "them","our","your","their","us","but","so","if","than","then","too","very","can","could",
    "would","should","may","might","will","just","also","not","no","yes","all","any","each","every",
    "more","most","some","such","into","within","about","over","under","per","etc","na","none","null"
}

def _clean_tokens(text: str):
    toks = re.findall(r"\b[a-zA-Z][a-zA-Z\-']+\b", str(text).lower())
    out = []
    for t in toks:
        t = t.strip("-'")
        if len(t) < 3:
            continue
        if t in _STOP:
            continue
        out.append(t)
    return out

def _top_phrases(texts, k: int = 6):
    from collections import Counter
    uni = Counter(); bi = Counter()
    for t in texts:
        toks = _clean_tokens(t)
        uni.update(toks)
        bi.update([" ".join(pair) for pair in zip(toks, toks[1:]) if pair[0] != pair[1]])
    top_bi = [w for w,_ in bi.most_common(k)]
    if len(top_bi) < k:
        need = k - len(top_bi)
        top_uni = [w for w,_ in uni.most_common(need)]
        return [*top_bi, *top_uni]
    return top_bi[:k]

def _pearson(df: pd.DataFrame, x: str, y: str) -> float | np.nan:
    if x not in df or y not in df: return np.nan
    d = df[[x,y]].dropna()
    if len(d) < 3: return np.nan
    r = d[x].corr(d[y])
    return float(r) if pd.notna(r) else np.nan

def _slope(df: pd.DataFrame, x: str, y: str) -> float | np.nan:
    if x not in df or y not in df: return np.nan
    d = df[[x,y]].dropna()
    if len(d) < 3: return np.nan
    try:
        b1, _ = np.polyfit(d[x].astype(float), d[y].astype(float), 1)  # y = b1*x + b0
        return float(b1)
    except Exception:
        return np.nan


# --- add: robust month normaliser & top-k table for snapshots ---
def _normalise_month_column(df: pd.DataFrame) -> tuple[pd.DataFrame, Optional[str]]:
    if df is None or len(df) == 0:
        return df, None
    if isinstance(df.index, pd.MultiIndex):
        if '_month' in df.index.names:
            df = df.reset_index('_month')
        elif 'month' in df.index.names:
            df = df.reset_index('month')
        else:
            df = df.reset_index()
    else:
        if df.index.name in ('_month', 'month'):
            df = df.reset_index()

    candidate_cols = [c for c in df.columns if c in ('_month','month','Month')]
    if not candidate_cols:
        maybe_time = [c for c in df.columns
                      if pd.api.types.is_period_dtype(df[c]) or pd.api.types.is_datetime64_any_dtype(df[c])]
        if maybe_time:
            month_col = maybe_time[0]
        else:
            return df, None
    else:
        month_col = candidate_cols[0]

    if list(df.columns).count(month_col) > 1:
        cols, seen = [], 0
        for c in df.columns:
            if c == month_col:
                seen += 1
                cols.append(c if seen == 1 else f"{c}_col")
            else:
                cols.append(c)
        df.columns = cols

    ser = df[month_col]
    if pd.api.types.is_period_dtype(ser):
        df['_month_str'] = ser.astype(str)
        month_col = '_month_str'
    elif pd.api.types.is_datetime64_any_dtype(ser):
        df['_month_str'] = pd.to_datetime(ser).dt.to_period('M').astype(str)
        month_col = '_month_str'
    else:
        df[month_col] = df[month_col].astype(str)
    return df, month_col

def _table_latest(df: pd.DataFrame, month_col: str, value_col: str, group_col: str, k: int = 8) -> pd.DataFrame:
    if df is None or df.empty or month_col is None:
        return pd.DataFrame()
    latest = df[month_col].max()
    tmp = df[df[month_col] == latest].copy()
    if value_col not in tmp.columns or group_col not in tmp.columns:
        return pd.DataFrame()
    out = (tmp.groupby(group_col, as_index=False)[value_col]
           .mean().sort_values(value_col, ascending=False).head(k))
    return out
# ----------------- palette & styling -----------------
_DARK_BLUE = "#0b3d91"   # titles
_DARK_GREY = "#333333"   # all chart fonts
_SOFT_GREY = "#E0E0E0"   # axes
_BUBBLE_FILL = "#8ECAE6" # soft pastel bubble
_BUBBLE_EDGE = "#5A7AA1" # soft pastel edge

# sentiments bar colors
_SENT_NEG = "#9BBBD4"   # soft pastel
_SENT_NEU = "#F4C27A"
_SENT_POS = "#7BC47F"

def _style_axes(ax: plt.Axes) -> None:
    ax.tick_params(colors=_DARK_GREY, labelcolor=_DARK_GREY)
    ax.xaxis.label.set_color(_DARK_GREY)
    ax.yaxis.label.set_color(_DARK_GREY)
    for sp in ("top","right"):
        ax.spines[sp].set_visible(False)
    for sp in ("bottom","left"):
        ax.spines[sp].set_color(_SOFT_GREY)
        ax.spines[sp].set_linewidth(1.25)
    ax.grid(False)

# --- Sentiments MoM figure ---
def _fig_mom_nps_pos(nps_df: pd.DataFrame, sd_df: pd.DataFrame):
    if nps_df is None: nps_df = pd.DataFrame()
    if sd_df is None:  sd_df  = pd.DataFrame()
    if not nps_df.empty:
        nm = (nps_df.groupby("_month")[["promoter","passive","detractor","unknown"]]
                    .sum(min_count=1).reset_index())
        nm["Total"] = nm[["promoter","passive","detractor","unknown"]].sum(axis=1)
        nm["NPS"] = ((nm["promoter"] - nm["detractor"]) / nm["Total"].replace(0, np.nan) * 100.0)
        nps_m = nm[["_month","NPS"]]
    else:
        nps_m = pd.DataFrame(columns=["_month","NPS"])
    if not sd_df.empty:
        sm = (sd_df.groupby("_month")
                    .agg(Pos=("sent_label", lambda s: (s=="positive").sum()),
                         Total=("sent_label","size"))
                    .reset_index())
        sm["Pos%"] = sm["Pos"] / sm["Total"].replace(0, np.nan) * 100.0
        pos_m = sm[["_month","Pos%"]]
    else:
        pos_m = pd.DataFrame(columns=["_month","Pos%"])
    if nps_m.empty and pos_m.empty:
        return None
    mm = pd.merge(nps_m, pos_m, on="_month", how="outer").sort_values("_month")
    mm["_label"] = mm["_month"].astype(str).tolist()
    x = np.arange(len(mm))
    fig, ax = plt.subplots(figsize=(8.8, 3.6))
    ax.plot(x, mm["NPS"], linewidth=2.8, label="NPS %")
    ax.plot(x, mm["Pos%"], linewidth=2.8, label="Positive %")
    ax.set_xticks(x); ax.set_xticklabels(mm["_label"].tolist(), rotation=0, color=_DARK_GREY)
    _style_axes(ax); ax.get_yaxis().set_visible(False); ax.set_xlabel("")
    ax.set_title("MoM Trend — NPS % & Positive Sentiment %", color=_DARK_BLUE, pad=10)
    ax.legend(frameon=False, loc="upper right")
    return fig

# ======================
# FPA multi-file loader with caching (replicates FPA treatment)
# ======================
def _pick(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    cols = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in cols:
            return cols[c.lower()]
    return None

def _find_fpa_workbooks() -> List[Path]:
    roots = [Path("data/first_pass_accuracy"), Path("first_pass_accuracy"), Path("data/first_pass_accuracy/")]
    patterns = ["FirstPassAccuracy*.xls*", "*FirstPassAccuracy*.xls*"]
    hits: List[Path] = []
    for root in roots:
        if not root.exists():
            continue
        for pat in patterns:
            hits.extend(root.glob(pat))
    hits = sorted(set(hits), key=lambda p: (p.name, p.stat().st_mtime_ns if p.exists() else 0))
    return hits

def _read_excel_any(path: Path) -> pd.DataFrame:
    try:
        return pd.read_excel(path)
    except Exception:
        return pd.read_excel(path, header=0)

def _workbook_cache_key(p: Path) -> str:
    try:
        return f"{p.resolve()}::{p.stat().st_mtime_ns}"
    except Exception:
        return str(p)

def _workbooks_signature(paths: List[Path]) -> str:
    parts = []
    for p in paths:
        try:
            stt = p.stat()
            parts.append(f"{p.resolve()}::{stt.st_size}::{stt.st_mtime_ns}")
        except Exception:
            parts.append(str(p))
    return "|".join(parts)

@st.cache_data(show_spinner=False)
def _read_fpa_cached(path_str: str, path_key: str) -> pd.DataFrame:
    return _read_excel_any(Path(path_str))

def _normalize_one_fpa(df: pd.DataFrame) -> pd.DataFrame:
    col_date = _pick(df, ["Activity Date", "ActivityDate", "Date", "Activity date"])
    col_result = _pick(df, ["Review Result", "Review result", "Result"])
    col_port = _pick(df, ["Portfolio", "portfolio"])
    if col_date is None or col_result is None:
        raise KeyError("Missing required FPA columns")
    out = df.rename(columns={col_date: "date", col_result: "result", col_port: "portfolio" if col_port else "portfolio"}).copy()
    out["portfolio"] = out.get("portfolio", "Unknown")
    out["portfolio"] = out["portfolio"].astype(str).map(_norm_portfolio)
    out["date"] = pd.to_datetime(out["date"], errors="coerce", dayfirst=True)
    out["_month"] = out["date"].dt.to_period("M")
    res = out["result"].astype(str).str.strip().str.lower()
    out["_pass"] = res.str.startswith("pass").astype(int)
    return out[["portfolio","_month","_pass"]]

@st.cache_data(show_spinner=False)
def _combine_normalised_fpa_cached(sig: str, path_strs: List[str], path_keys: List[str]) -> pd.DataFrame:
    frames = []
    for p_str, key in zip(path_strs, path_keys):
        df_src = _read_fpa_cached(p_str, key)
        frames.append(_normalize_one_fpa(df_src))
    if not frames:
        return pd.DataFrame(columns=["portfolio","_month","_pass"])
    return pd.concat(frames, ignore_index=True)

def _load_fpa_from_store_or_disk(store: Dict[str, Any]) -> pd.DataFrame:
    """
    Return normalized FPA month rollups across **ALL workbooks**:
    columns -> Portfolio, _month, FPA%
    Priority: store['fpa'] (already/raw) → cached combine of all workbooks on disk.
    """
    fpa_raw = store.get("fpa", pd.DataFrame()) if isinstance(store, dict) else pd.DataFrame()
    if fpa_raw is not None and not fpa_raw.empty:
        return _fpa_monthly_from_df_cached(fpa_raw)

    paths = _find_fpa_workbooks()
    if not paths:
        return pd.DataFrame(columns=["Portfolio","_month","FPA%"])

    sig = _workbooks_signature(paths)
    path_strs = [str(p) for p in paths]
    path_keys  = [_workbook_cache_key(p) for p in paths]
    allf = _combine_normalised_fpa_cached(sig, path_strs, path_keys)
    if allf.empty:
        return pd.DataFrame(columns=["Portfolio","_month","FPA%"])
    g = (allf.dropna(subset=["_month"])
              .groupby(["portfolio","_month"])["_pass"]
              .agg(passed="sum", total="count")
              .reset_index())
    g["FPA%"] = (g["passed"] * 100.0 / g["total"].replace(0, np.nan))
    g = g.rename(columns={"portfolio":"Portfolio"})
    return g[["Portfolio","_month","FPA%"]]

@st.cache_data(show_spinner=False)
def _fpa_monthly_from_df_cached(fpa_raw: pd.DataFrame) -> pd.DataFrame:
    df = fpa_raw.copy()
    # If already normalized, just return required columns
    if {"Portfolio","_month","FPA%"} <= set(df.columns):
        return df[["Portfolio","_month","FPA%"]]
    # Else try to normalize a raw file similar to the disk-normaliser
    if "date" in df.columns and "result" in df.columns:
        out = df.copy()
        out["portfolio"] = out.get("portfolio", out.get("Portfolio", "Unknown"))
        out["portfolio"] = out["portfolio"].astype(str).map(_norm_portfolio)
        out["date"] = pd.to_datetime(out["date"], errors="coerce", dayfirst=True)
        out["_month"] = out["date"].dt.to_period("M")
        res = out["result"].astype(str).str.strip().str.lower()
        out["_pass"] = res.str.startswith("pass").astype(int)
        g = (out.dropna(subset=["_month"])
                 .groupby(["portfolio","_month"])["_pass"]
                 .agg(passed="sum", total="count")
                 .reset_index())
        g["FPA%"] = (g["passed"] * 100.0 / g["total"].replace(0, np.nan))
        g = g.rename(columns={"portfolio":"Portfolio"})
        return g[["Portfolio","_month","FPA%"]]
    return pd.DataFrame(columns=["Portfolio","_month","FPA%"])

# ======================
# Surveys (NPS + suggestions) — cached transforms
# ======================
@st.cache_data(show_spinner=False)
def _prep_surveys_cached(df_raw: pd.DataFrame) -> pd.DataFrame:
    if df_raw is None or df_raw.empty: return pd.DataFrame()
    df = df_raw.copy()
    pcol = _find_col(df, ["portfolio"])
    df["Portfolio"] = df[pcol].map(_norm_portfolio) if pcol else "Unknown"
    mcol = _find_col(df, ["month_received","month received","month"])
    if mcol:
        parsed = pd.to_datetime(df[mcol], errors="coerce", dayfirst=True, infer_datetime_format=True)
        miss = parsed.isna()
        if miss.any():
            mm = df.loc[miss, mcol].astype(str).str[:3].str.title()
            parsed.loc[miss] = pd.to_datetime(mm + " 1 2025", errors="coerce", format="%b %d %Y")
        df["_month"] = parsed.dt.to_period("M")
    else:
        df["_month"] = pd.NaT
    scol = _find_col(df, ["nps","nps score","nps_score","nps (0-10)","score","rating"])
    score = pd.to_numeric(df[scol], errors="coerce") if scol else pd.Series([np.nan]*len(df))
    bucket = np.where(score >= 9, "promoter", np.where(score >= 7, "passive", np.where(score >= 0, "detractor","unknown")))
    df["nps_bucket"] = bucket
    sugcol = _find_col(df, ["suggestions","suggestion","comments","comment","feedback"])
    if sugcol:
        df["Suggestions"] = df[sugcol].astype(str).str.strip()
        df.loc[df["Suggestions"].str.lower().isin(["","nan","none","null"]), "Suggestions"] = np.nan
    else:
        df["Suggestions"] = np.nan
    return df

@st.cache_data(show_spinner=False)
def _aggregate_nps_cached(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty: return pd.DataFrame()
    df = df[df["_month"].notna()].copy()
    g = df.groupby(["Portfolio","_month"])["nps_bucket"].value_counts().unstack(fill_value=0)
    for c in ["promoter","passive","detractor","unknown"]:
        if c not in g.columns: g[c] = 0
    g["Total"] = g[["promoter","passive","detractor","unknown"]].sum(axis=1).replace(0, np.nan)
    g["NPS%"] = ((g["promoter"] - g["detractor"]) / g["Total"]) * 100.0
    return g.reset_index()

@st.cache_data(show_spinner=False)
def _sentiments_cached(df: pd.DataFrame) -> pd.DataFrame:
    sug = df[df["Suggestions"].notna()].copy()
    if sug.empty: return sug
    score = sug["Suggestions"].map(_lex_sentiment)
    lab = np.where(score >= 0.05, "positive", np.where(score <= -0.05, "negative", "neutral"))
    sug["sent_score"] = score; sug["sent_label"] = lab
    return sug

# ======================
# Cases / Complaints — cached rollups from provided store DFs
# ======================
@st.cache_data(show_spinner=False)
def _cases_monthly_from_df(cs: pd.DataFrame) -> pd.DataFrame:
    if cs is None or cs.empty:
        return pd.DataFrame(columns=["Portfolio","_month","Total Cases Complete"])
    df = cs.copy()
    p = _find_col(df, ["portfolio"])
    df["Portfolio"] = df[p].map(_norm_portfolio) if p else "Unknown"
    date_candidates = [
        "completed date","date completed","completion date","completed_date","completed",
        "closed date","date closed","closed_date","closed",
        "finish date","end date","resolved date","resolution date",
        "create date","created date","create_date","start date","report date","date"
    ]
    d = _find_col(df, date_candidates)
    def _parse_dates(series: pd.Series) -> pd.Series:
        s = pd.to_datetime(series, errors="coerce", dayfirst=True)
        if s.notna().sum() == 0:
            s = pd.to_datetime(series.astype(str), errors="coerce", infer_datetime_format=True)
        return s
    parsed = _parse_dates(df[d]) if d else pd.Series(pd.NaT, index=df.index)
    if parsed.notna().sum() == 0:
        for alt in ["create date","created date","report date","date"]:
            col = _find_col(df, [alt])
            if col:
                parsed = _parse_dates(df[col])
                if parsed.notna().sum() > 0:
                    break
    df["_month"] = parsed.dt.to_period("M")
    out = (df.dropna(subset=["_month"])
             .groupby(["Portfolio","_month"])
             .size().to_frame("Total Cases Complete")
             .reset_index())
    return out

@st.cache_data(show_spinner=False)
def _complaints_monthly_from_df(comp: pd.DataFrame) -> pd.DataFrame:
    if comp is None or comp.empty:
        return pd.DataFrame(columns=["Portfolio","_month","Total Complaints"])
    df = comp.copy()
    p = _find_col(df, ["portfolio"]); df["Portfolio"] = df[p].map(_norm_portfolio) if p else "Unknown"
    month_candidates = [
        "month", "date complaint received - dd/mm/yy","date complaint received","complaint date",
        "received date","received_date","date","report date","created date","create date"
    ]
    d = _find_col(df, month_candidates)
    if d:
        parsed = pd.to_datetime(df[d], errors="coerce", dayfirst=True)
        if parsed.isna().all():
            parsed = pd.to_datetime(df[d].astype(str), errors="coerce", infer_datetime_format=True)
        if parsed.isna().all() and d.lower() == "month":
            mm = df[d].astype(str).str[:3].str.title()
            parsed = pd.to_datetime(mm + " 2025", format="%b %Y", errors="coerce")
    else:
        parsed = pd.Series(pd.NaT, index=df.index)
    df["_month"] = parsed.dt.to_period("M")
    out = (df.dropna(subset=["_month"])
             .groupby(["Portfolio","_month"])
             .size().to_frame("Total Complaints")
             .reset_index())
    return out

# ======================
# Combined view cache (depends on inputs + filters)
# ======================
@st.cache_data(show_spinner=False)
def _build_combined_cached(
    base: pd.DataFrame,
    fpa_monthly: pd.DataFrame,
    cases_monthly: pd.DataFrame,
    comp_monthly: pd.DataFrame,
    sent_m: pd.DataFrame,
    sel_port: str,
    start: Optional[pd.Period],
    end: Optional[pd.Period],
) -> pd.DataFrame:
    combined = (base
        .merge(fpa_monthly,   on=["Portfolio","_month"], how="outer")
        .merge(cases_monthly, on=["Portfolio","_month"], how="outer")
        .merge(comp_monthly,  on=["Portfolio","_month"], how="outer")
        .merge(sent_m,        on=["Portfolio","_month"], how="left"))

    # Robust rate: only compute when denominator > 0
    num = pd.to_numeric(combined.get("Total Complaints"), errors="coerce")
    den = pd.to_numeric(combined.get("Total Cases Complete"), errors="coerce")
    combined["Complaints/1000"] = np.where((den > 0) & pd.notna(num),
                                           (num / den) * 1000.0,
                                           np.nan)

    combined["Detractors%"] = (
        pd.to_numeric(combined.get("detractor"), errors="coerce") /
        pd.to_numeric(combined.get("Total"), errors="coerce")
    ) * 100.0

    if sel_port != "(All)":
        combined = combined[combined["Portfolio"] == sel_port]
    if start is not None:
        combined = combined[combined["_month"] >= start]
    if end is not None:
        combined = combined[combined["_month"] <= end]

    view = combined.rename(columns={"_month":"Month"}).copy()
    for c in ["NPS","FPA%","Complaints/1000","Detractors%","Pos%","Neg%","NetSent%"]:
        if c in view.columns:
            view[c] = pd.to_numeric(view[c], errors="coerce").round(1)
    return view

# ----------------- UI entry -----------------
def run(store: Dict[str, Any], params: Dict[str, Any], user_text: Optional[str] = None):
    """Always returns ((title, subtitle), dataframe) and never raises to the host."""
    df_out = pd.DataFrame()

    try:
        # Load surveys (cached)
        surveys = store.get("surveys", pd.DataFrame())
        s = _prep_surveys_cached(surveys)
        if s.empty or s["_month"].isna().all():
            msg = pd.DataFrame([{"Message": "No usable surveys (check Month_received, NPS, Suggestions)."}])
            return ("NPS by Portfolio", "Surveys (Sheet 1)"), msg

        # Sidebar filters
        with st.sidebar:
            st.header("Filters")
            ports = ["(All)"] + sorted(s["Portfolio"].dropna().unique().tolist())
            sel_port = st.selectbox("Portfolio", ports, index=0)
            months = s["_month"].dropna().sort_values().unique().tolist()
            start = st.selectbox("From month", months, index=0) if months else None
            end = st.selectbox("To month", months, index=len(months)-1) if months else None

        # NPS aggregates (cached)
        nps = _aggregate_nps_cached(s)
        if sel_port != "(All)": nps = nps[nps["Portfolio"] == sel_port]
        if start is not None:  nps = nps[nps["_month"] >= start]
        if end   is not None:  nps = nps[nps["_month"] <= end]

        # overall NPS (weighted)
        if not nps.empty:
            weights = nps[["promoter","passive","detractor","unknown"]].sum(axis=1)
            overall_nps = float((nps["NPS%"] * weights).sum() / weights.sum()) if weights.sum() else np.nan
        else:
            overall_nps = np.nan
        st.markdown(f"### Overall NPS (selected range): **{overall_nps:.1f}**")

        # Pre-compute inputs (all cached)
        base = nps[["Portfolio","_month","NPS%","promoter","passive","detractor","unknown","Total"]]\
                  .rename(columns={"NPS%":"NPS"}).copy()
        fpa_monthly   = _load_fpa_from_store_or_disk(store)           # cached combine like FPA
        cases_monthly = _cases_monthly_from_df(store.get("cases", pd.DataFrame()))
        comp_monthly  = _complaints_monthly_from_df(store.get("complaints", pd.DataFrame()))
        sd_all        = _sentiments_cached(s)

        if not sd_all.empty:
            sent_m = (sd_all.groupby(["Portfolio","_month"])["sent_label"].value_counts()
                              .unstack(fill_value=0)
                              .reset_index())
            for c in ["positive","negative","neutral"]:
                if c not in sent_m.columns: sent_m[c] = 0
            sent_m["S_Tot"] = sent_m[["positive","negative","neutral"]].sum(axis=1)
            sent_m["Pos%"]  = (sent_m["positive"] / sent_m["S_Tot"].replace(0, np.nan) * 100.0)
            sent_m["Neg%"]  = (sent_m["negative"] / sent_m["S_Tot"].replace(0, np.nan) * 100.0)
            sent_m["NetSent%"] = sent_m["Pos%"] - sent_m["Neg%"]
            sent_m = sent_m[["Portfolio","_month","Pos%","Neg%","NetSent%"]]
        else:
            sent_m = pd.DataFrame(columns=["Portfolio","_month","Pos%","Neg%","NetSent%"])

        # Build the combined view (cached by inputs + filters)
        view = _build_combined_cached(base, fpa_monthly, cases_monthly, comp_monthly, sent_m,
                                      sel_port, start, end)

        # Tabs
        tab0, tab1, tab2, tab3 = st.tabs(["Insights", "Overview", "Sentiments", "NPS Correlation"])

        # -------------------- Tab 0: INSIGHTS (heuristic/statistical) --------------------
        with tab0:
            st.markdown(f"<h4 style='color:{_DARK_BLUE};margin:.25rem 0 1rem 0;'>What’s happening and why</h4>",
                        unsafe_allow_html=True)

            nps_m = pd.DataFrame()
            if not nps.empty:
                m = (nps.groupby("_month")[["promoter","passive","detractor","unknown"]].sum(min_count=1))
                m["Total"] = m.sum(axis=1)
                nps_m = pd.DataFrame({
                    "Month": m.index.astype("period[M]"),
                    "NPS":   ((m["promoter"] - m["detractor"]) / m["Total"].replace(0,np.nan) * 100.0)
                }).dropna()

            latest_month = str(view["Month"].dropna().max()) if "Month" in view.columns and not view.empty else None
            latest_slice = view[view["Month"] == view["Month"].dropna().max()] if latest_month else pd.DataFrame()

            delta_nps = np.nan
            if len(nps_m) >= 2:
                delta_nps = float(nps_m["NPS"].iloc[-1] - nps_m["NPS"].iloc[-2])

            best_txt = worst_txt = "n/a"
            if not latest_slice.empty and "NPS" in latest_slice:
                ls = latest_slice.dropna(subset=["NPS"])
                if not ls.empty:
                    best = ls.loc[ls["NPS"].idxmax()]
                    worst = ls.loc[ls["NPS"].idxmin()]
                    best_txt  = f"{best['Portfolio']} ({best['NPS']:.1f})"
                    worst_txt = f"{worst['Portfolio']} ({worst['NPS']:.1f})"

            r_fpa  = _pearson(view, "FPA%", "NPS")
            r_det  = _pearson(view, "Detractors%", "NPS")
            r_neg  = _pearson(view, "Neg%", "NPS")
            r_comp = _pearson(view, "Complaints/1000", "NPS")

            b_fpa  = _slope(view, "FPA%", "NPS")
            b_det  = _slope(view, "Detractors%", "NPS")
            b_neg  = _slope(view, "Neg%", "NPS")
            b_comp = _slope(view, "Complaints/1000", "NPS")

            def _r_words(r: float | np.nan) -> str:
                if pd.isna(r): return "n/a"
                a = abs(r); band = "weak"
                if a >= 0.7: band = "strong"
                elif a >= 0.4: band = "moderate"
                sign = "positive" if r >= 0 else "negative"
                return f"{band} {sign} (r={r:+.2f})"

            def _implication(label: str, slope: float | np.nan, unit: float = 10.0) -> str:
                if pd.isna(slope):
                    return f"- {label}: insufficient data to infer an effect size."
                change = slope * unit
                unit_txt = "pp" if "Complaints" not in label else "units"
                return f"- **{label}**: each **+{unit:g} {unit_txt}** is associated with **{change:+.1f} pp** in NPS."

            neg_story_overall: list[str] = []
            neg_story_latest: list[str] = []
            sd_all = _sentiments_cached(s)
            if not sd_all.empty:
                sd_f = sd_all.copy()
                if sel_port != "(All)": sd_f = sd_f[sd_f["Portfolio"] == sel_port]
                if start is not None:   sd_f = sd_f[sd_f["_month"] >= start]
                if end is not None:     sd_f = sd_f[sd_f["_month"] <= end]
                neg = sd_f[sd_f["sent_label"] == "negative"].dropna(subset=["Suggestions"])
                if not neg.empty:
                    neg_story_overall = _top_phrases(neg["Suggestions"].tolist(), k=6)
                    if latest_month:
                        neg_latest = neg[neg["_month"] == neg["_month"].max()]
                        if not neg_latest.empty:
                            neg_story_latest = _top_phrases(neg_latest["Suggestions"].tolist(), k=4)

            colA, colB = st.columns((1.1, 1))
            with colA:
                st.markdown("#### At a glance")
                st.markdown(
                    f"""
- **Overall NPS** in the selected range: **{overall_nps:.1f}**.
- **Latest month:** **{latest_month or 'n/a'}** • Δ vs previous: **{(delta_nps if not np.isnan(delta_nps) else np.nan):+.1f}** pp.
- **Top portfolio (latest):** {best_txt} • **Bottom:** {worst_txt}.
                    """.strip()
                )
                st.markdown("#### Drivers (correlations across the selection)")
                st.markdown(
                    f"""
- **Detractors% ↔ NPS:** {_r_words(r_det)}  
- **Negative suggestions% ↔ NPS:** {_r_words(r_neg)}  
- **FPA% ↔ NPS:** {_r_words(r_fpa)}  
- **Complaints/1000 ↔ NPS:** {_r_words(r_comp)}
                    """.strip()
                )
                st.markdown("#### What the numbers imply")
                st.markdown("\n".join([
                    _implication("Detractors%", b_det, 10.0),
                    _implication("Negative suggestions%", b_neg, 10.0),
                    _implication("FPA%", b_fpa, 10.0),
                    _implication("Complaints/1000", b_comp, 10.0),
                ]))

            with colB:
                st.markdown("#### Latest month snapshot")
                if not latest_slice.empty:
                    show_cols = [c for c in ["Portfolio","Month","NPS","Detractors%","Neg%","NetSent%","FPA%","Complaints/1000"]
                                 if c in latest_slice.columns]
                    st.dataframe(latest_slice[show_cols].sort_values("NPS", ascending=False),
                                 use_container_width=True)
                else:
                    st.info("No month-level records to show for the current selection.")

            if neg_story_overall:
                chips = " • ".join([f"`{t}`" for t in neg_story_overall[:6]])
                st.markdown(f"**Negative comment themes (overall selection):** {chips}")
            if neg_story_latest:
                chips2 = " • ".join([f"`{t}`" for t in neg_story_latest[:4]])
                st.markdown(f"**Latest month negatives:** {chips2}")

            # -------------------- NEW: Detractor sentiment & issues --------------------
            # Focus only on detractors' suggestions within current selection and summarise their sentiment + key issues.
            det_comment = ""
            try:
                sel = s.copy()
                if sel_port != "(All)": sel = sel[sel["Portfolio"] == sel_port]
                if start is not None:   sel = sel[sel["_month"] >= start]
                if end is not None:     sel = sel[sel["_month"] <= end]

                det = sel[(sel["nps_bucket"] == "detractor") & (sel["Suggestions"].notna())].copy()
                if not det.empty:
                    # Sentiment for detractors
                    det["sent_score"] = det["Suggestions"].map(_lex_sentiment)
                    det["sent_label"] = np.where(det["sent_score"] >= 0.05, "positive",
                                          np.where(det["sent_score"] <= -0.05, "negative", "neutral"))
                    total_det = int(det.shape[0])
                    neg_share = (det["sent_label"].eq("negative").mean() * 100.0) if total_det else 0.0
                    pos_share = (det["sent_label"].eq("positive").mean() * 100.0) if total_det else 0.0

                    # Key issues from detractor suggestions
                    det_overall_phr = _top_phrases(det["Suggestions"].tolist(), k=6)
                    latest_m = det["_month"].max()
                    det_latest_phr = []
                    if pd.notna(latest_m):
                        dlat = det[det["_month"] == latest_m]
                        if not dlat.empty:
                            det_latest_phr = _top_phrases(dlat["Suggestions"].tolist(), k=4)

                    det_chips = " • ".join([f"`{t}`" for t in det_overall_phr]) if det_overall_phr else "—"
                    lat_chips = " • ".join([f"`{t}`" for t in det_latest_phr]) if det_latest_phr else "—"

                    st.markdown("#### Detractor sentiment & key issues")
                    st.markdown(
                        f"- **Detractor comments analysed:** {total_det}  "
                        f"- **Negative share:** {neg_share:.1f}%  • **Positive share:** {pos_share:.1f}%"
                    )
                    st.markdown(f"- **Dominant detractor issues (overall):** {det_chips}")
                    if pd.notna(latest_m):
                        st.markdown(f"- **Latest month detractor issues ({str(latest_m)}):** {lat_chips}")
                else:
                    st.markdown("#### Detractor sentiment & key issues")
                    st.info("No detractor suggestions available in the selected range.")
            except Exception:
                # Keep insights resilient even if text parsing fails
                st.markdown("#### Detractor sentiment & key issues")
                st.info("Could not compute detractor sentiment due to missing or malformed data.")

            how_lines = [
                "**How to read this:**",
                "- When **Detractors%** or **Negative suggestions%** rise, NPS typically falls (see effect sizes above).",
                "- A positive **FPA% → NPS** link means better first-time accuracy shows up as happier customers.",
                "- **Complaints/1000** provides external context; higher complaint density aligns with lower NPS.",
                "- **Detractor sentiment & issues** highlights what detractors specifically say; recurring themes indicate where fixes will most move NPS.",
            ]
            st.markdown("\n".join(how_lines))

        # -------------------- Tab 1: OVERVIEW --------------------
        with tab1:
            left, right = st.columns([1,1])
            with left:
                if not nps.empty:
                    fig, ax = plt.subplots()
                    for i, (p, g) in enumerate(nps.groupby("Portfolio")):
                        g = g.sort_values("_month")
                        ax.plot(g["_month"].astype(str), g["NPS%"],
                                marker="o", linewidth=2, markersize=4,
                                label=p, color=_soft_pastels(8)[i % 8])
                    for sp in ["top","right","left"]: ax.spines[sp].set_visible(False)
                    ax.spines["bottom"].set_color("#D3D3D3")
                    ax.get_yaxis().set_visible(False); ax.grid(False); ax.set_xlabel("")
                    ax.set_title("NPS Trend", fontsize=12); ax.legend(loc="best", fontsize=8, frameon=False)
                    st.pyplot(fig, use_container_width=True)
            with right:
                if not nps.empty:
                    detail_df = nps[["Portfolio","_month","NPS%","promoter","passive","detractor","unknown"]].rename(
                        columns={"_month":"Month","NPS%":"NPS"}).copy()
                    detail_df["NPS"] = detail_df["NPS"].round(1)
                st.markdown("#### Detail (by Portfolio × Month)")
                st.dataframe(detail_df, use_container_width=True)

        # -------------------- Tab 2: SENTIMENTS --------------------
        with tab2:
            sd = _sentiments_cached(s)
            if sel_port != "(All)": sd = sd[sd["Portfolio"] == sel_port]
            if start is not None:  sd = sd[sd["_month"] >= start]
            if end   is not None:  sd = sd[sd["_month"] <= end]

            if sd.empty:
                st.info("No suggestions available in the selected range.")
            else:
                s_left, s_right = st.columns([1,1])
                with s_left:
                    pv = sd.pivot_table(index="Portfolio", columns="sent_label", values="Suggestions",
                                        aggfunc="count", fill_value=0)
                    order = [c for c in ["negative","neutral","positive"] if c in pv.columns]
                    pv = pv[order]
                    fig2, ax2 = plt.subplots()
                    x = np.arange(len(pv.index)); bottom = np.zeros(len(x))
                    color_map = {"negative": _SENT_NEG, "neutral": _SENT_NEU, "positive": _SENT_POS}
                    for col in pv.columns:
                        ax2.bar(x, pv[col].values, bottom=bottom, label=col.capitalize(), color=color_map.get(col, _SENT_NEU))
                        bottom += pv[col].values
                    ax2.set_xticks(x); 
                    ax2.set_xticklabels(pv.index, rotation=90, color=_DARK_GREY)
                    for sp in ["top","right","left"]: ax2.spines[sp].set_visible(False)
                    ax2.spines["bottom"].set_color("#D3D3D3")
                    ax2.get_yaxis().set_visible(False); ax2.grid(False)
                    ax2.set_title("Suggestions Sentiment by Portfolio", fontsize=12, color=_DARK_BLUE)
                    leg = ax2.legend(loc="best", fontsize=8, frameon=False)
                    for t in leg.get_texts(): t.set_color(_DARK_GREY)
                    st.pyplot(fig2, use_container_width=True)
                with s_right:
                    cat = (sd.groupby("Portfolio")["sent_label"].value_counts()
                             .unstack(fill_value=0)
                             .reindex(columns=[c for c in ["positive","neutral","negative"]
                                               if c in sd["sent_label"].unique()], fill_value=0))
                    cat["Sugg"]     = cat.sum(axis=1)
                    cat["Pos%"]     = (cat.get("positive",0)/cat["Sugg"]*100).round(1)
                    cat["Neg%"]     = (cat.get("negative",0)/cat["Sugg"]*100).round(1)
                    cat["NetSent%"] = (cat["Pos%"] - cat["Neg%"]).round(1)

                    if not nps.empty:
                        npstab = (nps.groupby("Portfolio")[["promoter","detractor","passive","unknown"]]
                                    .sum(min_count=1))
                        npstab["Total"] = npstab[["promoter","detractor","passive","unknown"]].sum(axis=1)
                        npstab["NPS"] = ((npstab["promoter"] - npstab["detractor"]) /
                                         npstab["Total"].replace(0, np.nan) * 100).round(1)
                        npstab = npstab.rename(columns={"promoter":"Promoters","detractor":"Detractors"})
                        cat = cat.join(npstab[["NPS","Promoters","Detractors"]], how="left")

                    ordered_cols = [c for c in ["Sugg","Pos%","Neg%","NetSent%","NPS","Promoters","Detractors"]
                                    if c in cat.columns]
                    st.markdown("#### Sentiment Summary (filtered range)")
                    st.dataframe(cat[ordered_cols], use_container_width=True)

            fig_mom = _fig_mom_nps_pos(nps, sd if 'sd' in locals() else pd.DataFrame())
            if fig_mom is not None:
                st.pyplot(fig_mom, use_container_width=True)

        # -------------------- Tab 3: NPS CORRELATION (with local Month filter) --------------------
        with tab3:
            st.markdown("### Combined KPIs — NPS, FPA%, Complaints/1000")

            month_opts = ["(All)"] + sorted(view["Month"].dropna().astype(str).unique().tolist())
            sel_corr_month = st.selectbox("Month (local to this tab)", options=month_opts,
                                          index=len(month_opts)-1 if len(month_opts)>1 else 0, key="nps_corr_month")
            if sel_corr_month != "(All)":
                corr_df = view[view["Month"].astype(str) == sel_corr_month].copy()
            else:
                corr_df = view.copy()

            left, right = st.columns([1,1])

            with left:
                plot_df = corr_df.dropna(subset=["NPS","Complaints/1000"]).copy()
                if not plot_df.empty:
                    size = (plot_df.get("FPA%", pd.Series(index=plot_df.index, dtype=float)).fillna(0.0)
                            .clip(lower=0, upper=100))
                    s_px = (size / 100.0) * 1800.0 + 80.0

                    fig_sc, ax_sc = plt.subplots(figsize=(6.6, 4.4))
                    ax_sc.scatter(
                        plot_df["NPS"], plot_df["Complaints/1000"],
                        s=s_px, color=_BUBBLE_FILL, edgecolor=_BUBBLE_EDGE, alpha=0.85
                    )
                    for _, r in plot_df.iterrows():
                        lab = f"{r.get('Portfolio','')}"
                        try:
                            ax_sc.text(float(r["NPS"]), float(r["Complaints/1000"]),
                                       lab, fontsize=8, color=_DARK_GREY, ha="center", va="bottom")
                        except Exception:
                            pass
                    ax_sc.set_xlabel("NPS %", color=_DARK_GREY)
                    ax_sc.set_ylabel("Complaints per 1000", color=_DARK_GREY)
                    _style_axes(ax_sc)
                    ttl_month = sel_corr_month if sel_corr_month != "(All)" else ""
                    ax_sc.set_title(f"NPS vs Complaints/1000 {f'({ttl_month})' if ttl_month else ''}", color=_DARK_BLUE, pad=6)
                    st.pyplot(fig_sc, use_container_width=True)
                    st.caption("Bubble size ∝ FPA% (larger = higher FPA).")
                else:
                    st.info("No rows available for the selected month.")

            with right:
                cols = [c for c in ["Portfolio","Month","Complaints/1000","FPA%","NPS","Detractors%","Pos%","Neg%","NetSent%"]
                        if c in corr_df.columns]
                st.dataframe(corr_df[cols].sort_values(["Portfolio","Month"]), use_container_width=True)

            st.markdown("#### Correlation snapshot (selected range)")
            corr_cols = st.columns(5)

            def _corr_pair(df: pd.DataFrame, x: str, y: str) -> str:
                if x not in df or y not in df: return "n/a"
                d = df[[x,y]].dropna()
                if len(d) < 3: return "n/a"
                r = d[x].corr(d[y])
                if pd.isna(r): return "n/a"
                return f"r = {r:+.2f}"

            with corr_cols[0]:
                st.metric("FPA% vs Complaints/1000", _corr_pair(corr_df, "FPA%", "Complaints/1000"))
            with corr_cols[1]:
                st.metric("FPA% vs NPS", _corr_pair(corr_df, "FPA%", "NPS"))
            with corr_cols[2]:
                st.metric("Detractors% vs NPS", _corr_pair(corr_df, "Detractors%", "NPS"))
            with corr_cols[3]:
                st.metric("NetSent% vs NPS", _corr_pair(corr_df, "NetSent%","NPS"))
            with corr_cols[4]:
                st.metric("NPS vs Complaints/1000", _corr_pair(corr_df, "NPS","Complaints/1000"))

            st.markdown("#### MoM Δ (latest vs previous)")
            if "Month" in view.columns and not view.empty:
                def _delta_last_two(g: pd.DataFrame, col: str):
                    g = g.sort_values("Month")
                    v = pd.to_numeric(g[col], errors="coerce").dropna()
                    if len(v) >= 2: return v.iloc[-1] - v.iloc[-2]
                    return np.nan
                momo = (view.groupby("Portfolio")
                            .apply(lambda g: pd.Series({
                                "ΔNPS": _delta_last_two(g, "NPS"),
                                "ΔFPA%": _delta_last_two(g, "FPA%"),
                                "ΔComplaints/1000": _delta_last_two(g, "Complaints/1000"),
                                "ΔNetSent%": _delta_last_two(g, "NetSent%"),
                            }))
                            .reset_index())
                for c in ["ΔNPS","ΔFPA%","ΔComplaints/1000","ΔNetSent%"]:
                    momo[c] = momo[c].round(1)
                st.dataframe(momo.sort_values("Portfolio"), use_container_width=True)
            else:
                st.caption("Not enough month data for MoM deltas in the current selection.")

            st.caption("Surveys (Sheet 1) with Sentiments and SLA/Complaints correlation")
            df_out = view

    except Exception as e:
        st.error(f"Unexpected error in NPS module: {e}")
        df_out = pd.DataFrame([{"error": str(e)}])

    return ("NPS by Portfolio", "Surveys (Sheet 1) with Sentiments and SLA/Complaints correlation"), df_out

# ---------------------------
# Missing helper for NPS snapshots (safe fallback)
# ---------------------------
def _fig_mom(series, title: str = "MoM"):
    """Simple Month-on-Month line chart for snapshot use.
       Always returns a matplotlib Figure (never raises)."""
    import matplotlib.pyplot as plt
    try:
        if series is None or len(series) == 0:
            raise ValueError("Empty series")

        idx = list(range(len(series)))
        labels = [str(i) for i in series.index]

        fig, ax = plt.subplots(figsize=(8.0, 3.0))
        ax.plot(idx, series.astype(float).values, linewidth=2.5, marker="o", color="#0b3d91")
        ax.set_xticks(idx)
        ax.set_xticklabels(labels, rotation=0, color=_DARK_GREY)
        _style_axes(ax)
        ax.get_yaxis().set_visible(False)
        ax.set_xlabel("")
        ax.set_title(title, color=_DARK_BLUE, pad=6)
        return fig
    except Exception:
        # Always return a placeholder chart so snapshot never breaks
        fig, ax = plt.subplots(figsize=(8.0, 3.0))
        ax.text(0.5, 0.5, "No MoM data available", ha="center", va="center", fontsize=10)
        ax.axis("off")
        return fig

# ---------------------------
# ... keep all your existing imports, caching, tabs, and run() exactly as-is ...

# ---------------------------
# Snapshot builder for NPS (used by app.py email/snapshot)
# ---------------------------

# ============================== NPS SNAPSHOT BUILDERS ================================
def build_snapshot(store, params):
    """Return NPS one‑pager payload dict (title, subtitle, charts, tables, notes)."""
    notes = []
    charts = []
    tables = []

    # raw surveys
    surveys = None
    for key in ("surveys","nps_surveys","df_surveys","df_nps"):
        surveys = store.get(key)
        if isinstance(surveys, pd.DataFrame) and not surveys.empty:
            break

    if not isinstance(surveys, pd.DataFrame) or surveys.empty:
        return {
            "title": "Halo Quality — NPS Snapshot",
            "subtitle": "Auto summary",
            "charts": charts,
            "tables": [("note", pd.DataFrame({"note":["No NPS survey data available."]}))],
            "notes": ["No NPS survey data available in store."]
        }

    df = surveys.copy()

    # portfolio
    pcol = _find_col(df, ["portfolio"])
    df["Portfolio"] = df[pcol].map(_norm_portfolio) if pcol else "Unknown"

    # month
    mcol = _find_col(df, ["month_received","month received","_month","month","date"])
    if mcol:
        m = pd.to_datetime(df[mcol], errors="coerce", dayfirst=True, infer_datetime_format=True)
        if m.isna().all():
            m = pd.to_datetime(df[mcol].astype(str), errors="coerce", infer_datetime_format=True)
        df["_month"] = m.dt.to_period("M")
    else:
        df["_month"] = pd.NaT

    # score/bucket
    scol = _find_col(df, ["nps","nps score","nps_score","nps (0-10)","score","rating"])
    s = pd.to_numeric(df[scol], errors="coerce") if scol else pd.Series([np.nan]*len(df))
    bucket = np.where(s >= 9, "promoter", np.where(s >= 7, "passive", np.where(s >= 0, "detractor","unknown")))
    df["nps_bucket"] = bucket

    # suggestions for themes
    sugcol = _find_col(df, ["suggestions","suggestion","comments","comment","feedback"])
    if sugcol:
        df["Suggestions"] = df[sugcol].astype(str).str.strip()
        df.loc[df["Suggestions"].str.lower().isin(["","nan","none","null"]), "Suggestions"] = np.nan
    else:
        df["Suggestions"] = np.nan

    df = df[df["_month"].notna()].copy()
    if df.empty:
        return {
            "title": "Halo Quality — NPS Snapshot",
            "subtitle": "Auto summary",
            "charts": charts,
            "tables": [("note", pd.DataFrame({"note":["No valid month info in surveys."]}))],
            "notes": ["No valid month info in surveys."]
        }

    g = df.groupby(["Portfolio","_month"])["nps_bucket"].value_counts().unstack(fill_value=0)
    for c in ["promoter","passive","detractor","unknown"]:
        if c not in g.columns: g[c] = 0
    g["Total"] = g[["promoter","passive","detractor","unknown"]].sum(axis=1).replace(0, np.nan)
    g["NPS%"] = ((g["promoter"]-g["detractor"]) / g["Total"]) * 100.0
    base = g.reset_index().rename(columns={"_month":"Month"})

    bm, month_col = _normalise_month_column(base.rename(columns={"Month":"_month"}))
    if month_col is None:
        month_col = "_month"
        notes.append("Month column could not be fully normalised; using fallback.")

    # MoM chart
    overall = (bm.groupby(month_col)[["promoter","passive","detractor","unknown"]].sum(min_count=1))
    if not overall.empty:
        overall["Total"] = overall.sum(axis=1)
        overall["NPS%"] = (overall["promoter"]-overall["detractor"]) / overall["Total"] * 100.0
        fig = _fig_mom(overall["NPS%"], "Overall NPS — Month on Month")
        charts.append(("Overall NPS — Month on Month", fig))
    else:
        notes.append("No MoM series available.")

    # Latest month table
    latest_tbl = _table_latest(bm, month_col, "NPS%", "Portfolio", k=8)
    if not latest_tbl.empty:
        tables.append(("Latest month — Top portfolios by NPS", latest_tbl))
    else:
        notes.append("Could not build latest-month table (NPS)." )

    # detractor key words (simple)
    detractor_tbl = pd.DataFrame()
    if sugcol and df["Suggestions"].notna().any():
        s2 = df[["_month","Suggestions"]].dropna().copy()
        s2, mcol2 = _normalise_month_column(s2.rename(columns={"_month":"Month"}))
        mcol2 = mcol2 or "Month"
        latest = s2[mcol2].max()
        det = s2[s2[mcol2]==latest]["Suggestions"].dropna()
        if not det.empty:
            toks = (det.str.lower().str.replace(r"[^a-z0-9\s]+"," ", regex=True).str.split())
            freq = {}
            for row in toks:
                for t in row:
                    if len(t) < 4: continue
                    freq[t] = freq.get(t,0)+1
            top = sorted(freq.items(), key=lambda x: x[1], reverse=True)[:10]
            if top:
                detractor_tbl = pd.DataFrame(top, columns=["term","count"])
                tables.append((f"Detractor key terms — {str(latest)}", detractor_tbl))
        else:
            notes.append("No suggestions present for latest month.")
    else:
        notes.append("No suggestions column found for themes.")

    return {
        "title": "Halo Quality — NPS Snapshot",
        "subtitle": params.get("period_label", "") or "Auto summary",
        "charts": charts,
        "tables": tables if tables else [("note", pd.DataFrame({"note":["No snapshot content found for NPS."]}))],
        "notes": notes
    }

def get_snapshot_content(state, include_small_tables: bool = True) -> dict:
    try:
        payload = build_snapshot(state, {})
    except Exception as ex:
        return {
            "title": "Halo Quality — NPS Snapshot",
            "charts": [],
            "tables": [("note", pd.DataFrame({"note":[f"Error generating NPS snapshot: {ex}"]}))],
            "notes": [str(ex)]
        }
    if not include_small_tables and payload.get("tables"):
        small = []
        for cap, df in payload["tables"]:
            if len(df) <= 15 and df.shape[1] <= 8:
                small.append((cap, df))
        payload["tables"] = small or payload["tables"]
    return payload
# ============================ /NPS SNAPSHOT BUILDERS =================================

