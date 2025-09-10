# questions/nps_by_portfolio.py
# Import-safe NPS module: exposes `run(store, params, user_text=None) -> ((title, subtitle), df)`
# - Heavy deps imported inside run()
# - No side-effects at import time
# - Robust guards so a missing dataset never crashes the app

__all__ = ["run"]

import re
import numpy as np
import pandas as pd
from pathlib import Path

# ---------- light helpers (safe at import time) ----------
def _find_col(df: pd.DataFrame, candidates):
    if df is None or df.empty:
        return None
    m = {str(c).strip().lower(): c for c in df.columns}
    for c in candidates:
        lc = str(c).strip().lower()
        if lc in m:
            return m[lc]
    # loose partial
    for c in df.columns:
        cl = str(c).strip().lower()
        for n in candidates:
            if n.strip().lower() in cl:
                return c
    return None

def _norm_portfolio(x):
    if not isinstance(x, str): 
        return "Unknown"
    t = x.strip().title()
    t = t.replace("Baes-Leatherhead", "Leatherhead - Baes").replace("Leatherhead  - Baes", "Leatherhead - Baes")
    return t

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

def _top_phrases(texts, k=6):
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

def _pearson(df: pd.DataFrame, x: str, y: str):
    if x not in df or y not in df: return np.nan
    d = df[[x,y]].dropna()
    if len(d) < 3: return np.nan
    r = d[x].corr(d[y])
    return float(r) if pd.notna(r) else np.nan

def _slope(df: pd.DataFrame, x: str, y: str):
    if x not in df or y not in df: return np.nan
    d = df[[x,y]].dropna()
    if len(d) < 3: return np.nan
    try:
        b1, _b0 = np.polyfit(d[x].astype(float), d[y].astype(float), 1)  # y = b1*x + b0
        return float(b1)
    except Exception:
        return np.nan

# ---------- core transforms (no heavy deps) ----------
def _prep_surveys(df_raw: pd.DataFrame) -> pd.DataFrame:
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
    bucket = np.where(score >= 9, "promoter", np.where(score >= 7, "passive",
                    np.where(score >= 0, "detractor","unknown")))
    df["nps_bucket"] = bucket
    sugcol = _find_col(df, ["suggestions","suggestion","comments","comment","feedback"])
    if sugcol:
        df["Suggestions"] = df[sugcol].astype(str).str.strip()
        df.loc[df["Suggestions"].str.lower().isin(["","nan","none","null"]), "Suggestions"] = np.nan
    else:
        df["Suggestions"] = np.nan
    return df

def _aggregate_nps(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty: return pd.DataFrame()
    df = df[df["_month"].notna()].copy()
    g = df.groupby(["Portfolio","_month"])["nps_bucket"].value_counts().unstack(fill_value=0)
    for c in ["promoter","passive","detractor","unknown"]:
        if c not in g.columns: g[c] = 0
    g["Total"] = g[["promoter","passive","detractor","unknown"]].sum(axis=1).replace(0, np.nan)
    g["NPS%"] = ((g["promoter"] - g["detractor"]) / g["Total"]) * 100.0
    return g.reset_index()

def _sentiments(df: pd.DataFrame) -> pd.DataFrame:
    sug = df[df["Suggestions"].notna()].copy()
    if sug.empty: return sug
    score = sug["Suggestions"].map(_lex_sentiment)
    lab = np.where(score >= 0.05, "positive", np.where(score <= -0.05, "negative", "neutral"))
    sug["sent_score"] = score; sug["sent_label"] = lab
    return sug

def _find_fpa_workbook() -> Path | None:
    roots = [Path("data/first_pass_accuracy"), Path("first_pass_accuracy"), Path("data/first_pass_accuracy/")]
    patterns = ["FirstPassAccuracy*.xls*", "*FirstPassAccuracy*.xls*"]
    for root in roots:
        if not root.exists(): continue
        hits = []
        for pat in patterns:
            hits.extend(root.glob(pat))
        if hits:
            return sorted(hits)[-1]
    return None

def _read_excel_any(path: Path) -> pd.DataFrame:
    try:
        return pd.read_excel(path)
    except Exception:
        return pd.read_excel(path, header=0)

def _load_fpa_from_store_or_disk(store) -> pd.DataFrame:
    try:
        fpa_raw = store.get("fpa", pd.DataFrame()) if isinstance(store, dict) else pd.DataFrame()
        if fpa_raw is None or fpa_raw.empty:
            p = _find_fpa_workbook()
            if p:
                try: fpa_raw = _read_excel_any(p)
                except Exception: fpa_raw = pd.DataFrame()
        if fpa_raw is None or fpa_raw.empty:
            return pd.DataFrame(columns=["Portfolio","_month","FPA%"])
        fpa = fpa_raw.copy()
        p_fpa = _find_col(fpa, ["portfolio"])
        fpa["Portfolio"] = fpa[p_fpa].map(_norm_portfolio) if p_fpa else "Unknown"
        d_fpa = _find_col(fpa, ["activity date","activity_date","activitydate","date"]) or "date" if "date" in fpa.columns else None
        fpa["_month"] = pd.to_datetime(fpa[d_fpa], errors="coerce", dayfirst=True).dt.to_period("M") if d_fpa else pd.NaT
        r_fpa = _find_col(fpa, ["review result","review_result","result","qa result","fpa result"])
        res = fpa[r_fpa].astype(str).str.strip().str.lower() if r_fpa else ""
        fpa["_pass"] = res.str.startswith("pass").astype(int) if isinstance(res, pd.Series) else 0
        g = (fpa.dropna(subset=["_month"])
               .groupby(["Portfolio","_month"])["_pass"]
               .agg(passed="sum", total="count").reset_index())
        g["FPA%"] = (g["passed"] * 100.0 / g["total"].replace(0, np.nan))
        return g[["Portfolio","_month","FPA%"]]
    except Exception:
        return pd.DataFrame(columns=["Portfolio","_month","FPA%"])

def _cases_monthly(store) -> pd.DataFrame:
    try:
        cs = store.get("cases", pd.DataFrame()) if isinstance(store, dict) else pd.DataFrame()
        if cs is None or cs.empty:
            return pd.DataFrame(columns=["Portfolio","_month","Total Cases Complete"])
        df = cs.copy()
        p = _find_col(df, ["portfolio"]); df["Portfolio"] = df[p].map(_norm_portfolio) if p else "Unknown"
        d = _find_col(df, ["create date","created date","create_date","start date","report date","date"])
        df["_month"] = pd.to_datetime(df[d], errors="coerce", dayfirst=True).dt.to_period("M") if d else pd.NaT
        id_col = _find_col(df, ["case id","case_id","unique identifier","unique id","id","case reference","case"])
        if id_col:
            out = (df.dropna(subset=["_month"])
                     .groupby(["Portfolio","_month"])[id_col].nunique()
                     .reset_index().rename(columns={id_col: "Total Cases Complete"}))
        else:
            out = (df.dropna(subset=["_month"])
                     .groupby(["Portfolio","_month"]).size()
                     .to_frame("Total Cases Complete").reset_index())
        return out
    except Exception:
        return pd.DataFrame(columns=["Portfolio","_month","Total Cases Complete"])

def _complaints_monthly(store) -> pd.DataFrame:
    try:
        comp = store.get("complaints", pd.DataFrame()) if isinstance(store, dict) else pd.DataFrame()
        if comp is None or comp.empty:
            return pd.DataFrame(columns=["Portfolio","_month","Total Complaints"])
        df = comp.copy()
        p = _find_col(df, ["portfolio"]); df["Portfolio"] = df[p].map(_norm_portfolio) if p else "Unknown"
        d = _find_col(df, ["date complaint received - dd/mm/yy","date complaint received","complaint date",
                           "received date","received_date","date","month"])
        if d and str(d).lower()=="month":
            m = df[d].astype(str).str.strip().str[:3].str.title()
            df["_month"] = pd.to_datetime(m + " 2025", format="%b %Y", errors="coerce").dt.to_period("M")
        else:
            df["_month"] = pd.to_datetime(df[d], errors="coerce", dayfirst=True).dt.to_period("M") if d else pd.NaT
        out = (df.dropna(subset=["_month"])
                 .groupby(["Portfolio","_month"])
                 .size().to_frame("Total Complaints").reset_index())
        return out
    except Exception:
        return pd.DataFrame(columns=["Portfolio","_month","Total Complaints"])

# ---------- main entry ----------
def run(store, params, user_text=None):
    """
    Returns: ((title, subtitle), dataframe)
    Never raises to host; shows Streamlit UI if available.
    """
    # Local imports (avoid import-time crashes)
    try:
        import streamlit as st
        import matplotlib.pyplot as plt
    except Exception:
        # If Streamlit isn't available (e.g., during offline tests), still return data
        st = None
        plt = None

    title = ("NPS by Portfolio", "Surveys (Sheet 1) with Sentiments & SLA/Complaints correlation")
    df_out = pd.DataFrame()

    try:
        surveys = store.get("surveys", pd.DataFrame()) if isinstance(store, dict) else pd.DataFrame()
        s = _prep_surveys(surveys)
        if s.empty or s["_month"].isna().all():
            msg = pd.DataFrame([{"Message": "No usable surveys (check Month_received, NPS, Suggestions)."}])
            return title, msg

        # Sidebar filters (only if Streamlit present)
        if st:
            with st.sidebar:
                st.header("Filters")
                ports = ["(All)"] + sorted(s["Portfolio"].dropna().unique().tolist())
                sel_port = st.selectbox("Portfolio", ports, index=0)
                months = s["_month"].dropna().sort_values().unique().tolist()
                start = st.selectbox("From month", months, index=0) if months else None
                end = st.selectbox("To month", months, index=len(months)-1) if months else None
        else:
            sel_port, start, end = "(All)", None, None

        nps = _aggregate_nps(s)
        if sel_port != "(All)": nps = nps[nps["Portfolio"] == sel_port]
        if start is not None:   nps = nps[nps["_month"] >= start]
        if end is not None:     nps = nps[nps["_month"] <= end]

        # overall NPS (weighted)
        if not nps.empty:
            weights = nps[["promoter","passive","detractor","unknown"]].sum(axis=1)
            overall_nps = float((nps["NPS%"] * weights).sum() / weights.sum()) if weights.sum() else np.nan
        else:
            overall_nps = np.nan
        if st: st.markdown(f"### Overall NPS (selected range): **{overall_nps:.1f}**")

        # correlated data
        base = nps[["Portfolio","_month","NPS%","promoter","passive","detractor","unknown","Total"]]\
                  .rename(columns={"NPS%":"NPS"}).copy()
        fpa_monthly   = _load_fpa_from_store_or_disk(store)
        cases_monthly = _cases_monthly(store)
        comp_monthly  = _complaints_monthly(store)
        sd_all        = _sentiments(s)

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

        combined = base.merge(fpa_monthly,   on=["Portfolio","_month"], how="outer")\
                       .merge(cases_monthly, on=["Portfolio","_month"], how="outer")\
                       .merge(comp_monthly,  on=["Portfolio","_month"], how="outer")\
                       .merge(sent_m,        on=["Portfolio","_month"], how="left")
        combined["Complaints/1000"] = (
            pd.to_numeric(combined.get("Total Complaints"), errors="coerce") /
            pd.to_numeric(combined.get("Total Cases Complete"), errors="coerce")
        ) * 1000.0
        combined["Detractors%"] = (
            pd.to_numeric(combined.get("detractor"), errors="coerce") /
            pd.to_numeric(combined.get("Total"), errors="coerce")
        ) * 100.0

        if sel_port != "(All)":
            combined = combined[combined["Portfolio"] == sel_port]
        if start is not None: combined = combined[combined["_month"] >= start]
        if end   is not None: combined = combined[combined["_month"] <= end]

        view = combined.rename(columns={"_month":"Month"}).copy()
        for c in ["NPS","FPA%","Complaints/1000","Detractors%","Pos%","Neg%","NetSent%"]:
            if c in view.columns:
                view[c] = pd.to_numeric(view[c], errors="coerce").round(1)

        # negative themes (overall + latest)
        neg_story_overall, neg_story_latest = [], []
        latest_month = str(view["Month"].dropna().max()) if "Month" in view.columns and not view.empty else None
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

        # ---------- UI ----------
        if st:
            tab0, tab1, tab2, tab3 = st.tabs(["Insights", "Overview", "Sentiments", "NPS Correlation"])

            # INSIGHTS
            with tab0:
                DARK_BLUE = "#0b3d91"
                st.markdown(f"<h4 style='color:{DARK_BLUE};margin:.25rem 0 1rem 0;'>What’s happening and why</h4>",
                            unsafe_allow_html=True)

                # overall monthly NPS for delta
                nps_m = pd.DataFrame()
                if not nps.empty:
                    m = (nps.groupby("_month")[["promoter","passive","detractor","unknown"]]
                           .sum(min_count=1))
                    m["Total"] = m.sum(axis=1)
                    nps_m = pd.DataFrame({
                        "Month": m.index.astype("period[M]"),
                        "NPS":   ((m["promoter"] - m["detractor"]) / m["Total"].replace(0,np.nan) * 100.0)
                    }).dropna()
                delta_nps = float(nps_m["NPS"].iloc[-1] - nps_m["NPS"].iloc[-2]) if len(nps_m) >= 2 else np.nan

                latest_slice = view[view["Month"] == view["Month"].dropna().max()] if latest_month else pd.DataFrame()
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

                def _r_words(r):
                    if pd.isna(r): return "n/a"
                    a = abs(r); band = "weak"
                    if a >= 0.7: band = "strong"
                    elif a >= 0.4: band = "moderate"
                    sign = "positive" if r >= 0 else "negative"
                    return f"{band} {sign} (r={r:+.2f})"

                def _implication(label, slope, unit=10.0):
                    if pd.isna(slope): 
                        return f"- {label}: insufficient data to infer an effect size."
                    change = slope * unit
                    unit_txt = "pp" if any(k in label for k in ["%","Neg","Detractors"]) else "units"
                    return f"- **{label}**: each **+{unit:g} {unit_txt}** is associated with **{change:+.1f} pp** in NPS."

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
                        st.info("No month-level records for the current selection.")

                if neg_story_overall:
                    chips = " • ".join([f"`{t}`" for t in neg_story_overall[:6]])
                    st.markdown(f"**Negative comment themes (overall selection):** {chips}")
                if neg_story_latest:
                    chips2 = " • ".join([f"`{t}`" for t in neg_story_latest[:4]])
                    st.markdown(f"**Latest month negatives:** {chips2}")

                how_lines = [
                    "**How to read this:**",
                    "- When **Detractors%** or **Negative suggestions%** rise, NPS typically falls (see effect sizes above).",
                    "- A positive **FPA% → NPS** link means better first-time accuracy shows up as happier customers.",
                    "- **Complaints/1000** adds operational context; higher complaint density aligns with lower NPS.",
                ]
                if neg_story_overall:
                    how_lines.append(
                        f"- Dominant **negative themes**: {', '.join(neg_story_overall[:6])}. "
                        "If these spike in the latest month, expect pressure on NPS."
                    )
                st.markdown("\n".join(how_lines))

            # OVERVIEW (kept simple & safe)
            with tab1:
                if not nps.empty:
                    st.markdown("#### Detail (by Portfolio × Month)")
                    detail_df = nps[["Portfolio","_month","NPS%","promoter","passive","detractor","unknown"]]\
                                   .rename(columns={"_month":"Month","NPS%":"NPS"}).copy()
                    detail_df["NPS"] = detail_df["NPS"].round(1)
                    st.dataframe(detail_df, use_container_width=True)

            # SENTIMENTS (condensed)
            with tab2:
                sd = _sentiments(s)
                if sel_port != "(All)": sd = sd[sd["Portfolio"] == sel_port]
                if start is not None:  sd = sd[sd["_month"] >= start]
                if end   is not None:  sd = sd[sd["_month"] <= end]

                if sd.empty:
                    st.info("No suggestions available in the selected range.")
                else:
                    cat = (sd.groupby("Portfolio")["sent_label"].value_counts()
                             .unstack(fill_value=0))
                    cat["Total"] = cat.sum(axis=1)
                    for c in ["positive","neutral","negative"]:
                        if c not in cat.columns: cat[c] = 0
                    cat["Pos%"] = (cat["positive"]/cat["Total"]*100).round(1)
                    cat["Neg%"] = (cat["negative"]/cat["Total"]*100).round(1)
                    st.dataframe(cat[["Total","Pos%","Neg%"]].sort_values("Total", ascending=False),
                                 use_container_width=True)

            # CORRELATION snapshot
            with tab3:
                st.markdown("#### Correlation snapshot (selected range)")
                def _corr_pair(df, x, y):
                    if x not in df or y not in df: return "n/a"
                    d = df[[x,y]].dropna()
                    if len(d) < 3: return "n/a"
                    r = d[x].corr(d[y])
                    return f"r = {r:+.2f}" if pd.notna(r) else "n/a"
                c1, c2, c3, c4, c5 = st.columns(5)
                c1.metric("FPA% vs Complaints/1000", _corr_pair(view, "FPA%", "Complaints/1000"))
                c2.metric("FPA% vs NPS", _corr_pair(view, "FPA%", "NPS"))
                c3.metric("Detractors% vs NPS", _corr_pair(view, "Detractors%", "NPS"))
                c4.metric("NetSent% vs NPS", _corr_pair(view, "NetSent%","NPS"))
                c5.metric("NPS vs Complaints/1000", _corr_pair(view, "NPS","Complaints/1000"))

        df_out = view

    except Exception as e:
        # Show error in UI, but never crash import
        if "streamlit" in globals():
            try:
                import streamlit as st  # local import if earlier failed
                st.error(f"Unexpected error in NPS module: {e}")
            except Exception:
                pass
        df_out = pd.DataFrame([{"error": str(e)}])

    return title, df_out
