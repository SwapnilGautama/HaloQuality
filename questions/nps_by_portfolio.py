# questions/nps_by_portfolio.py
from __future__ import annotations

from typing import Dict, Any, Optional, Iterable
import re
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt


# =============== utils ===============
def _find_col(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    if df is None or df.empty:
        return None
    m = {str(c).strip().lower(): c for c in df.columns}
    for c in candidates:
        lc = c.strip().lower()
        if lc in m:
            return m[lc]
    return None

def _soft_pastels(n: int) -> list:
    base = ["#A3C4F3","#CDE7BE","#F6C1C1","#FFD6A5","#BDB2FF","#FFAFCC","#BEE1E6","#E2ECE9"]
    return [base[i % len(base)] for i in range(n)]

def _norm_portfolio(x: str) -> str:
    if not isinstance(x, str): return "Unknown"
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

# =============== surveys (NPS) prep ===============
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
    df["nps_bucket"] = np.where(score >= 9, "promoter", np.where(score >= 7, "passive", np.where(score >= 0, "detractor", "unknown")))

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

# =============== ops/service (SLA) ===============
def _prep_ops(df_ops: pd.DataFrame) -> pd.DataFrame:
    if df_ops is None or df_ops.empty: return pd.DataFrame()
    d = df_ops.copy()
    pcol = _find_col(d, ["portfolio"])
    d["Portfolio"] = d[pcol].map(_norm_portfolio) if pcol else "Unknown"
    rcol = _find_col(d, ["report_date","report date","date","createddate","create_date"])
    d["_month"] = pd.to_datetime(d[rcol], errors="coerce").dt.to_period("M") if rcol else pd.NaT
    wcol = _find_col(d, ["within sla","sla_status","sla status"])
    d["_within"] = d[wcol].astype(str).str.lower().str.contains("within").astype(int) if wcol else 0
    ccol = _find_col(d, ["completes","completed","checks","volume","total"])
    d["_completes"] = pd.to_numeric(d[ccol], errors="coerce").fillna(0.0) if ccol else 1.0
    return d

def _ops_kpis(d_ops: pd.DataFrame) -> pd.DataFrame:
    if d_ops.empty: return pd.DataFrame()
    g = d_ops.groupby(["Portfolio","_month"]).agg(
        Within=("_within","sum"),
        Total=("Portfolio","size"),
        Completes=("_completes","sum"),
    ).reset_index()
    g["SLA%"] = (g["Within"] / g["Total"]) * 100.0
    return g

# =============== complaints (optional) ===============
def _prep_complaints(df_comp: pd.DataFrame) -> pd.DataFrame:
    if df_comp is None or df_comp.empty: return pd.DataFrame()
    c = df_comp.copy()
    pcol = _find_col(c, ["portfolio"])
    c["Portfolio"] = c[pcol].map(_norm_portfolio) if pcol else "Unknown"
    rm = _find_col(c, ["report month","report_month","month"])
    m1 = pd.to_datetime(c[rm].astype(str).str[:3].str.title() + " 1 2025", errors="coerce", format="%b %d %Y") if rm else pd.NaT
    dcol = _find_col(c, ["date complaint received - dd/mm/yy","date complaint received","received_date","date"])
    m2 = pd.to_datetime(c[dcol], dayfirst=True, errors="coerce") if dcol else pd.NaT
    months = m1 if isinstance(m1, pd.Series) else pd.Series([pd.NaT]*len(c))
    if isinstance(m2, pd.Series):
        months = months.fillna(m2)
    c["_month"] = pd.to_datetime(months, errors="coerce").dt.to_period("M")
    return c

# =============== UI entry ===============
def run(store: Dict[str, Any], params: Dict[str, Any], user_text: Optional[str] = None):
    try:
        surveys = store.get("surveys", pd.DataFrame())
        ops_raw = store.get("ops", store.get("fpa", pd.DataFrame()))
        complaints_raw = store.get("complaints", pd.DataFrame())

        s = _prep_surveys(surveys)
        if s.empty or s["_month"].isna().all():
            msg = pd.DataFrame([{"Message": "No usable surveys (check Month_received, NPS, Suggestions)."}])
            return ("NPS by Portfolio", "Surveys (Sheet 1)"), msg

        with st.sidebar:
            st.header("Filters")
            ports = ["(All)"] + sorted(s["Portfolio"].dropna().unique().tolist())
            sel_port = st.selectbox("Portfolio", ports, index=0)
            months = s["_month"].dropna().sort_values().unique().tolist()
            start = st.selectbox("From month", months, index=0) if months else None
            end = st.selectbox("To month", months, index=len(months)-1) if months else None

        nps = _aggregate_nps(s)
        if sel_port != "(All)": nps = nps[nps["Portfolio"] == sel_port]
        if start is not None:  nps = nps[nps["_month"] >= start]
        if end   is not None:  nps = nps[nps["_month"] <= end]

        if not nps.empty:
            weights = nps[["promoter","passive","detractor","unknown"]].sum(axis=1)
            overall_nps = float((nps["NPS%"] * weights).sum() / weights.sum()) if weights.sum() else np.nan
        else:
            overall_nps = np.nan
        st.markdown(f"### Overall NPS (selected range): **{overall_nps:.1f}**")

        tab1, tab2, tab3 = st.tabs(["Overview", "Sentiments", "NPS Correlation"])

        # ---- Tab 1: overview ----
        detail_df = pd.DataFrame()
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

        # ---- Tab 2: sentiments ----
        with tab2:
            sd = _sentiments(s)
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
                    for col in pv.columns:
                        ax2.bar(x, pv[col].values, bottom=bottom, label=col.capitalize())
                        bottom += pv[col].values
                    ax2.set_xticks(x); ax2.set_xticklabels(pv.index, rotation=90)
                    for sp in ["top","right","left"]: ax2.spines[sp].set_visible(False)
                    ax2.spines["bottom"].set_color("#D3D3D3")
                    ax2.get_yaxis().set_visible(False); ax2.grid(False)
                    ax2.set_title("Suggestions Sentiment by Portfolio", fontsize=12)
                    ax2.legend(loc="best", fontsize=8, frameon=False)
                    st.pyplot(fig2, use_container_width=True)
                with s_right:
                    cat = (sd.groupby("Portfolio")["sent_label"].value_counts()
                             .unstack(fill_value=0)
                             .reindex(columns=[c for c in ["positive","neutral","negative"] if c in sd["sent_label"].unique()], fill_value=0))
                    cat["Sugg"]   = cat.sum(axis=1)
                    cat["Pos%"]   = (cat.get("positive",0)/cat["Sugg"]*100).round(1)
                    cat["Neg%"]   = (cat.get("negative",0)/cat["Sugg"]*100).round(1)
                    cat["NetSent%"] = (cat["Pos%"] - cat["Neg%"]).round(1)
                    st.markdown("#### Sentiment Summary (filtered range)")
                    st.dataframe(cat[["Sugg","Pos%","Neg%","NetSent%"]], use_container_width=True)

        # ---- Tab 3: correlation (NPS × Sentiment × SLA% + Complaints chart) ----
        combined_df = pd.DataFrame()
        with tab3:
            ops = _prep_ops(ops_raw)
            ops_kpi = _ops_kpis(ops)
            comp = _prep_complaints(complaints_raw)

            sd_all = _sentiments(s)
            sent_m = (sd_all.groupby(["Portfolio","_month"])
                        .agg(Sugg=("Suggestions","count"),
                             Pos=("sent_label", lambda s: (s=="positive").sum()),
                             Neg=("sent_label", lambda s: (s=="negative").sum()))
                        .reset_index())
            if not sent_m.empty:
                sent_m["Pos%"] = (sent_m["Pos"]/sent_m["Sugg"]*100).round(1)
                sent_m["Neg%"] = (sent_m["Neg"]/sent_m["Sugg"]*100).round(1)
                sent_m["NetSent%"] = (sent_m["Pos%"] - sent_m["Neg%"]).round(1)

            base = nps[["Portfolio","_month","NPS%"]].rename(columns={"NPS%":"NPS"}).copy()
            combined_df = base.merge(sent_m[["Portfolio","_month","Sugg","Pos%","Neg%","NetSent%"]], how="left", on=["Portfolio","_month"])
            if not ops_kpi.empty:
                combined_df = combined_df.merge(ops_kpi[["Portfolio","_month","SLA%","Completes"]], how="left", on=["Portfolio","_month"])
            if not comp.empty:
                comp_m = comp.groupby(["Portfolio","_month"]).size().to_frame("Complaints").reset_index()
                combined_df = combined_df.merge(comp_m, how="left", on=["Portfolio","_month"])

            if sel_port != "(All)": combined_df = combined_df[combined_df["Portfolio"] == sel_port]
            if start is not None:  combined_df = combined_df[combined_df["_month"] >= start]
            if end   is not None:  combined_df = combined_df[combined_df["_month"] <= end]

            latest_m = combined_df["_month"].max() if not combined_df.empty else None
            latest = combined_df[combined_df["_month"] == latest_m].copy() if latest_m is not None else pd.DataFrame()

            c_left, c_right = st.columns([1,1])
            with c_left:
                # keep old SLA info if missing
                if (latest.empty) or ("SLA%" not in latest.columns) or (latest["SLA%"].notna().sum() == 0):
                    st.info("SLA/Service data not available for the latest month; showing table only.")

                # NEW: NPS vs Complaints (or Complaints/1000) chart
                if not latest.empty and "NPS" in latest.columns and "Complaints" in latest.columns:
                    # numeric complaints
                    compl = pd.to_numeric(latest["Complaints"], errors="coerce")
                    y = compl.copy()
                    y_label = "Complaints (count)"
                    if "Completes" in latest.columns:
                        completes = pd.to_numeric(latest["Completes"], errors="coerce")
                        rate = compl / completes * 1000
                        # use rate where completes>0, else fallback to counts
                        y = np.where((completes > 0) & np.isfinite(rate), rate, compl)
                        y = pd.Series(y, index=latest.index)
                        y_label = "Complaints per 1000 (fallback to count if completes=0)"

                    dfp = latest.copy()
                    dfp["y"] = y
                    dfp = dfp[dfp["NPS"].notna() & dfp["y"].notna()]
                    if len(dfp) >= 2:
                        figc, axc = plt.subplots()
                        size = (dfp["Sugg"].fillna(0).astype(float) + 1) * 3
                        axc.scatter(dfp["NPS"], dfp["y"], s=size)
                        for _, r in dfp.iterrows():
                            axc.annotate(r["Portfolio"], (r["NPS"], r["y"]), fontsize=8, xytext=(3,3), textcoords="offset points")
                        for sp in ["top","right","left"]: axc.spines[sp].set_visible(False)
                        axc.spines["bottom"].set_color("#D3D3D3")
                        axc.get_yaxis().set_visible(False); axc.grid(False)
                        axc.set_xlabel("NPS %"); axc.set_title(f"NPS vs {y_label} (Latest: {str(latest_m)})", fontsize=12)
                        st.pyplot(figc, use_container_width=True)
                    else:
                        st.info("Not enough data points to plot NPS vs Complaints for the latest month.")

            with c_right:
                view = combined_df.rename(columns={"_month":"Month"}).copy()
                for c in ["NPS","SLA%","Pos%","Neg%","NetSent%"]:
                    if c in view.columns: view[c] = view[c].round(1)
                cols = [c for c in ["Portfolio","Month","NPS","SLA%","NetSent%","Pos%","Neg%","Sugg","Complaints","Completes"] if c in view.columns]
                st.markdown("#### Combined KPIs (Portfolio × Month)")
                st.dataframe(view[cols].sort_values(["Month","Portfolio"]), use_container_width=True)

            # keep existing correlation captions
            try:
                c1, c2 = st.columns(2)
                with c1:
                    corr = view[["NPS","SLA%"]].dropna() if "SLA%" in view.columns else pd.DataFrame()
                    st.caption(f"Pearson(NPS, SLA%): **{float(corr.corr().iloc[0,1]):.2f}**" if len(corr)>=2 else "Pearson(NPS, SLA%): not enough data")
                with c2:
                    corr2 = view[["NPS","NetSent%"]].dropna() if "NetSent%" in view.columns else pd.DataFrame()
                    st.caption(f"Pearson(NPS, Net Sentiment): **{float(corr2.corr().iloc[0,1]):.2f}**" if len(corr2)>=2 else "Pearson(NPS, Net Sentiment): not enough data")
            except Exception:
                pass

        host_df = combined_df.copy() if not combined_df.empty else detail_df.copy()
        if host_df is None or host_df.empty:
            host_df = pd.DataFrame([{"Message": "No data for selected filters"}])
        return ("NPS by Portfolio", "Surveys (Sheet 1) with Sentiments and SLA correlation"), host_df

    except Exception as e:
        import traceback
        st.error(f"NPS module error: {e}")
        st.code(traceback.format_exc())
        safe = pd.DataFrame([{"error": str(e)}])
        return ("NPS by Portfolio", "Recovered from error"), safe
