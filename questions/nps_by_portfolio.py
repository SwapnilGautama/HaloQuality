# questions/nps_by_portfolio.py
from __future__ import annotations

from typing import Dict, Any, Optional
import re
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt


# ----------------- helpers -----------------
def _soft_pastels(n: int) -> list:
    base = ["#A3C4F3", "#CDE7BE", "#F6C1C1", "#FFD6A5", "#BDB2FF", "#FFAFCC", "#BEE1E6", "#E2ECE9"]
    return [base[i % len(base)] for i in range(n)]

def _norm_portfolio(x: str) -> str:
    if not isinstance(x, str):
        return "Unknown"
    t = x.strip().title()
    t = t.replace("Baes-Leatherhead", "Leatherhead - Baes").replace("Leatherhead  - Baes", "Leatherhead - Baes")
    return t

def _vlabels(ax):
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)

# ----------------- surveys (NPS + suggestions) -----------------
_POS = {"good","great","excellent","amazing","helpful","fast","quick","responsive","easy","clear",
        "friendly","polite","supportive","smooth","love","efficient","prompt","awesome","happy"}
_NEG = {"bad","poor","terrible","slow","delay","delayed","waiting","confusing","unclear","hard",
        "rude","unhelpful","expensive","issue","problem","bug","error","crash","difficult","worst"}

def _lex_sentiment(text: str) -> float:
    if not isinstance(text, str) or not text.strip():
        return 0.0
    toks = re.findall(r"\b\w+\b", text.lower())
    if not toks: return 0.0
    score = sum(1 for t in toks if t in _POS) - sum(1 for t in toks if t in _NEG)
    return max(-1.0, min(1.0, score / max(len(toks), 4)))

def _prep_surveys(df_raw: pd.DataFrame) -> pd.DataFrame:
    if df_raw is None or df_raw.empty: return pd.DataFrame()
    df = df_raw.copy()
    if "Portfolio" in df.columns:
        df["Portfolio"] = df["Portfolio"].map(_norm_portfolio)
    else:
        df["Portfolio"] = "Unknown"
    # month
    if "Month_received" in df.columns:
        parsed = pd.to_datetime(df["Month_received"], errors="coerce", dayfirst=True, infer_datetime_format=True)
        miss = parsed.isna()
        if miss.any():
            mm = df.loc[miss, "Month_received"].astype(str).str[:3]
            parsed.loc[miss] = pd.to_datetime(mm + " 1 2025", errors="coerce", format="%b %d %Y")
        df["_month"] = parsed.dt.to_period("M")
    else:
        df["_month"] = pd.NaT
    # NPS
    if "NPS" in df.columns:
        s = pd.to_numeric(df["NPS"], errors="coerce")
        bucket = np.where(s >= 9, "promoter", np.where(s >= 7, "passive", np.where(s >= 0, "detractor","unknown")))
        df["nps_bucket"] = bucket
    else:
        df["nps_bucket"] = "unknown"
    # suggestions
    if "Suggestions" in df.columns:
        df["Suggestions"] = df["Suggestions"].astype(str).str.strip()
        df.loc[df["Suggestions"].str.lower().isin(["", "nan", "none", "null"]), "Suggestions"] = np.nan
    else:
        df["Suggestions"] = np.nan
    return df

def _aggregate_nps(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty: return pd.DataFrame()
    df = df[df["_month"].notna()].copy()
    g = df.groupby(["Portfolio", "_month"])["nps_bucket"].value_counts().unstack(fill_value=0)
    for col in ["promoter","passive","detractor","unknown"]:
        if col not in g.columns: g[col] = 0
    g["Total"] = g[["promoter","passive","detractor","unknown"]].sum(axis=1).replace(0, np.nan)
    g["NPS%"] = ((g["promoter"] - g["detractor"]) / g["Total"]) * 100
    return g.reset_index()

def _sentiments(df: pd.DataFrame) -> pd.DataFrame:
    sug = df[df["Suggestions"].notna()].copy()
    if sug.empty: return sug
    score = sug["Suggestions"].map(_lex_sentiment)
    lab = np.where(score >= 0.05, "positive", np.where(score <= -0.05, "negative", "neutral"))
    sug["sent_score"] = score; sug["sent_label"] = lab
    return sug

# ----------------- service/accuracy proxies -----------------
def _prep_ops(df_ops: pd.DataFrame) -> pd.DataFrame:
    if df_ops is None or df_ops.empty: return pd.DataFrame()
    d = df_ops.copy()
    d["Portfolio"] = d["Portfolio"].map(_norm_portfolio)
    d["_month"] = pd.to_datetime(d["Report_Date"], errors="coerce").dt.to_period("M")
    # Within SLA → 1/0
    if "Within SLA" in d.columns:
        d["_within"] = d["Within SLA"].astype(str).str.lower().str.contains("within").astype(int)
    else:
        d["_within"] = 0
    # Completes
    comp_col = "Completes" if "Completes" in d.columns else None
    if comp_col:
        d["_completes"] = pd.to_numeric(d[comp_col], errors="coerce").fillna(0.0)
    else:
        d["_completes"] = 1.0
    return d

def _ops_kpis(d_ops: pd.DataFrame) -> pd.DataFrame:
    if d_ops.empty: return pd.DataFrame()
    g = d_ops.groupby(["Portfolio","_month"]).agg(
        Within=("._within".replace(".",""), "sum") if "._within" in d_ops.columns else ("_within","sum"),
        Total=("Portfolio","size"),
        Completes=("_completes","sum")
    ).reset_index()
    g["SLA%"] = (g["Within"] / g["Total"]) * 100
    return g

def _prep_complaints(df_comp: pd.DataFrame) -> pd.DataFrame:
    if df_comp is None or df_comp.empty: return pd.DataFrame()
    c = df_comp.copy()
    c["Portfolio"] = c["Portfolio"].map(_norm_portfolio)
    # Month from "Report Month" like "Jun", backfill with complaint date
    if "Report Month" in c.columns:
        mm = c["Report Month"].astype(str).str[:3].str.title()
        m1 = pd.to_datetime(mm + " 1 2025", errors="coerce", format="%b %d %Y")
    else:
        m1 = pd.NaT
    if "Date Complaint Received - DD/MM/YY" in c.columns:
        m2 = pd.to_datetime(c["Date Complaint Received - DD/MM/YY"], dayfirst=True, errors="coerce")
    else:
        m2 = pd.NaT
    months = m1.fillna(m2)
    c["_month"] = pd.to_datetime(months, errors="coerce").dt.to_period("M")
    return c

# ----------------- UI entry -----------------
def run(store: Dict[str, Any], params: Dict[str, Any], user_text: Optional[str] = None):
    """
    Expects in `store`:
      - surveys: NPS & Suggestions (Sheet 1)
      - ops or fpa: operational rows with Report_Date, Portfolio, Within SLA, Completes
      - complaints: optional complaints file
    """
    surveys = store.get("surveys", pd.DataFrame())
    ops_raw = store.get("ops", store.get("fpa", pd.DataFrame()))
    complaints_raw = store.get("complaints", pd.DataFrame())

    s = _prep_surveys(surveys)
    if s.empty or s["_month"].isna().all():
        msg = pd.DataFrame([{"Message": "No usable surveys (check Month_received, NPS, Suggestions)."}])
        return ("NPS by Portfolio", "Surveys (Sheet 1)"), msg

    # --- month & filters ---
    with st.sidebar:
        st.header("Filters")
        ports = ["(All)"] + sorted(s["Portfolio"].dropna().unique().tolist())
        sel_port = st.selectbox("Portfolio", ports, index=0)
        months = s["_month"].dropna().sort_values().unique().tolist()
        start = st.selectbox("From month", months, index=0) if months else None
        end = st.selectbox("To month", months, index=len(months)-1) if months else None

    nps = _aggregate_nps(s)
    if sel_port != "(All)": nps = nps[nps["Portfolio"] == sel_port]
    if start is not None: nps = nps[nps["_month"] >= start]
    if end is not None: nps = nps[nps["_month"] <= end]

    # overall NPS (weighted by Total)
    if not nps.empty:
        num = (nps["NPS%"] * (nps["promoter"] + nps["passive"] + nps["detractor"] + nps["unknown"])).sum()
        den = (nps[["promoter","passive","detractor","unknown"]].sum(axis=1)).sum()
        overall_nps = float(num/den) if den else np.nan
    else:
        overall_nps = np.nan
    st.markdown(f"### Overall NPS (selected range): **{overall_nps:.1f}**")

    # ------------- tabs -------------
    tab1, tab2, tab3 = st.tabs(["Overview", "Sentiments", "NPS Correlation"])

    # -- Tab 1: overview (trend + detail) --
    detail_df = pd.DataFrame()
    with tab1:
        left, right = st.columns([1,1])
        with left:
            if not nps.empty:
                fig, ax = plt.subplots()
                for i, (p, g) in enumerate(nps.groupby("Portfolio")):
                    g = g.sort_values("_month")
                    ax.plot(g["_month"].astype(str), g["NPS%"], marker="o", linewidth=2, markersize=4,
                            label=p, color=_soft_pastels(8)[i % 8])
                for sp in ["top","right","left"]: ax.spines[sp].set_visible(False)
                ax.spines["bottom"].set_color("#D3D3D3")
                ax.get_yaxis().set_visible(False); ax.grid(False); ax.set_xlabel(""); ax.set_title("NPS Trend", fontsize=12)
                ax.legend(loc="best", fontsize=8, frameon=False)
                st.pyplot(fig, use_container_width=True)
        with right:
            cols = ["Portfolio","_month","NPS%","promoter","passive","detractor","unknown"]
            if not nps.empty:
                detail_df = nps[cols].rename(columns={"_month":"Month","NPS%":"NPS"}).copy()
                detail_df["NPS"] = detail_df["NPS"].round(1)
            st.markdown("#### Detail (by Portfolio × Month)")
            st.dataframe(detail_df, use_container_width=True)

    # -- Tab 2: sentiments --
    with tab2:
        sd = _sentiments(s)
        if sel_port != "(All)": sd = sd[sd["Portfolio"] == sel_port]
        if start is not None: sd = sd[sd["_month"] >= start]
        if end is not None: sd = sd[sd["_month"] <= end]

        if sd.empty:
            st.info("No suggestions available in the selected range.")
        else:
            s_left, s_right = st.columns([1,1])
            with s_left:
                pv = sd.pivot_table(index="Portfolio", columns="sent_label", values="Suggestions", aggfunc="count", fill_value=0)
                # keep order negative/neutral/positive for stacked feel
                order = [c for c in ["negative","neutral","positive"] if c in pv.columns]
                pv = pv[order]
                fig2, ax2 = plt.subplots()
                x = np.arange(len(pv.index)); bottom = np.zeros(len(x))
                for col in pv.columns:
                    ax2.bar(x, pv[col].values, bottom=bottom, label=col.capitalize())
                    bottom += pv[col].values
                ax2.set_xticks(x); ax2.set_xticklabels(pv.index); _vlabels(ax2)
                for sp in ["top","right","left"]: ax2.spines[sp].set_visible(False)
                ax2.spines["bottom"].set_color("#D3D3D3")
                ax2.get_yaxis().set_visible(False); ax2.grid(False); ax2.set_title("Suggestions Sentiment by Portfolio", fontsize=12)
                ax2.legend(loc="best", fontsize=8, frameon=False)
                st.pyplot(fig2, use_container_width=True)
            with s_right:
                total = len(sd)
                cat = (sd.groupby("Portfolio")["sent_label"].value_counts()
                         .unstack(fill_value=0)
                         .reindex(columns=[c for c in ["positive","neutral","negative"] if c in sd["sent_label"].unique()], fill_value=0))
                cat["Sugg"] = cat.sum(axis=1)
                cat["Pos%"] = (cat.get("positive",0)/cat["Sugg"]*100).round(1)
                cat["Neg%"] = (cat.get("negative",0)/cat["Sugg"]*100).round(1)
                cat["NetSent%"] = (cat["Pos%"] - cat["Neg%"]).round(1)
                st.markdown("#### Sentiment Summary (filtered range)")
                st.dataframe(cat[["Sugg","Pos%","Neg%","NetSent%"]], use_container_width=True)

    # -- Tab 3: correlation (NPS × Sentiment × SLA%) --
    # ops/service
    ops = _prep_ops(ops_raw)
    ops_kpi = _ops_kpis(ops)
    # complaints (optional)
    comp = _prep_complaints(complaints_raw)

    # join
    combined_df = pd.DataFrame()
    nps_for_join = nps[["Portfolio","_month","NPS%"]].rename(columns={"NPS%":"NPS"})
    with tab3:
        # Sentiment month-level KPIs
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

        combined_df = nps_for_join.merge(sent_m[["Portfolio","_month","Sugg","Pos%","Neg%","NetSent%"]], how="left", on=["Portfolio","_month"])
        if not ops_kpi.empty:
            combined_df = combined_df.merge(ops_kpi[["Portfolio","_month","SLA%","Completes"]], how="left", on=["Portfolio","_month"])
        if not comp.empty:
            comp_m = comp.groupby(["Portfolio","_month"]).size().to_frame("Complaints").reset_index()
            combined_df = combined_df.merge(comp_m, how="left", on=["Portfolio","_month"])

        # Apply same filters
        if sel_port != "(All)": combined_df = combined_df[combined_df["Portfolio"] == sel_port]
        if start is not None: combined_df = combined_df[combined_df["_month"] >= start]
        if end is not None: combined_df = combined_df[combined_df["_month"] <= end]

        if combined_df.empty:
            st.info("No overlapped data across NPS, Sentiment, and Service for the selected range.")
        else:
            latest_m = combined_df["_month"].max()
            latest = combined_df[combined_df["_month"] == latest_m].copy()

            c_left, c_right = st.columns([1,1])
            with c_left:
                if "SLA%" in latest.columns and latest["SLA%"].notna().any():
                    fig3, ax3 = plt.subplots()
                    x = latest["NPS"]; y = latest["SLA%"]; s = (latest["Sugg"].fillna(0)+1)*3
                    ax3.scatter(x, y, s=s)
                    for _, r in latest.iterrows():
                        ax3.annotate(r["Portfolio"], (r["NPS"], r["SLA%"]), fontsize=8, xytext=(3,3), textcoords="offset points")
                    for sp in ["top","right","left"]: ax3.spines[sp].set_visible(False)
                    ax3.spines["bottom"].set_color("#D3D3D3")
                    ax3.get_yaxis().set_visible(False); ax3.grid(False)
                    ax3.set_xlabel("NPS %"); ax3.set_title(f"NPS vs SLA% (Latest: {str(latest_m)})", fontsize=12)
                    st.pyplot(fig3, use_container_width=True)
                else:
                    st.info("SLA/Service data not available for the latest month; showing table only.")

            with c_right:
                view = combined_df.rename(columns={"_month":"Month"}).copy()
                for c in ["NPS","SLA%","Pos%","Neg%","NetSent%"]:
                    if c in view.columns: view[c] = view[c].round(1)
                cols = [c for c in ["Portfolio","Month","NPS","SLA%","NetSent%","Pos%","Neg%","Sugg","Complaints","Completes"] if c in view.columns]
                st.markdown("#### Combined KPIs (Portfolio × Month)")
                st.dataframe(view[cols].sort_values(["Month","Portfolio"]), use_container_width=True)

            # mini correlations across the selection
            try:
                c1, c2 = st.columns(2)
                with c1:
                    corr = view[["NPS","SLA%"]].dropna()
                    st.caption(f"Pearson(NPS, SLA%): **{float(corr.corr().iloc[0,1]):.2f}**" if len(corr)>=2 else "Pearson(NPS, SLA%): not enough data")
                with c2:
                    corr2 = view[["NPS","NetSent%"]].dropna()
                    st.caption(f"Pearson(NPS, Net Sentiment): **{float(corr2.corr().iloc[0,1]):.2f}**" if len(corr2)>=2 else "Pearson(NPS, Net Sentiment): not enough data")
            except Exception:
                pass

    # ---------- host return (never None) ----------
    host_df = combined_df.copy() if not combined_df.empty else detail_df.copy()
    if host_df is None or host_df.empty:
        host_df = pd.DataFrame([{"Message": "No data for selected filters"}])

    return ("NPS by Portfolio", "Surveys (Sheet 1) with Sentiments and SLA correlation"), host_df
