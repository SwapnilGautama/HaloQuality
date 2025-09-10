# questions/nps_by_portfolio.py
from __future__ import annotations

from typing import Dict, Any, Optional, Iterable
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


# ----------------- palette & styling -----------------
_DARK_BLUE = "#0b3d91"   # titles
_DARK_GREY = "#333333"   # all chart fonts
_SOFT_GREY = "#E0E0E0"   # axes
_BUBBLE_FILL = "#8ECAE6" # soft pastel bubble
_BUBBLE_EDGE = "#5A7AA1" # soft pastel edge

_SENT_NEG = "#9BBBD4"
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
    ax.plot(x, mm["NPS"], linewidth=2.8, color=_BUBBLE_FILL, label="NPS %")
    ax.plot(x, mm["Pos%"], linewidth=2.8, color=_SENT_POS, label="Positive %")
    ax.set_xticks(x); ax.set_xticklabels(mm["_label"].tolist(), rotation=0, color=_DARK_GREY)
    _style_axes(ax); ax.get_yaxis().set_visible(False); ax.set_xlabel("")
    ax.set_title("MoM Trend — NPS % & Positive Sentiment %", color=_DARK_BLUE, pad=10)
    ax.legend(frameon=False, loc="upper right")
    return fig


# ----------------- surveys (NPS + suggestions) -----------------
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
    bucket = np.where(score >= 9, "promoter", np.where(score >= 7, "passive", np.where(score >= 0, "detractor","unknown")))
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


# ----------------- legacy ops helpers (used by other tabs) -----------------
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


# ----------------- complaints -----------------
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


# ----------------- UI entry -----------------
def run(store: Dict[str, Any], params: Dict[str, Any], user_text: Optional[str] = None):
    try:
        surveys = store.get("surveys", pd.DataFrame())
        complaints_raw = store.get("complaints", pd.DataFrame())

        s = _prep_surveys(surveys)
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

        # NPS aggregates
        nps = _aggregate_nps(s)
        if sel_port != "(All)": nps = nps[nps["Portfolio"] == sel_port]
        if start is not None:  nps = nps[nps["_month"] >= start]
        if end   is not None:  nps = nps[nps["_month"] <= end]

        # overall NPS
        if not nps.empty:
            weights = nps[["promoter","passive","detractor","unknown"]].sum(axis=1)
            overall_nps = float((nps["NPS%"] * weights).sum() / weights.sum()) if weights.sum() else np.nan
        else:
            overall_nps = np.nan
        st.markdown(f"### Overall NPS (selected range): **{overall_nps:.1f}**")

        tab1, tab2, tab3 = st.tabs(["Overview", "Sentiments", "NPS Correlation"])

        # ---------- Tab 1 (unchanged) ----------
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

        # ---------- Tab 2 (unchanged) ----------
        with tab2:
            sd = (s[s["Suggestions"].notna()].copy()
                    if "Suggestions" in s.columns else pd.DataFrame())
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
                             .reindex(columns=[c for c in ["positive","neutral","negative"] if c in sd["sent_label"].unique()], fill_value=0))
                    cat["Sugg"]   = cat.sum(axis=1)
                    cat["Pos%"]   = (cat.get("positive",0)/cat["Sugg"]*100).round(1)
                    cat["Neg%"]   = (cat.get("negative",0)/cat["Sugg"]*100).round(1)
                    cat["NetSent%"] = (cat["Pos%"] - cat["Neg%"]).round(1)

                    if not nps.empty:
                        npstab = (nps.groupby("Portfolio")[["promoter","detractor","passive","unknown"]]
                                    .sum(min_count=1))
                        npstab["Total"] = npstab[["promoter","detractor","passive","unknown"]].sum(axis=1)
                        npstab["NPS"] = ((npstab["promoter"] - npstab["detractor"]) /
                                         npstab["Total"].replace(0, np.nan) * 100).round(1)
                        npstab = npstab.rename(columns={"promoter":"Promoters","detractor":"Detractors"})
                        cat = cat.join(npstab[["NPS","Promoters","Detractors"]], how="left")

                    ordered_cols = [c for c in ["Sugg","Pos%","Neg%","NetSent%","NPS","Promoters","Detractors"] if c in cat.columns]
                    st.markdown("#### Sentiment Summary (filtered range)")
                    st.dataframe(cat[ordered_cols], use_container_width=True)

        # ---------- Tab 3: NPS Correlation (UPDATED to align with data_store) ----------
        with tab3:
            base = nps[["Portfolio","_month","NPS%"]].rename(columns={"NPS%":"NPS"}).copy()

            # FPA from 'fpa' (raw) and 'cases' (canonical); union then compute %
            def _fpa_from_df(df: pd.DataFrame) -> pd.DataFrame:
                if df is None or df.empty:
                    return pd.DataFrame(columns=["Portfolio","_month","FPA_num","FPA_den"])
                d = df.copy()
                # Portfolio normalization
                pcol = _find_col(d, ["portfolio"]) or "portfolio"
                if pcol in d.columns:
                    d["Portfolio"] = d[pcol].astype(str).map(_norm_portfolio)
                else:
                    d["Portfolio"] = "Unknown"
                # Month column:
                # prefer Activity Date, else the canonical 'date' that data_store creates for cases,
                # else common fallbacks.
                date_col = (_find_col(d, ["activity date","activity_date","activitydate"]) or
                            _find_col(d, ["date"]) or
                            _find_col(d, ["create date","created date","create_date","report date","report_date"]))
                d["_month"] = pd.to_datetime(d[date_col], errors="coerce", dayfirst=True).dt.to_period("M") if date_col else pd.NaT
                # Review result
                rcol = (_find_col(d, ["review result","review_result","qa result","fpa result","result"]) or
                        _find_col(d, ["pass"]))
                if rcol:
                    is_pass = d[rcol].astype(str).str.strip().str.lower().isin(["pass","p","true","1","yes"])
                else:
                    is_pass = pd.Series([np.nan]*len(d))
                d["_is_pass"] = pd.to_numeric(is_pass, errors="coerce")
                g = (d.dropna(subset=["_month"])
                       .groupby(["Portfolio","_month"])["_is_pass"]
                       .agg(FPA_num="sum", FPA_den="count")
                       .reset_index())
                return g

            fpa_sources = [
                store.get("fpa", pd.DataFrame()),   # raw FPA folder (not normalized in data_store)
                store.get("cases", pd.DataFrame()), # canonical cases (may contain review result)
            ]
            fpa_parts = [ _fpa_from_df(x) for x in fpa_sources if isinstance(x, pd.DataFrame) and not x.empty ]
            if fpa_parts:
                fpa_union = pd.concat(fpa_parts, ignore_index=True)
                fpa_union = (fpa_union
                             .groupby(["Portfolio","_month"], as_index=False)
                             .agg(FPA_num=("FPA_num","sum"), FPA_den=("FPA_den","sum")))
                fpa_union["FPA%"] = (pd.to_numeric(fpa_union["FPA_num"], errors="coerce") /
                                     pd.to_numeric(fpa_union["FPA_den"], errors="coerce")) * 100.0
                fpa_out = fpa_union[["Portfolio","_month","FPA%"]]
            else:
                fpa_out = pd.DataFrame(columns=["Portfolio","_month","FPA%"])

            # Total Cases Complete from canonical cases (data_store renames ID -> 'id', month -> '_month')
            cases_df = store.get("cases", pd.DataFrame())
            if cases_df is None or cases_df.empty:
                completes_out = pd.DataFrame(columns=["Portfolio","_month","Total Cases Complete"])
            else:
                d = cases_df.copy()
                if "portfolio" in d.columns:
                    d["Portfolio"] = d["portfolio"].astype(str).map(_norm_portfolio)
                else:
                    pcol = _find_col(d, ["portfolio"])
                    d["Portfolio"] = d[pcol].astype(str).map(_norm_portfolio) if pcol else "Unknown"
                # _month should already exist per data_store; still build if missing
                if "_month" not in d.columns:
                    date_c = _find_col(d, ["date","create date","created date","create_date"])
                    d["_month"] = pd.to_datetime(d[date_c], errors="coerce", dayfirst=True).dt.to_period("M") if date_c else pd.NaT
                idcol = "id" if "id" in d.columns else (_find_col(d, ["case id","unique identifier","case"]) or None)
                if idcol:
                    completes_out = (d.dropna(subset=["_month"])
                                       .groupby(["Portfolio","_month"])[idcol]
                                       .nunique().reset_index()
                                       .rename(columns={idcol: "Total Cases Complete"}))
                else:
                    completes_out = (d.dropna(subset=["_month"])
                                       .groupby(["Portfolio","_month"])
                                       .size().to_frame("Total Cases Complete")
                                       .reset_index())

            # Complaints from canonical complaints (already has '_month')
            comp_df = complaints_raw.copy()
            if comp_df is None or comp_df.empty:
                comp_out = pd.DataFrame(columns=["Portfolio","_month","Total Complaints"])
            else:
                d = comp_df.copy()
                if "portfolio" in d.columns:
                    d["Portfolio"] = d["portfolio"].astype(str).map(_norm_portfolio)
                else:
                    pcol = _find_col(d, ["portfolio"])
                    d["Portfolio"] = d[pcol].astype(str).map(_norm_portfolio) if pcol else "Unknown"
                if "_month" not in d.columns:
                    dcol = (_find_col(d, ["date complaint received - dd/mm/yy","date complaint received","date","received date"]))
                    d["_month"] = pd.to_datetime(d[dcol], errors="coerce", dayfirst=True).dt.to_period("M") if dcol else pd.NaT
                comp_out = (d.dropna(subset=["_month"])
                              .groupby(["Portfolio","_month"])
                              .size().to_frame("Total Complaints")
                              .reset_index())

            # Merge panel
            combined = base.merge(fpa_out, on=["Portfolio","_month"], how="outer")
            combined = combined.merge(completes_out, on=["Portfolio","_month"], how="outer")
            combined = combined.merge(comp_out, on=["Portfolio","_month"], how="outer")

            # KPI derivation
            combined["Complaints/1000"] = (
                pd.to_numeric(combined.get("Total Complaints"), errors="coerce") /
                pd.to_numeric(combined.get("Total Cases Complete"), errors="coerce")
            ) * 1000.0

            # Apply filters
            if sel_port != "(All)":
                combined = combined[combined["Portfolio"] == sel_port]
            if start is not None:
                combined = combined[combined["_month"] >= start]
            if end is not None:
                combined = combined[combined["_month"] <= end]

            # Present table
            view = combined.rename(columns={"_month":"Month"})
            for c in ["NPS","FPA%","Complaints/1000"]:
                if c in view.columns:
                    view[c] = pd.to_numeric(view[c], errors="coerce").round(1)

            cols = [c for c in [
                "Portfolio","Month","NPS","FPA%","Complaints/1000","Total Complaints","Total Cases Complete"
            ] if c in view.columns]

            st.markdown("#### Combined KPIs — NPS, FPA%, Complaints/1000")
            if view.empty or not cols:
                st.info("No data available for the selected filters.")
            else:
                st.dataframe(view[cols].sort_values(["Month","Portfolio"]), use_container_width=True)

        host_df = pd.DataFrame()
        return ("NPS by Portfolio", "Surveys (Sheet 1) with Sentiments and SLA/Complaints correlation"), host_df

    except Exception as e:
        import traceback
        st.error(f"NPS module error: {e}")
        st.code(traceback.format_exc())
        safe = pd.DataFrame([{"error": str(e)}])
        return ("NPS by Portfolio", "Recovered from error"), safe
