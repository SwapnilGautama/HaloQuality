# questions/nps_by_portfolio.py
from __future__ import annotations

from typing import Dict, Any, Optional, Iterable
import re
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# ----------------- palette & styling -----------------
_DARK_BLUE = "#0b3d91"   # titles
_DARK_GREY = "#333333"   # text
_SOFT_GREY = "#E0E0E0"   # axes
_SENT_NEG = "#9BBBD4"
_SENT_NEU = "#F4C27A"
_SENT_POS = "#7BC47F"
_BUBBLE_FILL = "#8ECAE6"
_BUBBLE_EDGE = "#5A7AA1"

def _soft_pastels(n: int) -> list:
    base = ["#A3C4F3","#CDE7BE","#F6C1C1","#FFD6A5","#BDB2FF","#FFAFCC","#BEE1E6","#E2ECE9"]
    return [base[i % len(base)] for i in range(n)]

def _style_axes(ax: plt.Axes) -> None:
    ax.tick_params(colors=_DARK_GREY, labelcolor=_DARK_GREY)
    for sp in ("top","right"):
        ax.spines[sp].set_visible(False)
    for sp in ("bottom","left"):
        ax.spines[sp].set_color(_SOFT_GREY)
        ax.spines[sp].set_linewidth(1.25)
    ax.grid(False)

# ----------------- helpers -----------------
def _find_col(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    if df is None or df.empty:
        return None
    m = {str(c).strip().lower(): c for c in df.columns}
    for c in candidates:
        lc = str(c).strip().lower()
        if lc in m:
            return m[lc]
    for c in df.columns:
        cl = str(c).strip().lower()
        for n in candidates:
            if str(n).strip().lower() in cl:
                return c
    return None

def _norm_portfolio(x: Any) -> str:
    if pd.isna(x):
        return "Unknown"
    t = str(x).strip().title()
    return (t.replace("Baes-Leatherhead", "Leatherhead - Baes")
             .replace("Leatherhead  - Baes", "Leatherhead - Baes"))

# ----------------- lexical sentiment -----------------
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

# ----------------- surveys (NPS + suggestions) -----------------
def _prep_surveys(df_raw: pd.DataFrame) -> pd.DataFrame:
    if df_raw is None or df_raw.empty:
        return pd.DataFrame()
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

    s_col = _find_col(df, ["suggestions","suggestion","comments","comment","feedback"])
    df["Suggestions"] = df[s_col].astype(str).str.strip() if s_col else np.nan
    df.loc[df["Suggestions"].astype(str).str.lower().isin(["","nan","none","null"]), "Suggestions"] = np.nan
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
    sug["sent_score"] = score
    sug["sent_label"] = lab
    return sug

# ----------------- MoM mini chart (sentiments tab) -----------------
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

    if not sd_df.empty and "sent_label" in sd_df.columns:
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

# ----------------- main -----------------
def run(store: Dict[str, Any], params: Dict[str, Any], user_text: Optional[str] = None):
    # Surveys
    surveys = store.get("surveys", pd.DataFrame())
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
        end   = st.selectbox("To month", months, index=len(months)-1) if months else None

    # NPS aggregate
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

    # -------- Tab 1 (unchanged) --------
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
                ax.spines["bottom"].set_color(_SOFT_GREY)
                ax.get_yaxis().set_visible(False); ax.grid(False); ax.set_xlabel("")
                ax.set_title("NPS Trend", fontsize=12, color=_DARK_BLUE)
                ax.legend(loc="best", fontsize=8, frameon=False)
                st.pyplot(fig, use_container_width=True)
        with right:
            if not nps.empty:
                detail_df = nps[["Portfolio","_month","NPS%","promoter","passive","detractor","unknown"]]\
                               .rename(columns={"_month":"Month","NPS%":"NPS"}).copy()
                detail_df["NPS"] = detail_df["NPS"].round(1)
            st.markdown("#### Detail (by Portfolio × Month)")
            st.dataframe(detail_df, use_container_width=True)

    # -------- Tab 2 (unchanged visuals; robust if suggestions missing) --------
    with tab2:
        sd = _sentiments(s)
        if sel_port != "(All)": sd = sd[sd["Portfolio"] == sel_port]
        if start is not None:  sd = sd[sd["_month"] >= start]
        if end   is not None:  sd = sd[sd["_month"] <= end]

        if sd.empty or "sent_label" not in sd.columns:
            st.info("No suggestions available in the selected range.")
        else:
            s_left, s_right = st.columns([1,1])
            with s_left:
                pv = sd.pivot_table(index="Portfolio", columns="sent_label",
                                    values="Suggestions", aggfunc="count", fill_value=0)
                order = [c for c in ["negative","neutral","positive"] if c in pv.columns]
                pv = pv[order]
                fig2, ax2 = plt.subplots()
                x = np.arange(len(pv.index)); bottom = np.zeros(len(x))
                cmap = {"negative": _SENT_NEG, "neutral": _SENT_NEU, "positive": _SENT_POS}
                for col in pv.columns:
                    ax2.bar(x, pv[col].values, bottom=bottom,
                            label=col.capitalize(), color=cmap.get(col, _SENT_NEU))
                    bottom += pv[col].values
                ax2.set_xticks(x); ax2.set_xticklabels(pv.index, rotation=90, color=_DARK_GREY)
                for sp in ["top","right","left"]: ax2.spines[sp].set_visible(False)
                ax2.spines["bottom"].set_color(_SOFT_GREY)
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

        fig_mom = _fig_mom_nps_pos(nps, sd if "sent_label" in sd.columns else pd.DataFrame())
        if fig_mom is not None:
            st.pyplot(fig_mom, use_container_width=True)

    # -------- Tab 3 (UPDATED; robust + FPA from cases) --------
    with tab3:
        st.markdown("#### Combined KPIs — NPS, FPA%, Complaints/1000")
        try:
            base = nps[["Portfolio","_month","NPS%"]].rename(columns={"NPS%":"NPS"}).copy()

            # Pull normalized tables
            cases_df = store.get("cases", pd.DataFrame()).copy()
            comp_df  = store.get("complaints", pd.DataFrame()).copy()

            # ---- FPA% from cases (Review Result == Pass) ----
            def _fpa_from_cases(cases: pd.DataFrame) -> pd.DataFrame:
                if cases is None or cases.empty:
                    return pd.DataFrame(columns=["portfolio","_month","FPA%","Total_Cases_Complete"])
                c = cases.copy()
                # Ensure canonical keys
                if "portfolio" not in c.columns:
                    pcol = _find_col(c, ["portfolio"]) or "portfolio"
                    c["portfolio"] = c[pcol] if pcol in c.columns else "Unknown"
                if "_month" not in c.columns:
                    dcol = _find_col(c, ["create date","created date","create_date","date"])
                    c["_month"] = pd.to_datetime(c[dcol], errors="coerce", dayfirst=True).dt.to_period("M") if dcol else pd.NaT

                pass_col = None
                for cand in ["review result","review_result","result","qa result","qa_result","outcome"]:
                    if cand in c.columns:
                        pass_col = cand
                        break
                if pass_col is None:
                    return pd.DataFrame(columns=["portfolio","_month","FPA%","Total_Cases_Complete"])

                pass_vals = {"pass","passed","p","true","1","yes","y"}
                c["is_pass"] = c[pass_col].astype(str).str.strip().str.lower().isin(pass_vals)

                id_col = "id" if "id" in c.columns else None
                if id_col:
                    g = (c.dropna(subset=["_month"])
                           .groupby(["portfolio","_month"], dropna=False)
                           .agg(Total_Cases_Complete=(id_col, "nunique"),
                                Pass=("is_pass", "sum"))
                           .reset_index())
                else:
                    g = (c.dropna(subset=["_month"])
                           .groupby(["portfolio","_month"], dropna=False)
                           .agg(Total_Cases_Complete=("is_pass","size"),
                                Pass=("is_pass","sum"))
                           .reset_index())

                g["FPA%"] = (g["Pass"] * 100.0 / g["Total_Cases_Complete"].replace(0, np.nan)).round(1)
                g["FPA%"] = g["FPA%"].fillna(0.0)
                return g[["portfolio","_month","FPA%","Total_Cases_Complete"]]

            fpa_agg = _fpa_from_cases(cases_df)
            if not fpa_agg.empty:
                fpa_agg["Portfolio"] = fpa_agg["portfolio"].map(_norm_portfolio)
                fpa_agg = fpa_agg.drop(columns=["portfolio"])
            else:
                fpa_agg = pd.DataFrame(columns=["Portfolio","_month","FPA%","Total_Cases_Complete"])

            # ---- Complaints per month ----
            if comp_df is not None and not comp_df.empty:
                if "portfolio" not in comp_df.columns:
                    p_comp = _find_col(comp_df, ["portfolio"]) or "portfolio"
                    comp_df["portfolio"] = comp_df[p_comp] if p_comp in comp_df.columns else "Unknown"
                if "_month" not in comp_df.columns:
                    d_comp = _find_col(comp_df, [
                        "date complaint received - dd/mm/yy","date complaint received",
                        "received_date","received date","date","report date"
                    ])
                    comp_df["_month"] = pd.to_datetime(comp_df[d_comp], errors="coerce", dayfirst=True).dt.to_period("M") if d_comp else pd.NaT
                comp_agg = (comp_df.dropna(subset=["_month"])
                              .groupby(["portfolio","_month"], dropna=False)
                              .size().rename("Total Complaints").reset_index())
                comp_agg["Portfolio"] = comp_agg["portfolio"].map(_norm_portfolio)
                comp_agg = comp_agg.drop(columns=["portfolio"])
            else:
                comp_agg = pd.DataFrame(columns=["Portfolio","_month","Total Complaints"])

            # ---- Completes from cases (for /1000) ----
            if cases_df is not None and not cases_df.empty:
                if "portfolio" not in cases_df.columns:
                    p_cases = _find_col(cases_df, ["portfolio"]) or "portfolio"
                    cases_df["portfolio"] = cases_df[p_cases] if p_cases in cases_df.columns else "Unknown"
                if "_month" not in cases_df.columns:
                    d_cases = _find_col(cases_df, ["create date","created date","create_date","date"])
                    cases_df["_month"] = pd.to_datetime(cases_df[d_cases], errors="coerce", dayfirst=True).dt.to_period("M") if d_cases else pd.NaT
                id_cases = "id" if "id" in cases_df.columns else None
                if id_cases:
                    completes = (cases_df.dropna(subset=["_month"])
                                   .groupby(["portfolio","_month"], dropna=False)[id_cases]
                                   .nunique().rename("Total Cases Complete").reset_index())
                else:
                    completes = (cases_df.dropna(subset=["_month"])
                                   .groupby(["portfolio","_month"], dropna=False)
                                   .size().rename("Total Cases Complete").reset_index())
                completes["Portfolio"] = completes["portfolio"].map(_norm_portfolio)
                completes = completes.drop(columns=["portfolio"])
            else:
                completes = pd.DataFrame(columns=["Portfolio","_month","Total Cases Complete"])

            # ---- Merge panel & derive KPI ----
            combined = base.merge(fpa_agg, on=["Portfolio","_month"], how="left")
            combined = combined.merge(completes, on=["Portfolio","_month"], how="left")
            combined = combined.merge(comp_agg, on=["Portfolio","_month"], how="left")
            combined["Complaints/1000"] = (
                pd.to_numeric(combined["Total Complaints"], errors="coerce") /
                pd.to_numeric(combined["Total Cases Complete"], errors="coerce")
            ) * 1000.0

            # Filters
            if sel_port != "(All)":
                combined = combined[combined["Portfolio"] == sel_port]
            if start is not None:
                combined = combined[combined["_month"] >= start]
            if end is not None:
                combined = combined[combined["_month"] <= end]

            # Present
            view = combined.rename(columns={"_month":"Month"})
            for c in ["NPS","FPA%","Complaints/1000"]:
                if c in view.columns:
                    view[c] = pd.to_numeric(view[c], errors="coerce").round(1)
            if "Total Complaints" in view.columns:
                view["Total Complaints"] = pd.to_numeric(view["Total Complaints"], errors="coerce").astype("Int64")
            if "Total Cases Complete" in view.columns:
                view["Total Cases Complete"] = pd.to_numeric(view["Total Cases Complete"], errors="coerce").astype("Int64")

            cols = [c for c in ["Portfolio","Month","NPS","FPA%","Complaints/1000","Total Complaints","Total Cases Complete"]
                    if c in view.columns]
            st.dataframe(view[cols].sort_values(["Portfolio","Month"]), use_container_width=True)

        except Exception as e:
            st.error("Correlation tab failed — showing details so we can fix quickly.")
            st.exception(e)

    return ("NPS by Portfolio", "Surveys (Sheet 1)"), detail_df
