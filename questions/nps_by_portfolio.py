# questions/nps_by_portfolio.py
from __future__ import annotations

from typing import Dict, Any, Optional, Iterable
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
    """Soft-grey axes, dark-grey fonts."""
    ax.tick_params(colors=_DARK_GREY, labelcolor=_DARK_GREY)
    ax.xaxis.label.set_color(_DARK_GREY)
    ax.yaxis.label.set_color(_DARK_GREY)
    for sp in ("top","right"):
        ax.spines[sp].set_visible(False)
    for sp in ("bottom","left"):
        ax.spines[sp].set_color(_SOFT_GREY)
        ax.spines[sp].set_linewidth(1.25)
    ax.grid(False)

# --- (used by Sentiments tab) ---
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

# ----------------- ops/service (legacy helpers kept as-is) -----------------
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

# ----------------- Correlation helpers (FPA loader, cases, complaints) -----------------
def _find_fpa_workbook() -> Optional[Path]:
    roots = [Path("data/first_pass_accuracy"), Path("first_pass_accuracy"), Path("data/first_pass_accuracy/")]
    patterns = ["FirstPassAccuracy*.xls*", "*FirstPassAccuracy*.xls*"]
    for root in roots:
        if not root.exists():
            continue
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

def _load_fpa_from_store_or_disk(store: Dict[str, Any]) -> pd.DataFrame:
    """
    Return a normalized FPA frame with columns: Portfolio, _month, FPA%
    Priority: store['fpa'] → workbook on disk.
    """
    fpa_raw = store.get("fpa", pd.DataFrame())
    if fpa_raw is None or fpa_raw.empty:
        # try disk (same strategy as FPA question)
        p = _find_fpa_workbook()
        if p:
            try:
                fpa_raw = _read_excel_any(p)
            except Exception:
                fpa_raw = pd.DataFrame()

    if fpa_raw is None or fpa_raw.empty:
        return pd.DataFrame(columns=["Portfolio","_month","FPA%"])

    fpa = fpa_raw.copy()
    p_fpa = _find_col(fpa, ["portfolio"])
    fpa["Portfolio"] = fpa[p_fpa].map(_norm_portfolio) if p_fpa else "Unknown"

    d_fpa = _find_col(fpa, ["activity date","activity_date","activitydate","date"])
    if d_fpa is None:
        d_fpa = "date" if "date" in fpa.columns else None
    fpa["_month"] = pd.to_datetime(fpa[d_fpa], errors="coerce", dayfirst=True).dt.to_period("M") if d_fpa else pd.NaT

    r_fpa = _find_col(fpa, ["review result","review_result","result","qa result","fpa result"])
    res = fpa[r_fpa].astype(str).str.strip().str.lower() if r_fpa else ""
    fpa["_pass"] = res.str.startswith("pass").astype(int)

    g = (fpa.dropna(subset=["_month"])
           .groupby(["Portfolio","_month"])["_pass"]
           .agg(passed="sum", total="count")
           .reset_index())
    g["FPA%"] = (g["passed"] * 100.0 / g["total"].replace(0, np.nan))
    return g[["Portfolio","_month","FPA%"]]

def _cases_monthly(store: Dict[str, Any]) -> pd.DataFrame:
    cs = store.get("cases", pd.DataFrame())
    if cs is None or cs.empty:
        return pd.DataFrame(columns=["Portfolio","_month","Total Cases Complete"])
    df = cs.copy()
    p = _find_col(df, ["portfolio"]); df["Portfolio"] = df[p].map(_norm_portfolio) if p else "Unknown"
    d = _find_col(df, ["create date","created date","create_date","start date","report date","date"])
    df["_month"] = pd.to_datetime(df[d], errors="coerce", dayfirst=True).dt.to_period("M") if d else pd.NaT
    id_col = _find_col(df, ["case id","case_id","unique identifier","unique id","unique identifier.","id","case reference","case"])
    if id_col:
        out = (df.dropna(subset=["_month"])
                 .groupby(["Portfolio","_month"])[id_col].nunique()
                 .reset_index()
                 .rename(columns={id_col: "Total Cases Complete"}))
    else:
        out = (df.dropna(subset=["_month"])
                 .groupby(["Portfolio","_month"])
                 .size().to_frame("Total Cases Complete")
                 .reset_index())
    return out

def _complaints_monthly(store: Dict[str, Any]) -> pd.DataFrame:
    comp = store.get("complaints", pd.DataFrame())
    if comp is None or comp.empty:
        return pd.DataFrame(columns=["Portfolio","_month","Total Complaints"])
    df = comp.copy()
    p = _find_col(df, ["portfolio"]); df["Portfolio"] = df[p].map(_norm_portfolio) if p else "Unknown"
    d = _find_col(df, ["date complaint received - dd/mm/yy","date complaint received","complaint date",
                       "received date","received_date","date","month"])
    if d and d.lower()=="month":
        # month text like 'June' (assume 2025)
        m = df[d].astype(str).str.strip().str[:3].str.title()
        df["_month"] = pd.to_datetime(m + " 2025", format="%b %Y", errors="coerce").dt.to_period("M")
    else:
        df["_month"] = pd.to_datetime(df[d], errors="coerce", dayfirst=True).dt.to_period("M") if d else pd.NaT
    out = (df.dropna(subset=["_month"])
             .groupby(["Portfolio","_month"])
             .size().to_frame("Total Complaints")
             .reset_index())
    return out

# ----------------- UI entry -----------------
def run(store: Dict[str, Any], params: Dict[str, Any], user_text: Optional[str] = None):
    """Always returns ((title, subtitle), dataframe) and never raises to the host."""
    df_out = pd.DataFrame()

    try:
        # Load surveys
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
            end = st.selectbox("To month", months, index=len(months)-1) if months else None

        # NPS aggregates
        nps = _aggregate_nps(s)
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

        # Tabs
        tab1, tab2, tab3 = st.tabs(["Overview", "Sentiments", "NPS Correlation"])

        # ---- Tab 1: overview (unchanged) ----
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

        # ---- Tab 2: sentiments (unchanged) ----
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

        # ---- Tab 3: NPS Correlation (UPDATED: correlation panel + MoM deltas) ----
        with tab3:
            st.markdown("### Combined KPIs — NPS, FPA%, Complaints/1000")

            # Base NPS (includes component counts)
            base = nps[["Portfolio","_month","NPS%","promoter","passive","detractor","unknown","Total"]]\
                      .rename(columns={"NPS%":"NPS"}).copy()

            # FPA (from store OR disk)
            fpa_monthly = _load_fpa_from_store_or_disk(store)

            # Cases & Complaints
            cases_monthly = _cases_monthly(store)
            comp_monthly  = _complaints_monthly(store)

            # Month-level sentiments by portfolio
            sd_all = _sentiments(s)
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

            # Merge panel
            combined = base.copy()
            combined = combined.merge(fpa_monthly, on=["Portfolio","_month"], how="outer")
            combined = combined.merge(cases_monthly, on=["Portfolio","_month"], how="outer")
            combined = combined.merge(comp_monthly, on=["Portfolio","_month"], how="outer")
            combined = combined.merge(sent_m, on=["Portfolio","_month"], how="left")

            # KPIs
            combined["Complaints/1000"] = (
                pd.to_numeric(combined.get("Total Complaints"), errors="coerce") /
                pd.to_numeric(combined.get("Total Cases Complete"), errors="coerce")
            ) * 1000.0
            combined["Detractors%"] = (
                pd.to_numeric(combined.get("detractor"), errors="coerce") /
                pd.to_numeric(combined.get("Total"), errors="coerce")
            ) * 100.0

            # Apply filters
            if sel_port != "(All)":
                combined = combined[combined["Portfolio"] == sel_port]
            if start is not None:
                combined = combined[combined["_month"] >= start]
            if end is not None:
                combined = combined[combined["_month"] <= end]

            # Pretty view & formatting
            view = combined.rename(columns={"_month":"Month"})
            for c in ["NPS","FPA%","Complaints/1000","Detractors%","Pos%","Neg%","NetSent%"]:
                if c in view.columns:
                    view[c] = pd.to_numeric(view[c], errors="coerce").round(1)

            # --- layout: chart (left) + table (right) ---
            left, right = st.columns([1,1])

            with left:
                plot_df = view.dropna(subset=["NPS","Complaints/1000"]).copy()
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
                ttl_month = str(plot_df["Month"].max()) if "Month" in plot_df.columns and not plot_df["Month"].isna().all() else ""
                ax_sc.set_title(f"NPS vs Complaints/1000 {f'({ttl_month})' if ttl_month else ''}", color=_DARK_BLUE, pad=6)
                st.pyplot(fig_sc, use_container_width=True)
                st.caption("Bubble size ∝ FPA% (larger = higher FPA).")

            with right:
                cols = [c for c in ["Portfolio","Month","Complaints/1000","FPA%","NPS","Detractors%","Pos%","Neg%","NetSent%"]
                        if c in view.columns]
                st.dataframe(view[cols].sort_values(["Portfolio","Month"]), use_container_width=True)

            # ---- Correlation snapshot (selected range) ----
            st.markdown("#### Correlation snapshot (selected range)")
            corr_cols = st.columns(4)

            def _corr_pair(df: pd.DataFrame, x: str, y: str) -> str:
                if x not in df or y not in df: return "n/a"
                d = df[[x,y]].dropna()
                if len(d) < 3: return "n/a"
                r = d[x].corr(d[y])
                if pd.isna(r): return "n/a"
                return f"r = {r:+.2f}"

            with corr_cols[0]:
                st.metric("FPA% vs Complaints/1000", _corr_pair(view, "FPA%", "Complaints/1000"))
            with corr_cols[1]:
                st.metric("FPA% vs NPS", _corr_pair(view, "FPA%", "NPS"))
            with corr_cols[2]:
                st.metric("Detractors% vs NPS", _corr_pair(view, "Detractors%", "NPS"))
            with corr_cols[3]:
                st.metric("NetSent% vs NPS", _corr_pair(view, "NetSent%", "NPS"))

            # ---- MoM deltas (latest vs previous) ----
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

    # Always return a tuple so the app runner never throws
    return ("NPS by Portfolio", "Surveys (Sheet 1) with Sentiments and SLA/Complaints correlation"), df_out
