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


# ----------------- ops/service (SLA) -----------------
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


# ----------------- complaints (optional) -----------------
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
    """Always returns ((title, subtitle), dataframe) and never raises to the host."""
    try:
        # Load
        surveys = store.get("surveys", pd.DataFrame())
        ops_raw = store.get("ops", store.get("fpa", pd.DataFrame()))
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

        # overall NPS (weighted)
        if not nps.empty:
            weights = nps[["promoter","passive","detractor","unknown"]].sum(axis=1)
            overall_nps = float((nps["NPS%"] * weights).sum() / weights.sum()) if weights.sum() else np.nan
        else:
            overall_nps = np.nan
        st.markdown(f"### Overall NPS (selected range): **{overall_nps:.1f}**")

        # Tabs
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

        # ---- Tab 2: sentiments (UNCHANGED) ----
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

        # ---- Tab 3: correlation (UPDATED ONLY) ----
        combined_df = pd.DataFrame()
        with tab3:
            # Build panel
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

            # safety: ensure NPS column exists
            if "NPS" not in combined_df.columns and "NPS%" in combined_df.columns:
                combined_df["NPS"] = combined_df["NPS%"]

            # compute Complaints/1000 for analysis (we won't show 'Completes' in the table)
            if "Complaints" in combined_df.columns:
                if "Completes" in combined_df.columns:
                    combined_df["Complaints_per_1000"] = (
                        pd.to_numeric(combined_df["Complaints"], errors="coerce") /
                        pd.to_numeric(combined_df["Completes"], errors="coerce")
                    ) * 1000
                else:
                    combined_df["Complaints_per_1000"] = np.nan

            # apply filters
            if sel_port != "(All)": combined_df = combined_df[combined_df["Portfolio"] == sel_port]
            if start is not None:  combined_df = combined_df[combined_df["_month"] >= start]
            if end   is not None:  combined_df = combined_df[combined_df["_month"] <= end]
            combined_df = combined_df.sort_values(["Portfolio","_month"]).reset_index(drop=True)

            # MoM deltas (guard each col)
            for c in ["NPS","NetSent%","Complaints_per_1000"]:
                if c in combined_df.columns:
                    combined_df[f"Δ{c}"] = combined_df.groupby("Portfolio")[c].transform(lambda s: s - s.shift(1))

            latest_m = combined_df["_month"].max() if not combined_df.empty else None
            prev_m   = (combined_df["_month"].dropna().unique()[-2] if combined_df["_month"].nunique() >= 2 else None)
            latest = combined_df[combined_df["_month"] == latest_m].copy() if latest_m is not None else pd.DataFrame()
            prev   = combined_df[combined_df["_month"] == prev_m].copy() if prev_m is not None else pd.DataFrame()

            # Top row layout
            top_left, top_right = st.columns([1,1])

            # --- Left: NPS vs Complaints (with toggle + arrows) ---
            with top_left:
                plotted_any = False
                size_by = st.radio("Bubble size", ["Suggestions","Complaints/1000"], index=0, horizontal=True, key="size_by_corr")
                show_move = st.checkbox("Show movement arrows (latest vs prior)", value=False, key="move_arrows")

                if not latest.empty and "NPS" in latest.columns and ("Complaints_per_1000" in latest.columns or "Complaints" in latest.columns):
                    # pick y
                    if "Complaints_per_1000" in latest.columns and latest["Complaints_per_1000"].notna().any():
                        y = pd.to_numeric(latest["Complaints_per_1000"], errors="coerce")
                        y_label = "Complaints per 1000"
                    else:
                        y = pd.to_numeric(latest.get("Complaints"), errors="coerce")
                        y_label = "Complaints (count)"

                    dfp = latest.copy()
                    dfp["y"] = y
                    dfp = dfp[dfp["NPS"].notna() & dfp["y"].notna()]
                    if len(dfp) >= 1:
                        figc, axc = plt.subplots()
                        if size_by == "Suggestions":
                            size = (dfp["Sugg"].fillna(0).astype(float) + 1) * 3
                        else:
                            size = (dfp["y"].fillna(0).astype(float) + 1) * 5
                        axc.scatter(dfp["NPS"], dfp["y"], s=size, c=_BUBBLE_FILL, alpha=0.75,
                                    edgecolors=_BUBBLE_EDGE, linewidths=0.6)
                        for _, r in dfp.iterrows():
                            axc.annotate(r["Portfolio"], (r["NPS"], r["y"]), fontsize=9,
                                         xytext=(3,3), textcoords="offset points", color=_DARK_GREY)
                        axc.set_xlabel("NPS %"); axc.set_ylabel(y_label)
                        _style_axes(axc)
                        axc.set_title(f"NPS vs {y_label} (Latest: {str(latest_m)})", fontsize=12, color=_DARK_BLUE)

                        # arrows
                        if show_move and not prev.empty:
                            src = prev[["Portfolio","NPS"]].copy()
                            if y_label == "Complaints per 1000":
                                src["y_prev"] = prev["Complaints_per_1000"]
                            else:
                                src["y_prev"] = prev.get("Complaints")
                            step = src.merge(dfp[["Portfolio","NPS","y"]], on="Portfolio", how="inner")
                            for _, r in step.iterrows():
                                if np.isfinite(r["NPS_x"]) and np.isfinite(r["y_prev"]) and np.isfinite(r["NPS_y"]) and np.isfinite(r["y"]):
                                    axc.annotate("", xy=(r["NPS_y"], r["y"]), xytext=(r["NPS_x"], r["y_prev"]),
                                                 arrowprops=dict(arrowstyle="->", lw=1, alpha=0.4, color=_DARK_GREY))

                        st.pyplot(figc, use_container_width=True)
                        cap = "Bubble size ∝ " + ("Suggestions (Sugg)" if size_by=="Suggestions" else "Complaints per 1000")
                        st.caption(cap)
                        plotted_any = True

                if not plotted_any:
                    st.info("Not enough data to draw the correlation chart for the latest month.")

            # --- Right: Driver model + outliers ---
            with top_right:
                drv_cols = [c for c in ["NetSent%","SLA%","Complaints_per_1000"] if c in combined_df.columns]
                dfm = combined_df.dropna(subset=["NPS"] + drv_cols).copy() if drv_cols else pd.DataFrame()
                if not dfm.empty and len(dfm) >= max(5, len(drv_cols)+2):
                    # standardize
                    Xz = (dfm[drv_cols].apply(pd.to_numeric, errors="coerce") - dfm[drv_cols].mean()) / dfm[drv_cols].std(ddof=0)
                    y  = (pd.to_numeric(dfm["NPS"], errors="coerce") - dfm["NPS"].mean()) / dfm["NPS"].std(ddof=0)
                    Xz = Xz.fillna(0.0); y = y.fillna(0.0)
                    # weights by suggestions (>=1)
                    w = pd.to_numeric(dfm.get("Sugg", 1), errors="coerce").fillna(1.0).clip(lower=1.0)
                    sw = np.sqrt(w).to_numpy()
                    Xi = np.column_stack([Xz.to_numpy(), np.ones(len(Xz))])
                    Xw = Xi * sw[:, None]
                    yw = y.to_numpy() * sw
                    try:
                        beta, _, _, _ = np.linalg.lstsq(Xw, yw, rcond=None)
                        yhat = Xi @ beta
                        # weighted R^2
                        num = np.average((y.to_numpy() - yhat)**2, weights=w)
                        den = np.average((y.to_numpy() - np.average(y.to_numpy(), weights=w))**2, weights=w)
                        r2 = float(1 - num/den) if den > 0 else np.nan
                        coefs = pd.Series(beta[:-1], index=drv_cols).sort_values(key=np.abs, ascending=False)
                        st.markdown("#### Drivers of NPS (standardized, weighted)")
                        st.dataframe(coefs.rename("Effect (β)").to_frame().style.format("{:.2f}"), use_container_width=True)
                        st.caption(f"Weighted R² ≈ {r2:.2f} — β shows relative impact on NPS (↑ positive, ↓ negative).")
                        # Outliers
                        res = y.to_numpy() - yhat
                        df_out = dfm.assign(Residual=res)
                        c1, c2 = st.columns(2)
                        with c1:
                            st.markdown("**Over-performers**")
                            st.dataframe(df_out.nlargest(3, "Residual")[["Portfolio","_month","NPS"]], use_container_width=True)
                        with c2:
                            st.markdown("**Under-performers**")
                            st.dataframe(df_out.nsmallest(3, "Residual")[["Portfolio","_month","NPS"]], use_container_width=True)
                    except Exception:
                        st.info("Driver model could not be fit on the current selection.")
                else:
                    st.info("Not enough data to fit a driver model on the current selection.")

            # Row 2: Quadrant + lag-1 + KPI table
            row2_left, row2_right = st.columns([1,1])

            with row2_left:
                if not latest.empty and "NetSent%" in latest.columns and latest["NetSent%"].notna().any() and "NPS" in latest.columns:
                    figq, axq = plt.subplots()
                    xs = pd.to_numeric(latest["NetSent%"], errors="coerce")
                    ys = pd.to_numeric(latest["NPS"], errors="coerce")
                    ok = np.isfinite(xs) & np.isfinite(ys)
                    xs, ys = xs[ok], ys[ok]
                    labs = latest.loc[ok, "Portfolio"]
                    axq.scatter(xs, ys, s=60, c=_BUBBLE_FILL, alpha=0.75, edgecolors=_BUBBLE_EDGE, linewidths=0.6)
                    mx, my = np.nanmedian(xs), np.nanmedian(ys)
                    axq.axvline(mx, color=_SOFT_GREY); axq.axhline(my, color=_SOFT_GREY)
                    for xi, yi, lab in zip(xs, ys, labs):
                        axq.annotate(lab, (xi, yi), fontsize=9, xytext=(3,3), textcoords="offset points", color=_DARK_GREY)
                    axq.set_xlabel("NetSent %"); axq.set_ylabel("NPS %")
                    _style_axes(axq); axq.set_title("Quadrant — NPS vs NetSent% (latest)", color=_DARK_BLUE)
                    st.pyplot(figq, use_container_width=True)
                    low_low = latest.loc[(latest["NetSent%"]<=mx) & (latest["NPS"]<=my), "Portfolio"].tolist()
                    hi_low  = latest.loc[(latest["NetSent%"]> mx) & (latest["NPS"]<=my), "Portfolio"].tolist()
                    st.caption(f"Low Sentiment & Low NPS: {', '.join(low_low) if low_low else '—'}  |  High Sentiment & Low NPS: {', '.join(hi_low) if hi_low else '—'}")
                else:
                    st.info("Not enough NetSent% data for the latest month to build the quadrant.")

            with row2_right:
                # lag-1 across portfolio-month panel
                def lag1_corr(df: pd.DataFrame, xcol: str, ycol: str) -> float:
                    if xcol not in df.columns or ycol not in df.columns: return np.nan
                    d = df.sort_values(["Portfolio","_month"]).copy()
                    d["y_next"] = d.groupby("Portfolio")[ycol].shift(-1)
                    z = d[[xcol,"y_next"]].dropna()
                    return float(z[xcol].corr(z["y_next"])) if len(z) >= 3 else np.nan

                lag_net = lag1_corr(combined_df, "NetSent%", "NPS")
                lag_sla = lag1_corr(combined_df, "SLA%", "NPS") if "SLA%" in combined_df.columns else np.nan
                m1, m2 = st.columns(2)
                with m1: st.metric("Lag-1: NetSent% → next-month NPS", f"{lag_net:.2f}" if pd.notna(lag_net) else "n/a")
                with m2: st.metric("Lag-1: SLA% → next-month NPS", f"{lag_sla:.2f}" if pd.notna(lag_sla) else "n/a")

                # KPI table (no 'Completes')
                view = combined_df.rename(columns={"_month":"Month"}).copy()
                for c in ["NPS","SLA%","Pos%","Neg%","NetSent%","Complaints_per_1000","ΔNPS","ΔNetSent%","ΔComplaints_per_1000"]:
                    if c in view.columns: view[c] = pd.to_numeric(view[c], errors="coerce").round(1)
                cols = [c for c in [
                    "Portfolio","Month","NPS","ΔNPS","NetSent%","ΔNetSent%","Pos%","Neg%","Sugg",
                    "Complaints","Complaints_per_1000","ΔComplaints_per_1000","SLA%"
                ] if c in view.columns]
                st.markdown("#### Combined KPIs (Portfolio × Month)")
                st.dataframe(view[cols].sort_values(["Month","Portfolio"]), use_container_width=True)

            # quick Pearson captions (guarded)
            try:
                c1, c2 = st.columns(2)
                with c1:
                    corr = combined_df[["NPS","SLA%"]].dropna() if "SLA%" in combined_df.columns else pd.DataFrame()
                    st.caption(f"Pearson(NPS, SLA%): **{float(corr.corr().iloc[0,1]):.2f}**" if len(corr)>=2 else "Pearson(NPS, SLA%): not enough data")
                with c2:
                    corr2 = combined_df[["NPS","NetSent%"]].dropna() if "NetSent%" in combined_df.columns else pd.DataFrame()
                    st.caption(f"Pearson(NPS, Net Sentiment): **{float(corr2.corr().iloc[0,1]):.2f}**" if len(corr2)>=2 else "Pearson(NPS, Net Sentiment): not enough data")
            except Exception:
                pass

        # final return (hide duplicated bottom table)
        host_df = pd.DataFrame()
        return ("NPS by Portfolio", "Surveys (Sheet 1) with Sentiments and SLA/Complaints correlation"), host_df

    except Exception as e:
        import traceback
        st.error(f"NPS module error: {e}")
        st.code(traceback.format_exc())
        safe = pd.DataFrame([{"error": str(e)}])
        return ("NPS by Portfolio", "Recovered from error"), safe
