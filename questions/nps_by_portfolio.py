# questions/nps_by_portfolio.py
from __future__ import annotations

from typing import Dict, Any, Tuple, Optional
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt


def _find_col(df: pd.DataFrame, candidates) -> Optional[str]:
    cols = {str(c).strip().lower(): c for c in df.columns}
    for c in candidates:
        if c in cols:
            return cols[c]
    return None


def _prepare_surveys(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Normalizes the surveys dataframe:
      - Month_received -> month period
      - portfolio -> proper-cased string
      - NPS -> numeric score 0-10 or textual class mapped to score buckets
    """
    if df_raw is None or df_raw.empty:
        return pd.DataFrame()

    df = df_raw.copy()
    df.columns = [str(c).strip() for c in df.columns]
    lower_map = {c.lower(): c for c in df.columns}

    # ---- portfolio ----
    portfolio_col = _find_col(df, ["portfolio"])
    if portfolio_col:
        df["portfolio"] = df[portfolio_col].astype(str).str.strip().str.title()
    else:
        df["portfolio"] = "Unknown"

    # ---- Month_received -> _month ----
    month_col = _find_col(df, ["month_received", "month received", "received_month", "month"])
    if month_col:
        parsed = pd.to_datetime(df[month_col], errors="coerce", dayfirst=True, infer_datetime_format=True)
        needs_fill = parsed.isna()
        if needs_fill.any():
            mm = df.loc[needs_fill, month_col].astype(str).str.strip().str[:3].str.title()
            parsed.loc[needs_fill] = pd.to_datetime(mm + " 1 2025", errors="coerce", format="%b %d %Y")
        df["_month"] = parsed.dt.to_period("M")
        df["date"] = parsed
    else:
        df["_month"] = pd.NaT
        df["date"] = pd.NaT

    # ---- NPS score / label ----
    nps_col = _find_col(df, ["nps", "nps score", "nps_score", "nps (0-10)", "nps_score_0_10"])
    if not nps_col:
        nps_col = _find_col(df, ["score", "rating"])

    score = pd.to_numeric(df[nps_col], errors="coerce") if nps_col else pd.Series([np.nan] * len(df))
    if score.isna().mean() > 0.6:
        lbl_col = nps_col or _find_col(df, ["nps_label", "category", "type"])
        labels = df[lbl_col].astype(str).str.strip().str.lower() if lbl_col else pd.Series([""] * len(df))
        cat = pd.Series(np.where(labels.str.contains("promot"), "promoter",
                         np.where(labels.str.contains("passiv"), "passive",
                         np.where(labels.str.contains("detract"), "detractor", "unknown"))))
    else:
        cat = pd.Series(
            np.where(score >= 9, "promoter",
            np.where(score >= 7, "passive",
            np.where(score >= 0, "detractor", "unknown"))),
            index=score.index
        )

    df["nps_bucket"] = cat
    return df


def _aggregate_nps(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      by_month_portfolio: (portfolio, _month) with NPS%, Promoters, Passives, Detractors, Total
      latest_pivot: kept for internal use if needed (not rendered)
    """
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()

    df = df[df["_month"].notna()].copy()

    grp = df.groupby(["portfolio", "_month"], dropna=False)["nps_bucket"].value_counts().unstack(fill_value=0)
    for col in ["promoter", "passive", "detractor", "unknown"]:
        if col not in grp.columns:
            grp[col] = 0

    grp["Total"] = grp[["promoter", "passive", "detractor", "unknown"]].sum(axis=1).replace(0, np.nan)
    grp["NPS%"] = ((grp["promoter"] - grp["detractor"]) / grp["Total"]) * 100.0

    by_month_portfolio = grp.reset_index().sort_values(["portfolio", "_month"])

    last_m = by_month_portfolio["_month"].max()
    latest = by_month_portfolio[by_month_portfolio["_month"] == last_m].copy()
    latest_pivot = latest.pivot_table(index="portfolio", values="NPS%", aggfunc="mean").sort_values("NPS%", ascending=False)

    return by_month_portfolio, latest_pivot


def _sidebar_filters(df: pd.DataFrame) -> Dict[str, Any]:
    with st.sidebar:
        st.header("Filters")
        ports = ["(All)"] + sorted(df["portfolio"].dropna().unique().tolist())
        port = st.selectbox("Portfolio", ports, index=0)

        months = df["_month"].dropna().sort_values().unique().tolist()
        if months:
            start = st.selectbox("From month", months, index=0)
            end = st.selectbox("To month", months, index=len(months) - 1)
        else:
            start = end = None
    return {"portfolio": port, "start": start, "end": end}


def run(store: Dict[str, Any], params: Dict[str, Any], user_text: Optional[str] = None):
    """
    Entry point required by app.py.
    Returns:
      (title, subtitle), dataframe
    """
    surveys = store.get("surveys", pd.DataFrame())
    if surveys is None or surveys.empty:
        return ("NPS by Portfolio", "No surveys data found. Put files under data/surveys/"), pd.DataFrame()

    df = _prepare_surveys(surveys)
    if df.empty or df["_month"].isna().all():
        return ("NPS by Portfolio", "Could not parse Month_received; please check column name/values."), pd.DataFrame()

    # Sidebar filters
    flt = _sidebar_filters(df)

    by_month_portfolio, _latest_pivot = _aggregate_nps(df)

    # Apply filters
    if flt["portfolio"] and flt["portfolio"] != "(All)":
        by_month_portfolio = by_month_portfolio[by_month_portfolio["portfolio"] == flt["portfolio"]]
    if flt["start"] is not None:
        by_month_portfolio = by_month_portfolio[by_month_portfolio["_month"] >= flt["start"]]
    if flt["end"] is not None:
        by_month_portfolio = by_month_portfolio[by_month_portfolio["_month"] <= flt["end"]]

    # ----- Headline KPI for the selected slice -----
    if not by_month_portfolio.empty:
        k = by_month_portfolio.copy()
        num = (k["NPS%"] * k["Total"]).sum(skipna=True)
        den = k["Total"].sum(skipna=True)
        overall_nps = float(num / den) if den and den > 0 else np.nan
    else:
        overall_nps = np.nan

    st.markdown(f"### Overall NPS (selected range): **{overall_nps:.1f}**")

    # ===== Layout: Chart | Table (side by side) =====
    left, right = st.columns([1, 1])

    # ----- Left: Styled NPS Trend Chart -----
    with left:
        if not by_month_portfolio.empty:
            # Pastel color cycle
            pastel = ["#A3C4F3", "#CDE7BE", "#F6C1C1", "#FFD6A5", "#BDB2FF", "#FFAFCC", "#BEE1E6", "#E2ECE9"]
            fig, ax = plt.subplots()
            for i, (p, g) in enumerate(by_month_portfolio.groupby("portfolio")):
                g = g.sort_values("_month")
                ax.plot(
                    g["_month"].astype(str),
                    g["NPS%"],
                    marker="o",
                    linewidth=2.0,
                    markersize=4.5,
                    label=p,
                    color=pastel[i % len(pastel)],
                )

            # Styling per request
            # No border: hide all spines except bottom (which we'll set grey)
            for spine in ["top", "right", "left"]:
                ax.spines[spine].set_visible(False)
            ax.spines["bottom"].set_color("#D3D3D3")  # soft grey x-axis
            ax.tick_params(axis="x", colors="#6E6E6E")
            # No y-axis
            ax.get_yaxis().set_visible(False)
            # No gridlines
            ax.grid(False)

            ax.set_xlabel("")  # cleaner
            ax.set_title("NPS Trend", fontsize=12, pad=6)
            ax.legend(loc="best", fontsize=8, frameon=False)

            st.pyplot(fig, use_container_width=True)

    # ----- Right: Detail Table -----
    with right:
        show_cols = ["portfolio", "_month", "NPS%", "promoter", "passive", "detractor", "unknown", "Total"]
        detail = by_month_portfolio[show_cols].rename(columns={"_month": "Month", "NPS%": "NPS"})
        # Lightweight formatting
        if not detail.empty:
            detail = detail.copy()
            # Round NPS to 1 decimal
            detail["NPS"] = detail["NPS"].round(1)
        st.markdown("#### Detail (by Portfolio × Month)")
        st.dataframe(detail, use_container_width=True)

    # Return the detail DF to the host app
    return ("NPS by Portfolio", "Reads surveys/ (Sheet 1), buckets by Month_received, and computes NPS."), detail
