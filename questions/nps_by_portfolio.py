# questions/nps_by_portfolio.py
from __future__ import annotations

from typing import Dict, Any, Tuple, Optional
import re
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt


# ---------- helpers ----------
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
      - NPS -> bucket (promoter/passive/detractor)
      - Suggestions -> normalized text (if present)
    """
    if df_raw is None or df_raw.empty:
        return pd.DataFrame()

    df = df_raw.copy()
    df.columns = [str(c).strip() for c in df.columns]

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

    # ---- Suggestions (optional) ----
    sug_col = _find_col(df, ["suggestions", "suggestion", "comments", "comment", "feedback"])
    if sug_col:
        df["suggestions"] = df[sug_col].astype(str).str.strip()
        df.loc[df["suggestions"].str.len() == 0, "suggestions"] = np.nan
        df.loc[df["suggestions"].str.lower().isin(["nan", "none", "na", "null"]), "suggestions"] = np.nan
    else:
        df["suggestions"] = np.nan

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


# ---------- Sentiment & Category labelling for Suggestions ----------
_POS_WORDS = {
    "good","great","excellent","amazing","helpful","fast","quick","responsive","easy","clear",
    "friendly","polite","supportive","smooth","love","efficient","prompt","awesome","happy"
}
_NEG_WORDS = {
    "bad","poor","terrible","slow","delay","delayed","waiting","confusing","unclear","hard",
    "rude","unhelpful","expensive","issue","problem","bug","error","crash","difficult","worst"
}

_CATEGORY_PATTERNS = [
    ("Turnaround Time / Speed", [r"\bslow\b", r"\bdelay", r"waiting", r"\bresponse time\b", r"\bfaster\b", r"\bspeed", r"\bturnaround\b", r"\bSLA\b"]),
    ("Communication & Support", [r"call back", r"communicat", r"follow[- ]?up", r"support", r"contact", r"update", r"rude", r"behaviou?r"]),
    ("Charges & Billing", [r"charge", r"fee", r"billing", r"invoice", r"refund", r"payment", r"expensive", r"cost"]),
    ("Process & Policy", [r"process", r"policy", r"approval", r"paperwork", r"form", r"step", r"escalat"]),
    ("App / Portal / Tech", [r"\bapp\b", r"portal", r"website", r"login", r"error", r"bug", r"crash", r"ui\b", r"\bux\b", r"technical"]),
    ("Product / Features", [r"feature", r"option", r"coverage", r"benefit", r"plan", r"pricing", r"add", r"improv"]),
    ("Staff & Behavior", [r"agent", r"executive", r"representative", r"staff", r"polite", r"rude", r"attitude", r"helpful"]),
    ("Clarity & Information", [r"clear", r"explain", r"information", r"transparen", r"confus", r"guid", r" educate "]),
]

def _get_vader():
    try:
        from nltk.sentiment import SentimentIntensityAnalyzer  # type: ignore
        return SentimentIntensityAnalyzer()
    except Exception:
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer  # type: ignore
            return SentimentIntensityAnalyzer()
        except Exception:
            return None

_SIA = _get_vader()

def _fallback_sentiment(text: str) -> float:
    toks = re.findall(r"\b\w+\b", text.lower())
    if not toks:
        return 0.0
    score = sum(1 for t in toks if t in _POS_WORDS) - sum(1 for t in toks if t in _NEG_WORDS)
    return max(-1.0, min(1.0, score / max(len(toks), 4)))

def _sentiment_label(text: str) -> Tuple[str, float]:
    if not isinstance(text, str) or not text.strip():
        return "neutral", 0.0
    if _SIA is not None:
        try:
            comp = float(_SIA.polarity_scores(text).get("compound", 0.0))
        except Exception:
            comp = _fallback_sentiment(text)
    else:
        comp = _fallback_sentiment(text)
    if comp >= 0.05:
        return "positive", comp
    if comp <= -0.05:
        return "negative", comp
    return "neutral", comp

_CAT_REGEX = [(name, [re.compile(pat, re.IGNORECASE) for pat in pats]) for name, pats in _CATEGORY_PATTERNS]

def _categorize(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        return "Other"
    for name, regs in _CAT_REGEX:
        for r in regs:
            if r.search(text):
                return name
    return "Other"


def _analyze_suggestions(df: pd.DataFrame) -> pd.DataFrame:
    sug = df[df["suggestions"].notna()].copy()
    if sug.empty:
        return sug
    labs, scores, cats = [], [], []
    for s in sug["suggestions"].astype(str):
        lab, sc = _sentiment_label(s)
        labs.append(lab); scores.append(sc); cats.append(_categorize(s))
    sug["sentiment"] = labs
    sug["sentiment_score"] = scores
    sug["category"] = cats
    return sug


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


# ---------- UI entry ----------
def run(store: Dict[str, Any], params: Dict[str, Any], user_text: Optional[str] = None):
    """
    Entry point required by app.py.
    Returns:
      (title, subtitle), dataframe  # we will return an EMPTY df to avoid duplicate 3rd-row table
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

    # Apply filters to NPS view
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
            # Styling (no border, no y-axis, soft grey x-axis, no grid)
            for spine in ["top", "right", "left"]:
                ax.spines[spine].set_visible(False)
            ax.spines["bottom"].set_color("#D3D3D3")
            ax.tick_params(axis="x", colors="#6E6E6E")
            ax.get_yaxis().set_visible(False)
            ax.grid(False)
            ax.set_xlabel("")
            ax.set_title("NPS Trend", fontsize=12, pad=6)
            ax.legend(loc="best", fontsize=8, frameon=False)
            st.pyplot(fig, use_container_width=True)

    # ----- Right: Detail Table (This is the ONLY table we render) -----
    with right:
        show_cols = ["portfolio", "_month", "NPS%", "promoter", "passive", "detractor", "unknown", "Total"]
        detail = by_month_portfolio[show_cols].rename(columns={"_month": "Month", "NPS%": "NPS"})
        if not detail.empty:
            detail = detail.copy()
            detail["NPS"] = detail["NPS"].round(1)
        st.markdown("#### Detail (by Portfolio × Month)")
        st.dataframe(detail, use_container_width=True)

    # ===== Suggestions Analysis =====
    st.markdown("---")
    st.markdown("### Suggestions Analysis")

    sug = _analyze_suggestions(df)
    if not sug.empty:
        if flt["portfolio"] and flt["portfolio"] != "(All)":
            sug = sug[sug["portfolio"] == flt["portfolio"]]
        if flt["start"] is not None:
            sug = sug[sug["_month"] >= flt["start"]]
        if flt["end"] is not None:
            sug = sug[sug["_month"] <= flt["end"]]

    if sug.empty:
        st.info("No suggestions available in the selected range.")
    else:
        # Layout: sentiment chart | category table
        s_left, s_right = st.columns([1, 1])

        with s_left:
            # Stacked bar: sentiment counts by portfolio (soft pastel colors, vertical x labels)
            sent_pivot = sug.pivot_table(index="portfolio", columns="sentiment", values="suggestions", aggfunc="count", fill_value=0)
            # consistent column order
            order = [c for c in ["negative", "neutral", "positive"] if c in sent_pivot.columns]
            sent_pivot = sent_pivot[order]
            colors = {"negative": "#F6C1C1", "neutral": "#E2ECE9", "positive": "#CDE7BE"}  # soft pastels

            fig2, ax2 = plt.subplots()
            x = np.arange(len(sent_pivot.index))
            bottom = np.zeros(len(x))
            for col in sent_pivot.columns:
                ax2.bar(x, sent_pivot[col].values, bottom=bottom, label=col.capitalize(), color=colors.get(col))
                bottom += sent_pivot[col].values

            ax2.set_xticks(x)
            ax2.set_xticklabels(sent_pivot.index, rotation=90)  # vertical labels

            # style (soft grey x-axis, no y-axis/grid, no border)
            for spine in ["top", "right", "left"]:
                ax2.spines[spine].set_visible(False)
            ax2.spines["bottom"].set_color("#D3D3D3")
            ax2.get_yaxis().set_visible(False)
            ax2.grid(False)
            ax2.set_title("Suggestions Sentiment by Portfolio", fontsize=12, pad=6)
            ax2.legend(loc="best", fontsize=8, frameon=False)
            st.pyplot(fig2, use_container_width=True)

        with s_right:
            total_rows = len(sug)
            cat_summary = (
                sug.groupby("category")
                   .agg(Count=("category","size"),
                        Pos=("sentiment", lambda s: (s=="positive").sum()),
                        Neg=("sentiment", lambda s: (s=="negative").sum()))
                   .sort_values("Count", ascending=False)
            )
            if total_rows > 0:
                cat_summary["%"] = (cat_summary["Count"] / total_rows * 100).round(1)
                cat_summary["Positive %"] = (cat_summary["Pos"] / cat_summary["Count"] * 100).round(1)
                cat_summary["Negative %"] = (cat_summary["Neg"] / cat_summary["Count"] * 100).round(1)
            examples = (
                sug.groupby("category")["suggestions"]
                   .apply(lambda s: (s.dropna().iloc[0] if len(s.dropna()) else ""))
            )
            cat_summary = cat_summary.join(examples.rename("Example")).drop(columns=["Pos","Neg"])
            st.markdown("#### Categories (filtered range)")
            st.dataframe(cat_summary[["Count","%","Positive %","Negative %","Example"]], use_container_width=True)

    # IMPORTANT: return empty df to avoid the host app rendering a duplicate 3rd-row table
    return ("NPS by Portfolio", "Reads surveys/ (Sheet 1), computes NPS, and analyzes Suggestions sentiment & categories."), pd.DataFrame()
