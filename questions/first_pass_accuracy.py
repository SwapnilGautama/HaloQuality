# questions/first_pass_accuracy.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple, Optional, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# ---------------------------
# Brand / palette
# ---------------------------
_DARK_BLUE = "#0b3d91"
_DARK_GREY = "#333333"
_SOFT_GREY = "#E0E0E0"

_PASTEL_LINE = "#8ECAE6"
_PASTEL_LINE_2 = "#A1D99B"

_RCA1_BARS = [
    "#9ECAE1", "#A1D99B", "#BDBDBD", "#FDAE6B", "#C6DBEF", "#FDD0A2",
    "#D9F0A3", "#BCBDDC", "#C7E9C0", "#F2F0F7", "#E5F5E0", "#FEE6CE"
]
_RCA1_CUM_LINE = "#74C69D"

JAN_2025 = pd.Period("2025-01")

# ======================
# Data loading
# ======================
def _find_fpa_workbooks() -> List[Path]:
    roots = [
        Path("data/first_pass_accuracy"),
        Path("first_pass_accuracy"),
        Path("data/first_pass_accuracy/"),
    ]
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

def _pick(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    cols = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in cols:
            return cols[c.lower()]
    return None

def _workbook_cache_key(p: Path) -> str:
    try:
        return f"{p.resolve()}::{p.stat().st_mtime_ns}"
    except Exception:
        return str(p)

@st.cache_data(show_spinner=False)
def _read_fpa_cached(path_str: str, path_key: str) -> pd.DataFrame:
    return _read_excel_any(Path(path_str))

def _normalize_one_fpa(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, str]]:
    col_map = {
        "date": _pick(df, ["Activity Date", "ActivityDate", "Date", "Activity date"]),
        "result": _pick(df, ["Review Result", "Review result", "Result"]),
        "portfolio": _pick(df, ["Portfolio", "portfolio"]),
        "scheme": _pick(df, ["Scheme", "Scheme Name", "Plan", "Plan Name"]),
        "comment": _pick(df, ["Case Comment", "Comments", "Reviewer Comment", "Comment"]),
        "rca2": _pick(df, ["RCA2", "Root Cause 2", "RCA 2"]),
        "team": _pick(df, ["Team manager", "Team Manager", "Manager", "Team", "Assign To Team", "Department", "TeamManager"]),
        "work_type": _pick(df, ["Work Type", "WorkType", "Activity Type"]),
        "individual": _pick(df, ["Administrator", "Reviewer", "User", "Owner", "Analyst"]),
        "location": _pick(df, ["Location", "Region", "Site", "Office", "Branch"]),
    }
    missing = [k for k, v in col_map.items() if k in ("date", "result") and v is None]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    out = df.rename(columns={v: k for k, v in col_map.items() if v}).copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce", dayfirst=True)
    out["_m"] = out["date"].dt.to_period("M")
    res = out["result"].astype(str).str.strip().str.lower()
    out["is_pass"] = res.str.startswith("pass")
    for opt in ("portfolio", "team", "individual", "location"):
        if opt in out.columns:
            out[opt] = out[opt].astype("category")

    keep = [c for c in ["date", "_m", "result", "is_pass",
                        "portfolio", "scheme", "comment", "rca2",
                        "team", "work_type", "individual", "location"]
            if c in out.columns]
    return out[keep], col_map

# ---------- fast combine cache ----------
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
def _combine_normalised_fpa_cached(sig: str, path_strs: List[str], path_keys: List[str]) -> Tuple[pd.DataFrame, Dict[str, str]]:
    frames = []
    last_map: Dict[str, str] = {}
    for p_str, key in zip(path_strs, path_keys):
        df_src = _read_fpa_cached(p_str, key)
        norm, cmap = _normalize_one_fpa(df_src)
        frames.append(norm)
        last_map = cmap
    if not frames:
        empty = pd.DataFrame(columns=["date", "_m", "result", "is_pass", "portfolio"])
        return empty, {}
    combined = pd.concat(frames, ignore_index=True)
    return combined, last_map

def _load_fpa() -> Tuple[pd.DataFrame, Dict[str, str]]:
    paths = _find_fpa_workbooks()
    if not paths:
        raise FileNotFoundError("Could not find any FirstPassAccuracy workbooks (FirstPassAccuracy*.xlsx).")
    sig = _workbooks_signature(paths)
    path_strs = [str(p) for p in paths]
    path_keys  = [_workbook_cache_key(p) for p in paths]
    combined, last_map = _combine_normalised_fpa_cached(sig, path_strs, path_keys)
    return combined, last_map

# ======================
# Normalization helpers
# ======================
def _norm_series(s: pd.Series) -> pd.Series:
    return s.astype(str).str.normalize("NFKC").str.strip().str.casefold()

def _mask_eq(s: pd.Series, value: str) -> pd.Series:
    if value is None or s is None:
        return pd.Series(False, index=s.index)
    return _norm_series(s) == (str(value).strip().casefold())

def _mask_in(s: pd.Series, values: List[str]) -> pd.Series:
    if not values or s is None:
        return pd.Series(False, index=s.index)
    target = {str(v).strip().casefold() for v in values}
    return _norm_series(s).isin(target)

# ======================
# Pass% & MoM helpers
# ======================
def _series_mom(df: pd.DataFrame) -> pd.DataFrame:
    if df["_m"].dropna().empty:
        return pd.DataFrame(columns=["month", "pass_pct"])
    start = JAN_2025
    end = df["_m"].max()
    months = pd.period_range(start, end, freq="M")
    g = (df.groupby("_m")
           .agg(total=("is_pass", "size"), passed=("is_pass", "sum"))
           .reindex(months, fill_value=0))
    pct = (g["passed"] * 100.0 / g["total"].replace(0, np.nan)).fillna(0.0).round(0)
    label = [pd.Period(m).to_timestamp().strftime("%b-%y") for m in months]
    return pd.DataFrame({"month": label, "pass_pct": pct.values})

def _table_portfolio_mom(df: pd.DataFrame) -> pd.DataFrame:
    if df["_m"].dropna().empty:
        return pd.DataFrame()
    start = JAN_2025
    end = df["_m"].max()
    months = pd.period_range(start, end, freq="M")
    grp = (df.groupby(["portfolio", "_m"])
             .agg(total=("is_pass", "size"), passed=("is_pass", "sum"))
             .reset_index())
    grp["pass_%"] = (grp["passed"] * 100.0 / grp["total"].replace(0, np.nan)).fillna(0).round(0)
    piv = grp.pivot(index="portfolio", columns="_m", values="pass_%").reindex(columns=months)
    piv.columns = [pd.Period(m).to_timestamp().strftime("%b-%y") for m in piv.columns]
    return piv.sort_index().fillna(0).astype(int)

# ======================
# Reason labelling (cached)
# ======================
@st.cache_data(show_spinner=False)
def _label_all_cached_for_2025(df_raw: pd.DataFrame) -> pd.DataFrame:
    from core.reason_labeller import label_dataframe
    df = df_raw.copy()
    fails = df[(df["_m"] >= JAN_2025) & (~df["is_pass"])].copy()
    if fails.empty:
        return pd.DataFrame(columns=list(df.columns) + ["reason"])
    lab_df = pd.DataFrame({
        "Case Comment": fails["comment"].fillna("").astype(str) if "comment" in fails.columns else "",
        "RCA2": (fails["rca2"].fillna("").astype(str) if "rca2" in fails.columns else "")
    })
    fails["reason"] = label_dataframe(lab_df, text_col="Case Comment", rca2_col="RCA2").fillna("Other").astype(str)
    return fails

def _label_all(df: pd.DataFrame) -> pd.DataFrame:
    return _label_all_cached_for_2025(df)

def _label_all_latest(df: pd.DataFrame, fails_precomputed: Optional[pd.DataFrame] = None) -> Tuple[pd.DataFrame, pd.Period]:
    latest = df["_m"].max()
    if pd.isna(latest):
        return pd.DataFrame(columns=["reason", "count", "percent", "cum_percent"]), latest
    fails = fails_precomputed if fails_precomputed is not None else _label_all(df)
    fails_latest = fails[fails["_m"] == latest].copy()
    if fails_latest.empty:
        return pd.DataFrame(columns=["reason", "count", "percent", "cum_percent"]), latest
    vc = fails_latest["reason"].value_counts().rename_axis("reason").reset_index(name="count")
    vc = vc.sort_values("count", ascending=False).reset_index(drop=True)
    total = int(vc["count"].sum()) or 1
    vc["percent"] = (vc["count"] * 100.0 / total).round(1)
    vc["cum_percent"] = vc["percent"].cumsum().clip(upper=100.0).round(1)
    return vc, latest

def _pivot_fail_matrix(fails: pd.DataFrame) -> pd.DataFrame:
    if fails.empty:
        return pd.DataFrame()
    fails_2025 = fails[fails["_m"] >= JAN_2025].copy()
    if fails_2025.empty:
        return pd.DataFrame()
    months = pd.period_range(JAN_2025, fails_2025["_m"].max(), freq="M")
    month_labels = [pd.Period(m).to_timestamp().strftime("%b-%y") for m in months]
    g = fails_2025.groupby(["portfolio", "reason", "_m"]).size().reset_index(name="count")
    mat = g.pivot_table(index=["portfolio", "reason"], columns="_m", values="count", fill_value=0)
    mat = mat.reindex(columns=months, fill_value=0)
    mat.columns = month_labels
    return mat.sort_index()

# ======================
# Insights helpers — story-style for fail reasons
# ======================
def _reason_trend_story(fails_all: pd.DataFrame, take_top: int = 3) -> List[str]:
    if fails_all.empty:
        return ["No failure-reason signals are available for the selected range."]

    df = fails_all[fails_all["_m"] >= JAN_2025].copy()
    if df.empty:
        return ["No 2025 failure-reason signals are available for the selected range."]

    overall = df["reason"].value_counts().rename_axis("reason").reset_index(name="count")
    overall["share%"] = (overall["count"] * 100.0 / overall["count"].sum()).round(1)
    top_overall = overall.head(take_top)

    ts = (df.groupby(["_m", "reason"]).size().rename("count").reset_index())
    months = sorted(ts["_m"].unique())
    window = months[-4:] if len(months) >= 4 else months

    slopes = []
    for r, sub in ts[ts["_m"].isin(window)].groupby("reason"):
        xs = np.arange(len(window))
        counts = pd.Series(0, index=window).add(sub.set_index("_m")["count"], fill_value=0).values
        if len(xs) >= 2 and counts.sum() > 0:
            b1 = float(np.polyfit(xs, counts, 1)[0])
        else:
            b1 = 0.0
        slopes.append((r, b1, counts[-1] if len(counts) else 0))
    sl = pd.DataFrame(slopes, columns=["reason","slope","last_count"])
    rising  = sl.sort_values("slope", ascending=False).head(take_top)
    falling = sl.sort_values("slope", ascending=True).head(take_top)

    latest = df["_m"].max()
    latest_df = df[df["_m"] == latest]
    latest_mix = latest_df["reason"].value_counts().rename_axis("reason").reset_index(name="count")
    if not latest_mix.empty:
        latest_mix["share%"] = (latest_mix["count"] * 100.0 / latest_mix["count"].sum()).round(1)
    top_latest = latest_mix.head(1) if not latest_mix.empty else pd.DataFrame(columns=["reason","count","share%"])

    bullets: List[str] = []
    if not top_overall.empty:
        top_txt = ", ".join([f"**{r}** ({p:.1f}%)" for r, p in zip(top_overall["reason"], top_overall["share%"])])
        bullets.append(f"**Top fail drivers overall** (since Jan ’25): {top_txt}.")
    if not rising.empty:
        r_txt = ", ".join([f"**{r}** (↗ ~{s:.1f}/mo)" for r, s in zip(rising["reason"], rising["slope"]) if s > 0])
        if r_txt: bullets.append(f"**Rising reasons (last {len(window)} months)**: {r_txt}.")
    if not falling.empty:
        f_txt = ", ".join([f"**{r}** (↘ ~{abs(s):.1f}/mo)" for r, s in zip(falling["reason"], falling["slope"]) if s < 0])
        if f_txt: bullets.append(f"**Falling reasons (last {len(window)} months)**: {f_txt}.")
    if not top_latest.empty:
        reason0 = str(top_latest.iloc[0]["reason"])
        share0  = float(top_latest.iloc[0]["share%"])
        by_port = (latest_df[latest_df["reason"] == reason0]
                   .groupby("portfolio").size().sort_values(ascending=False).head(2))
        where = ", ".join(by_port.index.astype(str).tolist()) if not by_port.empty else "various portfolios"
        bullets.append(f"**Latest month**: **{reason0}** leads ({share0:.1f}% of fails), concentrated in {where}.")
    return bullets if bullets else ["No strong up/down movements detected in fail reasons."]

# ======================
# Small figs
# ======================
def _fig_mom(df: pd.DataFrame, title: str):
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    ax.plot(df["month"], df["pass_pct"], linewidth=3.2, color=_PASTEL_LINE)
    for x, y in zip(df["month"], df["pass_pct"]):
        ax.text(x, y + 2, f"{y:.0f}%", ha="center", va="bottom", fontsize=9, color=_DARK_GREY)
    for sp in ["left", "right", "top"]:
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color(_SOFT_GREY)
    ax.spines["bottom"].set_linewidth(1.25)
    ax.get_yaxis().set_visible(False)
    ax.set_xlabel(""); ax.set_ylabel(""); ax.grid(False)
    ax.set_title(title, pad=8, color=_DARK_BLUE)
    ax.set_ylim(bottom=0, top=100)
    return fig

# ======================
# Comparison helpers (existing)
# ======================
def _pass_mom_by_dim(df: pd.DataFrame, dim_col: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if df["_m"].dropna().empty:
        return pd.DataFrame(), pd.DataFrame()
    start = JAN_2025
    end = df["_m"].max()
    months = pd.period_range(start, end, freq="M")
    lab = [m.to_timestamp().strftime("%b-%y") for m in months]

    g = (df.groupby([dim_col, "_m"])
           .agg(total=("is_pass", "size"), passed=("is_pass", "sum"))
           .reset_index())
    g["pass_%"] = (g["passed"] * 100.0 / g["total"].replace(0, np.nan)).fillna(0).round(0)

    piv = g.pivot(index=dim_col, columns="_m", values="pass_%").reindex(columns=months)
    piv.columns = lab
    piv = piv.fillna(0).sort_index()

    latest = end
    lt = g[g["_m"] == latest][[dim_col, "pass_%", "total"]]
    lt = lt[lt["total"] > 0].copy()
    lt = lt[lt["pass_%"] > 0].copy()
    latest_tab = lt.set_index(dim_col)[["pass_%"]].sort_values("pass_%", ascending=False)
    return piv, latest_tab

# ------------- Complaints & Accuracy figs (unchanged) -------------
def _build_month_any(s: pd.Series, assume_year_if_name_is_month: bool = False) -> pd.Series:
    if s is None:
        return pd.Series(dtype="period[M]")
    if assume_year_if_name_is_month:
        try:
            coerced = pd.to_datetime(s.astype(str) + " 2025", format="%B %Y", errors="coerce")
        except Exception:
            coerced = pd.to_datetime(s.astype(str) + " 2025", errors="coerce")
        return coerced.dt.to_period("M")
    return pd.to_datetime(s, errors="coerce", dayfirst=True).dt.to_period("M")

def _complaints_cases_series_2025(cases: pd.DataFrame, comp: pd.DataFrame) -> pd.DataFrame:
    if cases is None or comp is None or cases.empty or comp.empty:
        return pd.DataFrame(columns=["month", "per_1000"])
    def _pick_local(df, opts): return _pick(df, opts)
    cases_date = _pick_local(cases, ["Create Date (cases)", "Create Date", "Create date", "Start Date", "StartDate", "Created On", "CreateDt"])
    comp_date = _pick_local(comp, ["Date Complaint Received - DD/MM/YY", "Date Complaint Received", "Complaint Date", "Received Date", "Month"])
    if cases_date is None or comp_date is None:
        return pd.DataFrame(columns=["month", "per_1000"])
    cases_m = _build_month_any(cases[cases_date])
    comp_m = _build_month_any(comp[comp_date], assume_year_if_name_is_month=(comp_date.lower()=="month"))
    mask_cases = (cases_m.dt.year == 2025); mask_comp = (comp_m.dt.year == 2025)
    cases = cases.loc[mask_cases].copy(); comp = comp.loc[mask_comp].copy()
    cases["_m"] = cases_m[mask_cases]; comp["_m"] = comp_m[mask_comp]
    if cases.empty: return pd.DataFrame(columns=["month", "per_1000"])
    months = pd.period_range(JAN_2025, max(cases["_m"].max(), comp["_m"].max() if not comp.empty else JAN_2025), freq="M")
    cs = cases.groupby("_m").size().reindex(months, fill_value=0)
    cp = comp.groupby("_m").size().reindex(months, fill_value=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        per = (cp * 1000.0 / cs.replace(0, np.nan)).fillna(0.0)
    lab = [m.to_timestamp().strftime("%b-%y") for m in months]
    return pd.DataFrame({"month": lab, "per_1000": per.round(1).values})

def _merge_complaints_fpa_for_chart(m_comp: pd.DataFrame, m_fpa: pd.DataFrame) -> pd.DataFrame:
    if m_comp.empty and m_fpa.empty:
        return pd.DataFrame(columns=["month", "complaints_per_1000", "pass_pct"])
    df = pd.merge(
        m_comp.rename(columns={"per_1000": "complaints_per_1000"}),
        m_fpa.rename(columns={"pass_pct": "pass_pct"}),
        on="month", how="outer",
    ).sort_values("month", key=lambda s: pd.to_datetime(s, format="%b-%y"))
    df["complaints_per_1000"] = df["complaints_per_1000"] .fillna(0).astype(float).round(1)
    df["pass_pct"] = df["pass_pct"].fillna(0).astype(float).round(0)
    return df

def _fig_complaints_accuracy(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8.4, 3.6))
    ax.plot(df["month"], df["complaints_per_1000"], linewidth=2.8, color=_PASTEL_LINE, label="Complaints / 1000")
    ax.plot(df["month"], df["pass_pct"], linewidth=2.8, color=_PASTEL_LINE_2, label="Pass %")
    for x, y in zip(df["month"], df["complaints_per_1000"]):
        ax.text(x, y + max(0.05, 0.02*y), f"{y:.1f}", ha="center", va="bottom", fontsize=8, color=_DARK_GREY)
    for x, y in zip(df["month"], df["pass_pct"]):
        ax.text(x, y + 1.2, f"{y:.0f}%", ha="center", va="bottom", fontsize=8, color=_DARK_GREY)
    for sp in ["left", "right", "top"]:
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color(_SOFT_GREY); ax.spines["bottom"].set_linewidth(1.25)
    ax.get_yaxis().set_visible(False); ax.set_xlabel(""); ax.grid(False)
    ax.legend(frameon=False, loc="upper right")
    return fig

# ======================
# Streamlit entry
# ======================
def run(store: Dict, params: Dict, user_text: str = "") -> Tuple[str, pd.DataFrame]:
    # Load
    try:
        df_raw, _ = _load_fpa()
    except FileNotFoundError as e:
        st.error(str(e)); return ("", pd.DataFrame())
    except KeyError as e:
        st.error(f"FPA file found, but a required column is missing: {e}")
        return ("", pd.DataFrame())

    # Overview series
    mom = _series_mom(df_raw)
    if mom.empty:
        st.info("No First-Pass Accuracy rows found from Jan-25 onward.")
        return ("", pd.DataFrame())

    latest = df_raw["_m"].max()
    piv_portfolio_mom = _table_portfolio_mom(df_raw)

    # Build tabs (Insights updated; others preserved)
    tab_insights, tab_overview, tab_comparisons, tab_comp_acc = st.tabs(
        ["Insights", "Overview", "Comparisons", "Complaints and Accuracy"]
    )

    # ---------------- Insights (Story + Outliers) ----------------
    with tab_insights:
        st.markdown(
            f"<h4 style='color:{_DARK_BLUE};margin:0 0 .5rem 0;'>FPA Insights — Jan–{pd.Period(latest).to_timestamp().strftime('%b %y')}</h4>",
            unsafe_allow_html=True,
        )

        # High-level trend bullets
        def _heuristic_insights(mom_df: pd.DataFrame) -> List[str]:
            overall_series = mom_df["pass_pct"].astype(float)
            last_val = overall_series.iloc[-1] if len(overall_series) else np.nan
            prev_val = overall_series.iloc[-2] if len(overall_series) >= 2 else np.nan
            delta = (last_val - prev_val) if not (np.isnan(last_val) or np.isnan(prev_val)) else np.nan
            slope = 0.0
            if len(overall_series) >= 3:
                x = np.arange(len(overall_series))
                slope = float(np.polyfit(x, overall_series, 1)[0])
            hit_rate = float((overall_series >= 90).mean() * 100.0) if len(overall_series) else np.nan

            bullets = []
            if not np.isnan(last_val):
                if not np.isnan(delta):
                    bullets.append(f"**Month-on-Month Pass Rate**: {mom_df['month'].iloc[-1]} **{last_val:.0f}%** ({'+' if delta>=0 else ''}{delta:.0f} pp MoM).")
                else:
                    bullets.append(f"**Month-on-Month Pass Rate**: Latest {mom_df['month'].iloc[-1]} **{last_val:.0f}%**.")
            if len(overall_series) >= 3:
                dir_word = "upward" if slope > 0.2 else ("downward" if slope < -0.2 else "flat")
                bullets.append(f"**Trend**: {dir_word} over the observed period (slope {slope:+.2f} pp/month).")
            if not np.isnan(hit_rate):
                bullets.append(f"**Quality target adherence**: {hit_rate:.0f}% of months at or above 90% pass.")
            return bullets[:4]

        for b in _heuristic_insights(mom):
            st.markdown(f"- {b}")

        # Left: MoM line | Right: fail-reason story (no chart)
        c1, c2 = st.columns((1.05, 1.0), gap="large")
        with c1:
            st.pyplot(_fig_mom(mom, "Overall Pass % — Month on Month"))
        with c2:
            with st.spinner("Analysing fail reasons…"):
                fails_all = _label_all(df_raw)  # cached AI labelling
                latest_reasons, lastp = _label_all_latest(df_raw, fails_precomputed=fails_all)

            st.markdown(
                f"<h5 style='color:{_DARK_BLUE};margin:.25rem 0 .5rem 0;'>Why cases fail — story</h5>",
                unsafe_allow_html=True,
            )
            if fails_all.empty:
                st.info("No fail-reason data available for the selected range.")
            else:
                story_lines = _reason_trend_story(fails_all, take_top=3)
                for ln in story_lines:
                    st.markdown(f"- {ln}")

                if not latest_reasons.empty:
                    lead = latest_reasons.iloc[0]
                    st.caption(
                        f"Latest month ({pd.Period(lastp).to_timestamp().strftime('%b-%y')}): "
                        f"**{lead['reason']}** leads with **{int(lead['count'])}** cases ({lead['percent']:.1f}% of fails)."
                    )

        st.divider()

        # >>>>> Outliers (re-added, unchanged logic) <<<<<
        st.markdown(f"<h5 style='color:{_DARK_BLUE};margin:.25rem 0 .5rem 0;'>Outliers — latest month pass %</h5>", unsafe_allow_html=True)

        def _top_bottom(latest_tab: pd.DataFrame, k: int = 3) -> Tuple[pd.DataFrame, pd.DataFrame]:
            if latest_tab.empty:
                return pd.DataFrame(), pd.DataFrame()
            best = latest_tab.head(k).copy()
            worst = latest_tab.tail(k).copy().sort_values("pass_%")
            return best, worst

        # Portfolio
        _, lt_port = _pass_mom_by_dim(df_raw, "portfolio") if "portfolio" in df_raw.columns else (pd.DataFrame(), pd.DataFrame())
        b1, b2 = st.columns(2)
        with b1:
            if not lt_port.empty:
                best, worst = _top_bottom(lt_port, 3)
                st.markdown("**Portfolio — Top 3**")
                st.dataframe(best.rename(columns={"pass_%": "pass_%"}), use_container_width=True)
        with b2:
            if not lt_port.empty:
                best, worst = _top_bottom(lt_port, 3)
                st.markdown("**Portfolio — Bottom 3**")
                st.dataframe(worst.rename(columns={"pass_%": "pass_%"}), use_container_width=True)

        # Managers
        _, lt_mgr = _pass_mom_by_dim(df_raw, "team") if "team" in df_raw.columns else (pd.DataFrame(), pd.DataFrame())
        c3, c4 = st.columns(2)
        with c3:
            if not lt_mgr.empty:
                best, worst = _top_bottom(lt_mgr, 3)
                st.markdown("**Managers — Top 3**")
                st.dataframe(best.rename(columns={"pass_%": "pass_%"}), use_container_width=True)
        with c4:
            if not lt_mgr.empty:
                best, worst = _top_bottom(lt_mgr, 3)
                st.markdown("**Managers — Bottom 3**")
                st.dataframe(worst.rename(columns={"pass_%": "pass_%"}), use_container_width=True)

        # Individuals
        _, lt_ind = _pass_mom_by_dim(df_raw, "individual") if "individual" in df_raw.columns else (pd.DataFrame(), pd.DataFrame())
        c5, c6 = st.columns(2)
        with c5:
            if not lt_ind.empty:
                best, worst = _top_bottom(lt_ind, 3)
                st.markdown("**Individuals — Top 3**")
                st.dataframe(best.rename(columns={"pass_%": "pass_%"}), use_container_width=True)
        with c6:
            if not lt_ind.empty:
                best, worst = _top_bottom(lt_ind, 3)
                st.markdown("**Individuals — Bottom 3**")
                st.dataframe(worst.rename(columns={"pass_%": "pass_%"}), use_container_width=True)

        st.caption("Notes: Outliers are based on latest-month pass %. Groups with zero cases are excluded automatically. Fail reasons use cached AI labelling to keep this tab snappy.")

    # ---------------- Overview (unchanged) ----------------
    with tab_overview:
        st.markdown(f"<h4 style='color:{_DARK_BLUE};margin:0 0 .5rem 0;'>First-Pass Accuracy — Jan–{pd.Period(latest).to_timestamp().strftime('%b %y')}</h4>", unsafe_allow_html=True)
        st.pyplot(_fig_mom(mom, "Overall Pass % — Month on Month"))
        st.markdown(f"<h4 style='color:{_DARK_BLUE};margin:1rem 0 .5rem 0;'>FPA % by Portfolio — Month on Month</h4>", unsafe_allow_html=True)
        if not piv_portfolio_mom.empty:
            st.dataframe(piv_portfolio_mom, use_container_width=True)

        with st.expander("Fail reasons (AI-labelled) — click to compute", expanded=False):
            with st.spinner("Labelling failures…"):
                fails_all = _label_all(df_raw)  # cached
            reasons_latest, lastp = _label_all_latest(df_raw, fails_precomputed=fails_all)
            matrix_2025 = _pivot_fail_matrix(fails_all)

            if not reasons_latest.empty:
                st.markdown("**Latest-month leading reasons**")
                st.dataframe(reasons_latest, use_container_width=True)
            else:
                st.info("No fail reasons available for the latest month.")

            if not matrix_2025.empty:
                st.markdown("**Fail Reasons × Portfolio — Month on Month (2025)**")
                st.dataframe(matrix_2025, use_container_width=True)
            else:
                st.info("No 2025 fail reason data available to populate the matrix.")

    # ---------------- Comparisons (unchanged) ----------------
    with tab_comparisons:
        st.markdown(f"<h4 style='color:{_DARK_BLUE};margin:0 0 1rem 0;'>Comparison analysis — Accuracy (Pass %)</h4>", unsafe_allow_html=True)

        have_portfolio = "portfolio" in df_raw.columns
        have_team = "team" in df_raw.columns
        have_individual = "individual" in df_raw.columns
        have_location = "location" in df_raw.columns

        def _overall_latest_pass_pct(df_sel: pd.DataFrame) -> Optional[float]:
            dfx = df_sel.copy()
            if dfx["_m"].dropna().empty:
                return None
            latest_local = dfx["_m"].max()
            sub = dfx[dfx["_m"] == latest_local]["is_pass"]
            total = int(sub.size)
            if total == 0:
                return None
            passed = int(sub.sum())
            return round(passed * 100.0 / total, 0)

        def _bar_with_avg(latest_tab: pd.DataFrame, title: str, overall_pct: Optional[float]):
            if latest_tab.empty:
                return
            fig, ax = plt.subplots(figsize=(7.2, 3.2))
            x = np.arange(len(latest_tab))
            vals = latest_tab["pass_%"].values.astype(float)
            bars = ax.bar(x, vals, color="#BFD7EA")
            lift = (np.nanmax(vals) if len(vals) else 0) * 0.02 + 1
            for i, b in enumerate(bars):
                ax.text(b.get_x()+b.get_width()/2, b.get_height()+lift, f"{vals[i]:.0f}%", ha="center", va="bottom", fontsize=9, color=_DARK_GREY)
            if overall_pct is not None:
                ax.axhline(y=overall_pct, linewidth=2.2, color=_PASTEL_LINE)
                ax.text(len(x)-0.5, overall_pct + 1.5, f"Avg {overall_pct:.0f}%", ha="right", va="bottom", fontsize=9, color=_DARK_GREY)
            ax.set_xticks(x); ax.set_xticklabels(latest_tab.index.tolist(), rotation=90, ha="center", fontsize=8, color=_DARK_GREY)
            for sp in ["left", "right", "top"]: ax.spines[sp].set_visible(False)
            ax.spines["bottom"].set_color(_SOFT_GREY)
            ax.get_yaxis().set_visible(False); ax.grid(False)
            ax.set_title(title, color=_DARK_BLUE)
            st.pyplot(fig)

        # Managers
        st.markdown("### Managers")
        if have_team:
            if have_portfolio:
                p_opts = sorted(df_raw["portfolio"].dropna().astype(str).unique().tolist())
                sel_p_for_mgr = st.selectbox("Portfolio (Managers section)", options=p_opts, index=0 if p_opts else 0, key="cmp_mgr_portfolio")
                df_mgr = df_raw[_mask_eq(df_raw["portfolio"], sel_p_for_mgr)].copy()
            else:
                st.caption("Portfolio column not found; showing all portfolios for Managers section.")
                df_mgr = df_raw.copy()

            piv_team, latest_tab_team = _pass_mom_by_dim(df_mgr, "team")
            overall_mgr = _overall_latest_pass_pct(df_mgr)
            if overall_mgr is not None and not latest_tab_team.empty:
                latest_tab_team = latest_tab_team.copy(); latest_tab_team.loc["Overall"] = overall_mgr

            col_chart, col_table = st.columns((1.0, 1.0), gap="large")
            with col_chart:
                if not latest_tab_team.empty:
                    _bar_with_avg(latest_tab_team, f"Team managers — {sel_p_for_mgr if have_portfolio else 'All'} (latest Pass %)", overall_mgr)
                else:
                    st.info("No latest-month data for selected portfolio.")
            with col_table:
                if not latest_tab_team.empty:
                    st.dataframe(latest_tab_team.rename(columns={"pass_%": "pass_%"}), use_container_width=True)
        else:
            st.info("Team manager column not present in data.")
        st.divider()

        # Individuals
        st.markdown("### Individuals")
        if have_individual:
            if have_team:
                mgr_opts = sorted(df_raw["team"].dropna().astype(str).unique().tolist())
                default_mgrs = mgr_opts
                sel_mgrs_for_ind = st.multiselect("Team manager (Individuals section)", options=mgr_opts, default=default_mgrs, key="cmp_ind_managers")
                df_ind = df_raw[_mask_in(df_raw["team"], sel_mgrs_for_ind)].copy() if sel_mgrs_for_ind else df_raw.head(0).copy()
            else:
                st.caption("Team manager column not found; Individuals section will not filter by manager.")
                df_ind = df_raw.copy()

            _, latest_tab_ind = _pass_mom_by_dim(df_ind, "individual")
            overall_ind = _overall_latest_pass_pct(df_ind)
            if overall_ind is not None and not latest_tab_ind.empty:
                latest_tab_ind = latest_tab_ind.copy(); latest_tab_ind.loc["Overall"] = overall_ind

            col_chart, col_table = st.columns((1.0, 1.0), gap="large")
            with col_chart:
                if not latest_tab_ind.empty:
                    _bar_with_avg(latest_tab_ind, "Individuals — (latest Pass %)", overall_ind)
                else:
                    st.info("No latest-month data for selected managers.")
            with col_table:
                if not latest_tab_ind.empty:
                    st.dataframe(latest_tab_ind.rename(columns={"pass_%": "pass_%"}), use_container_width=True)
        else:
            st.info("Individuals column not present in data.")
        st.divider()

        # Locations
        st.markdown("### Locations")
        if have_location:
            if have_portfolio:
                p_opts_loc = sorted(df_raw["portfolio"].dropna().astype(str).unique().tolist())
                sel_p_for_loc = st.selectbox("Portfolio (Locations section)", options=p_opts_loc, index=0 if p_opts_loc else 0, key="cmp_loc_portfolio")
                df_loc = df_raw[_mask_eq(df_raw["portfolio"], sel_p_for_loc)].copy()
            else:
                st.caption("Portfolio column not found; showing all portfolios for Locations section.")
                df_loc = df_raw.copy()

            _, latest_tab_loc = _pass_mom_by_dim(df_loc, "location")
            overall_loc = _overall_latest_pass_pct(df_loc)
            if overall_loc is not None and not latest_tab_loc.empty:
                latest_tab_loc = latest_tab_loc.copy(); latest_tab_loc.loc["Overall"] = overall_loc

            col_chart, col_table = st.columns((1.0, 1.0), gap="large")
            with col_chart:
                if not latest_tab_loc.empty:
                    _bar_with_avg(latest_tab_loc, f"Locations — {sel_p_for_loc if have_portfolio else 'All'} (latest Pass %)", overall_loc)
                else:
                    st.info("No latest-month data for selected portfolio.")
            with col_table:
                if not latest_tab_loc.empty:
                    st.dataframe(latest_tab_loc.rename(columns={"pass_%": "pass_%"}), use_container_width=True)
        else:
            st.info("Location column not present in data.")
        st.divider()

        # Portfolio
        st.markdown("### Portfolio")
        if have_portfolio:
            if have_location:
                loc_opts = sorted(df_raw["location"].dropna().astype(str).unique().tolist())
                sel_loc_for_port = st.multiselect("Location (Portfolio section)", options=loc_opts, default=loc_opts, key="cmp_port_locations")
                df_port = df_raw[_mask_in(df_raw["location"], sel_loc_for_port)].copy() if sel_loc_for_port else df_raw.head(0).copy()
            else:
                st.caption("Location column not found; Portfolio section will not filter by location.")
                df_port = df_raw.copy()

            _, latest_tab_port = _pass_mom_by_dim(df_port, "portfolio")
            overall_port = _overall_latest_pass_pct(df_port)
            if overall_port is not None and not latest_tab_port.empty:
                latest_tab_port = latest_tab_port.copy(); latest_tab_port.loc["Overall"] = overall_port

            col_chart, col_table = st.columns((1.0, 1.0), gap="large")
            with col_chart:
                if not latest_tab_port.empty:
                    _bar_with_avg(latest_tab_port, "Portfolio — (latest Pass %)", overall_port)
                else:
                    st.info("No latest-month data for selected locations.")
            with col_table:
                if not latest_tab_port.empty:
                    st.dataframe(latest_tab_port.rename(columns={"pass_%": "pass_%"}), use_container_width=True)
        else:
            st.info("Portfolio column not present in data.")

    # ---------------- Complaints & Accuracy (unchanged) ----------------
    with tab_comp_acc:
        st.markdown(f"<h4 style='color:{_DARK_BLUE};margin:0 0 1rem 0;'>Complaints and Accuracy — Jan to latest 2025</h4>", unsafe_allow_html=True)
        cases: pd.DataFrame = store.get("cases", pd.DataFrame())
        complaints: pd.DataFrame = store.get("complaints", pd.DataFrame())
        if cases is None or complaints is None or cases.empty or complaints.empty:
            st.info("Cases and/or Complaints data not found in the app store. This tab uses `store['cases']` and `store['complaints']`.")
        else:
            cases_port = _pick(cases, ["Portfolio", "portfolio"])
            comp_port = _pick(complaints, ["Portfolio", "portfolio"])
            fpa_port = "portfolio" if "portfolio" in df_raw.columns else None
            port_values = set()
            if cases_port: port_values |= set(cases[cases_port].dropna().astype(str).unique().tolist())
            if comp_port: port_values |= set(complaints[comp_port].dropna().astype(str).unique().tolist())
            if fpa_port:  port_values |= set(df_raw[fpa_port].dropna().astype(str).unique().tolist())
            port_options = sorted([p for p in port_values if p and p.lower() != "nan"])
            sel_ports_local = st.multiselect("Portfolio (local filter for Complaints vs Accuracy)", options=port_options, default=port_options, key="comp_vs_acc_portfolio_local")
            cases_f = cases.copy(); complaints_f = complaints.copy(); df_fpa_f = df_raw.copy()
            if sel_ports_local:
                if cases_port: cases_f = cases_f[_mask_in(cases[cases_port], sel_ports_local)]
                if comp_port: complaints_f = complaints_f[_mask_in(complaints[comp_port], sel_ports_local)]
                if fpa_port:  df_fpa_f = df_fpa_f[_mask_in(df_fpa_f[fpa_port], sel_ports_local)]
            comp_mom = _complaints_cases_series_2025(cases_f, complaints_f)
            fpa_mom = _series_mom(df_fpa_f)
            merged = _merge_complaints_fpa_for_chart(comp_mom, fpa_mom)
            c1, c2 = st.columns((1.2, 1.0), gap="large")
            with c1:
                if not merged.empty: st.pyplot(_fig_complaints_accuracy(merged))
                else: st.info("No overlapping months available to plot for the selected portfolio(s).")
            with c2:
                if not merged.empty:
                    tbl = merged.rename(columns={"complaints_per_1000": "Complaints / 1000", "pass_pct": "Pass %"})
                    st.dataframe(tbl, use_container_width=True)
                else:
                    st.info("No data to display in the table for the selected portfolio(s).")

    return ("", pd.DataFrame())
