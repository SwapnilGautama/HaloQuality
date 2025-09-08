# questions/first_pass_accuracy.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple, Optional, List
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# ---------------------------
# Brand / palette
# ---------------------------
_DARK_BLUE = "#0b3d91"
_DARK_GREY = "#333333"
_SOFT_GREY = "#E0E0E0"   # softer axis baseline

# Pastel palette for the FPA line
_PASTEL_LINE = "#8ECAE6"

# RCA1-like pastel bar palette (soft blues/greens/greys/oranges)
_RCA1_BARS = [
    "#9ECAE1", "#A1D99B", "#BDBDBD", "#FDAE6B", "#C6DBEF", "#FDD0A2",
    "#D9F0A3", "#BCBDDC", "#C7E9C0", "#F2F0F7", "#E5F5E0", "#FEE6CE"
]
# Smooth cumulative line (soft green/teal)
_RCA1_CUM_LINE = "#74C69D"

# ======================
# Data loading
# ======================
def _find_fpa_workbook() -> Optional[Path]:
    roots = [Path("data/first_pass_accuracy"), Path("first_pass_accuracy"), Path("data/first_pass_accuracy/")]
    patterns = ["FirstPassAccuracy*.xls*", "*FirstPassAccuracy*.xls*"]
    for root in roots:
        if not root.exists():
            continue
        for pat in patterns:
            hits = sorted(root.glob(pat))
            if hits:
                return hits[-1]
    return None

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

def _coerce_month(s: pd.Series) -> pd.Series:
    dt = pd.to_datetime(s, errors="coerce", dayfirst=True)
    return dt.dt.to_period("M")

def _load_fpa() -> Tuple[pd.DataFrame, Dict[str, str]]:
    p = _find_fpa_workbook()
    if not p:
        raise FileNotFoundError("Could not find a FirstPassAccuracy workbook (FirstPassAccuracy*.xlsx).")
    df = _read_excel_any(p)
    col_map = {
        "date": _pick(df, ["Activity Date", "ActivityDate", "Date", "Activity date"]),
        "result": _pick(df, ["Review Result", "Review result", "Result"]),
        "portfolio": _pick(df, ["Portfolio", "portfolio"]),
        "scheme": _pick(df, ["Scheme", "Scheme Name", "Plan", "Plan Name"]),
        "comment": _pick(df, ["Case Comment", "Comments", "Reviewer Comment", "Comment"]),
        "rca2": _pick(df, ["RCA2", "Root Cause 2", "RCA 2"]),
        # comparison fields we will try to render on the Comparisons tab
        "administrator": _pick(df, ["Administrator", "Admin"]),
        "team_manager": _pick(df, ["Team Manager", "Manager", "TeamManager"]),
        "team": _pick(df, ["Team", "Assign To Team", "Department"]),
        "work_type": _pick(df, ["Work Type", "WorkType", "Activity Type"]),
        "individual": _pick(df, ["Reviewer", "User", "Owner", "Analyst"]),
        "location": _pick(df, ["Location", "Region", "Site"]),
    }
    missing = [k for k, v in col_map.items() if k in ("date", "result") and v is None]
    if missing:
        raise KeyError(f"Missing required columns for FPA: {missing}")
    df = df.rename(columns={v: k for k, v in col_map.items() if v})
    df["date"] = pd.to_datetime(df["date"], errors="coerce", dayfirst=True)
    return df, col_map

# ======================
# Pass% and tables
# ======================
def _is_pass(x: str) -> bool:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return False
    t = str(x).strip().lower()
    return t.startswith("pass")

def _series_mom(df: pd.DataFrame) -> pd.DataFrame:
    s = _coerce_month(df["date"])
    df = df.assign(_m=s)
    if df["_m"].dropna().empty:
        return pd.DataFrame(columns=["month", "pass_pct"])
    start = pd.Period("2025-01")
    end = df["_m"].max()
    months = pd.period_range(start, end, freq="M")
    g = df.groupby("_m")["result"].agg(
        total="count", passed=lambda x: np.sum([_is_pass(v) for v in x])
    ).reindex(months, fill_value=0)
    pct = (g["passed"] * 100.0 / g["total"].replace(0, np.nan)).fillna(0.0).round(0)
    label = [pd.Period(m).to_timestamp().strftime("%b-%y") for m in months]
    return pd.DataFrame({"month": label, "pass_pct": pct.values})

def _table_portfolio_mom(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["_m"] = _coerce_month(df["date"])
    if df["_m"].dropna().empty:
        return pd.DataFrame()

    start = pd.Period("2025-01")
    end = df["_m"].max()
    months = pd.period_range(start, end, freq="M")

    grp = df.groupby(["portfolio", "_m"])["result"].agg(
        total="count", passed=lambda x: np.sum([_is_pass(v) for v in x])
    ).reset_index()
    grp["pass_%"] = (grp["passed"] * 100.0 / grp["total"]).round(0)

    piv = grp.pivot(index="portfolio", columns="_m", values="pass_%").reindex(columns=months)
    piv.columns = [pd.Period(m).to_timestamp().strftime("%b-%y") for m in piv.columns]
    piv = piv.sort_index().fillna(0).astype(int)
    return piv

# ======================
# Reason labelling helpers
# ======================
def _label_all(df: pd.DataFrame) -> pd.DataFrame:
    from core.reason_labeller import label_dataframe
    df = df.copy()
    df["_m"] = _coerce_month(df["date"])
    fails = df[~df["result"].apply(_is_pass)].copy()
    if fails.empty:
        return pd.DataFrame(columns=list(df.columns) + ["reason"])
    lab_df = pd.DataFrame({
        "Case Comment": fails["comment"].fillna("").astype(str),
        "RCA2": (fails["rca2"].fillna("").astype(str) if "rca2" in fails.columns else "")
    })
    fails["reason"] = label_dataframe(lab_df, text_col="Case Comment", rca2_col="RCA2")\
        .fillna("Other").astype(str)
    return fails

def _label_all_latest(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Period]:
    from core.reason_labeller import label_dataframe
    df = df.assign(_m=_coerce_month(df["date"]))
    latest = df["_m"].max()
    if pd.isna(latest):
        return pd.DataFrame(), latest

    fails = df[(df["_m"] == latest) & (~df["result"].apply(_is_pass))].copy()
    if fails.empty:
        return pd.DataFrame(columns=["reason", "count", "percent", "cum_percent"]), latest

    lab_df = pd.DataFrame({
        "Case Comment": fails["comment"].fillna("").astype(str),
        "RCA2": (fails["rca2"].fillna("").astype(str) if "rca2" in fails.columns else "")
    })
    labels = label_dataframe(lab_df, text_col="Case Comment", rca2_col="RCA2")\
        .fillna("Other").astype(str)

    vc = labels.value_counts().rename_axis("reason").reset_index(name="count")
    vc = vc.sort_values("count", ascending=False).reset_index(drop=True)

    total = int(vc["count"].sum()) or 1
    vc["percent"] = (vc["count"] * 100.0 / total)
    vc["cum_percent"] = vc["percent"].cumsum().clip(upper=100.0)
    vc["percent"] = vc["percent"].round(1)
    vc["cum_percent"] = vc["cum_percent"].round(1)
    return vc, latest

def _pivot_fail_matrix(fails: pd.DataFrame) -> pd.DataFrame:
    if fails.empty:
        return pd.DataFrame()
    start_2025 = pd.Period("2025-01")
    fails_2025 = fails[fails["_m"] >= start_2025].copy()
    if fails_2025.empty:
        return pd.DataFrame()

    months = pd.period_range(start_2025, fails_2025["_m"].max(), freq="M")
    month_labels = [pd.Period(m).to_timestamp().strftime("%b-%y") for m in months]

    g = fails_2025.groupby(["portfolio", "reason", "_m"]).size().reset_index(name="count")
    mat = g.pivot_table(index=["portfolio", "reason"], columns="_m", values="count", fill_value=0)
    mat = mat.reindex(columns=months, fill_value=0)
    mat.columns = month_labels
    mat = mat.sort_index()
    return mat

# ======================
# Insights (AI first, then heuristic fallback)
# ======================
def _format_period(p: pd.Period) -> str:
    try:
        return pd.Period(p).to_timestamp().strftime("%b-%y")
    except Exception:
        return str(p)

def _safe_int(x) -> int:
    try:
        return int(x)
    except Exception:
        return 0

def _portfolio_pass_table(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["_m"] = _coerce_month(df["date"])
    grp = df.groupby(["portfolio", "_m"])["result"].agg(
        total="count", passed=lambda x: np.sum([_is_pass(v) for v in x])
    ).reset_index()
    grp["pass_%"] = (grp["passed"] * 100.0 / grp["total"].replace(0, np.nan)).fillna(0).round(0)
    return grp

def _get_prev_period(periods: List[pd.Period], latest: pd.Period) -> Optional[pd.Period]:
    prevs = [p for p in periods if p < latest]
    return prevs[-1] if prevs else None

def _heuristic_insights(mom: pd.DataFrame, df_raw: pd.DataFrame, fails_all: pd.DataFrame) -> List[str]:
    overall_series = mom["pass_pct"].astype(float)
    last_val = overall_series.iloc[-1] if len(overall_series) else np.nan
    prev_val = overall_series.iloc[-2] if len(overall_series) >= 2 else np.nan
    delta = (last_val - prev_val) if not (np.isnan(last_val) or np.isnan(prev_val)) else np.nan

    df_raw = df_raw.copy()
    df_raw["_m"] = _coerce_month(df_raw["date"])
    periods = sorted(df_raw["_m"].dropna().unique().tolist())
    latest = periods[-1] if periods else None
    prev = _get_prev_period(periods, latest) if latest else None

    ptab = _portfolio_pass_table(df_raw)

    def _extract_pass(grp, at_period):
        if at_period is None:
            return pd.Series(dtype=float)
        sub = grp[grp["_m"] == at_period][["portfolio", "pass_%"]].set_index("portfolio")["pass_%"]
        return sub

    curr = _extract_pass(ptab, latest)
    prevp = _extract_pass(ptab, prev)
    change = (curr - prevp).dropna().sort_values(ascending=False) if not curr.empty and not prevp.empty else pd.Series(dtype=float)
    top_up = change.head(2)
    top_down = change.tail(2).sort_values()

    reasons_points = []
    contrib_points = []
    obs_points = []

    if not fails_all.empty and latest is not None:
        latest_fails = fails_all[fails_all["_m"] == latest]
        prev_fails = fails_all[fails_all["_m"] == prev] if prev is not None else pd.DataFrame(columns=fails_all.columns)

        by_port = latest_fails.groupby("portfolio").size().sort_values(ascending=False)
        if not by_port.empty:
            top_port, top_port_cnt = by_port.index[0], int(by_port.iloc[0])
            total_latest = int(by_port.sum()) or 1
            share = round(top_port_cnt * 100.0 / total_latest, 1)
            obs_points.append(f"Failures concentrated in **{top_port}** ({share}% of { _format_period(latest) } fails).")

        by_reason = latest_fails.groupby("reason").size().sort_values(ascending=False)
        if not by_reason.empty:
            top_reason, top_reason_cnt = by_reason.index[0], int(by_reason.iloc[0])
            total_latest = int(by_reason.sum()) or 1
            share_r = round(top_reason_cnt * 100.0 / total_latest, 1)
            obs_points.append(f"**{top_reason}** accounts for **{share_r}%** of { _format_period(latest) } fails.")

        top2 = by_reason.head(2).index.tolist()
        for r in top2:
            c_now = _safe_int((latest_fails["reason"] == r).sum())
            c_prev = _safe_int((prev_fails["reason"] == r).sum()) if not prev_fails.empty else 0
            delta_r = c_now - c_prev
            reasons_points.append(
                f"**{r}**: {c_now} in { _format_period(latest) } "
                f"({'+' if delta_r>=0 else ''}{delta_r} vs { _format_period(prev) if prev else 'prior' })."
            )
            tops = latest_fails[latest_fails["reason"] == r].groupby("portfolio").size().sort_values(ascending=False).head(3)
            if not tops.empty:
                parts = ", ".join([f"{k} ({int(v)})" for k, v in tops.items()])
                contrib_points.append(f"Top portfolios for **{r}**: {parts}.")

    bullets: List[str] = []
    if not np.isnan(last_val):
        if not np.isnan(delta):
            bullets.append(
                f"**Month-on-Month Pass Rate**: {mom['month'].iloc[-1]} **{last_val:.0f}%** "
                f"({'+' if delta>=0 else ''}{delta:.0f} pp MoM)."
            )
        else:
            bullets.append(f"**Month-on-Month Pass Rate**: Latest {mom['month'].iloc[-1]} **{last_val:.0f}%**.")

    if not change.empty:
        lines = []
        for idx, val in top_up.items():
            lines.append(f"  - *{idx}*: {('+' if val>=0 else '')}{val:.0f} pp MoM")
        for idx, val in top_down.items():
            lines.append(f"  - *{idx}*: {val:.0f} pp MoM")
        if lines:
            bullets.append("**Standout Portfolios**:\n" + "\n".join(lines))

    if reasons_points:
        bullets.append("**Fail Reasons (Top 2)**:\n" + "\n".join([f"  - {p}" for p in reasons_points]))

    if obs_points or contrib_points:
        combined = obs_points + contrib_points
        bullets.append("**Standout Observations**:\n" + "\n".join([f"  - {p}" for p in combined]))

    if len(bullets) > 4:
        bullets = bullets[:4]
    return bullets

def _openai_insights(mom: pd.DataFrame, df_raw: pd.DataFrame, fails_all: pd.DataFrame) -> Optional[List[str]]:
    try:
        from openai import OpenAI  # type: ignore
    except Exception:
        return None
    try:
        df_tmp = df_raw.copy()
        df_tmp["_m"] = _coerce_month(df_tmp["date"])
        latest = df_tmp["_m"].max()
        prevs = sorted(df_tmp["_m"].dropna().unique().tolist())
        prev = prevs[-2] if len(prevs) >= 2 else None

        mom_str = mom.to_csv(index=False)

        grp = _portfolio_pass_table(df_raw)
        ptab_str = grp.to_csv(index=False)

        if fails_all.empty:
            fr_str = "none"
        else:
            latest_fails = fails_all[fails_all["_m"] == latest]
            prev_fails = fails_all[fails_all["_m"] == prev] if prev is not None else pd.DataFrame(columns=fails_all.columns)
            fr_now = latest_fails.groupby("reason").size().sort_values(ascending=False).head(10)
            fr_prev = prev_fails.groupby("reason").size().sort_values(ascending=False).head(10)
            fr_df = pd.DataFrame({"latest": fr_now}).join(pd.DataFrame({"prev": fr_prev}), how="outer").fillna(0).astype(int)
            fr_str = fr_df.to_csv()

        api_key = os.environ.get("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY", None)
        if not api_key:
            return None
        client = OpenAI(api_key=api_key)

        sys_prompt = (
            "You are a concise analytics assistant. Produce 3–4 bullets using Markdown.\n"
            "- Bullet 1: **Month-on-Month Pass Rate** with latest value and MoM delta.\n"
            "- Bullet 2: **Standout Portfolios** with two indented sub-bullets (biggest ↑/↓), format '  - *Portfolio*: +X pp MoM'.\n"
            "- Bullet 3: **Fail Reasons (Top 2)** with two indented sub-bullets showing counts and MoM deltas.\n"
            "- Bullet 4: **Standout Observations** with indented sub-bullets."
        )
        user_prompt = (
            f"MoM Pass% table (month,pass_pct):\n{mom_str}\n\n"
            f"Portfolio pass% by month (portfolio,_m,pass_%):\n{ptab_str}\n\n"
            f"Fail reasons counts (latest vs prev):\n{fr_str}\n"
        )
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.3,
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        text = resp.choices[0].message.content.strip()
        lines = [ln for ln in text.split("\n") if ln.strip()]
        joined = "\n".join(lines)
        blocks = []
        for part in joined.split("\n- "):
            p = part.strip()
            if not p:
                continue
            if not p.startswith("- "):
                p = "- " + p
            blocks.append(p[2:])
        return blocks[:4] if blocks else None
    except Exception:
        return None

def _render_insights(mom: pd.DataFrame, df_raw: pd.DataFrame, fails_all: pd.DataFrame) -> None:
    bullets = _openai_insights(mom, df_raw, fails_all)
    if not bullets:
        bullets = _heuristic_insights(mom, df_raw, fails_all)

    st.markdown(f"<h4 style='color:{_DARK_BLUE};margin:0 0 .5rem 0;'>AI Insights</h4>", unsafe_allow_html=True)
    if not bullets:
        st.caption("No insights available.")
        return
    for b in bullets[:4]:
        if b:
            st.markdown(f"- {b}")

# ======================
# Plots (styling)
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

def _fig_pareto_full(df: pd.DataFrame):
    fig, ax1 = plt.subplots(figsize=(8.6, 4.0))
    x = np.arange(len(df))
    colors = [_RCA1_BARS[i % len(_RCA1_BARS)] for i in range(len(df))]
    bars = ax1.bar(x, df["count"], color=colors)

    lift = max(df["count"]) * 0.015 if len(df) else 1
    for b in bars:
        ax1.text(b.get_x() + b.get_width()/2, b.get_height() + lift,
                 f"{int(b.get_height())}", ha="center", va="bottom",
                 fontsize=9, color=_DARK_GREY)

    ax2 = ax1.twinx()
    x_dense = np.linspace(x.min(), x.max(), num=max(200, len(x) * 20))
    y_dense = np.interp(x_dense, x, df["cum_percent"].values)
    ax2.plot(x_dense, y_dense, linewidth=2.8, color=_RCA1_CUM_LINE)

    for xi, cp in zip(x, df["cum_percent"]):
        ax2.text(xi, cp + 2, f"{cp:.0f}%", ha="center", va="bottom",
                 fontsize=8, color=_DARK_GREY)

    ax1.set_xticks(x)
    ax1.set_xticklabels(df["reason"], rotation=90, ha="center", color=_DARK_GREY)

    for sp in ["left", "right", "top"]:
        ax1.spines[sp].set_visible(False)
    ax1.spines["bottom"].set_color(_SOFT_GREY)
    ax1.spines["bottom"].set_linewidth(1.25)

    ax1.get_yaxis().set_visible(False)
    ax2.get_yaxis().set_visible(False)
    for sp in ["left", "right", "top", "bottom"]:
        ax2.spines[sp].set_visible(False)

    ax1.set_xlabel(""); ax1.set_ylabel(""); ax2.set_ylabel("")
    ax1.grid(False)
    ax2.set_ylim(0, 100)
    return fig

# ======================
# Comparison helpers (new)
# ======================
def _available_dim(df: pd.DataFrame, logical_name: str, col_map: Dict[str, str]) -> Optional[str]:
    return col_map.get(logical_name) if col_map.get(logical_name) in df.columns else None

def _latest_pass_by_dim(df: pd.DataFrame, dim_col: str) -> Tuple[pd.DataFrame, Optional[pd.Period], Optional[pd.Period]]:
    """
    Latest month pass% by dimension (single column 'pass_%').
    Returns (table, latest_period, prev_period)
    """
    dfx = df.copy()
    dfx["_m"] = _coerce_month(dfx["date"])
    if dfx["_m"].dropna().empty:
        return pd.DataFrame(), None, None
    latest = dfx["_m"].max()
    prevs = sorted(dfx["_m"].dropna().unique().tolist())
    prev = prevs[-2] if len(prevs) >= 2 else None

    g = dfx.groupby([dim_col, "_m"])["result"].agg(
        total="count", passed=lambda x: np.sum([_is_pass(v) for v in x])
    ).reset_index()
    g["pass_%"] = (g["passed"] * 100.0 / g["total"].replace(0, np.nan)).fillna(0).round(0)

    latest_tab = g[g["_m"] == latest][[dim_col, "pass_%"]].set_index(dim_col).sort_values("pass_%", ascending=False)
    return latest_tab, latest, prev

def _mom_delta_by_dim(df: pd.DataFrame, dim_col: str, latest: pd.Period, prev: Optional[pd.Period]) -> pd.Series:
    if latest is None or prev is None:
        return pd.Series(dtype=float)
    dfx = df.copy()
    dfx["_m"] = _coerce_month(dfx["date"])
    g = dfx.groupby([dim_col, "_m"])["result"].agg(
        total="count", passed=lambda x: np.sum([_is_pass(v) for v in x])
    ).reset_index()
    g["pass_%"] = (g["passed"] * 100.0 / g["total"].replace(0, np.nan)).fillna(0).round(0)
    cur = g[g["_m"] == latest].set_index(dim_col)["pass_%"]
    prv = g[g["_m"] == prev].set_index(dim_col)["pass_%"]
    delta = (cur - prv).dropna()
    return delta.sort_values(ascending=False)

def _comparison_insights_block(df_raw: pd.DataFrame, col_map: Dict[str, str]) -> List[str]:
    """
    Build 4 bullets: Administrator, Portfolio, Team manager, Location (if present).
    Each bullet: top performer + biggest MoM mover (▲/▼).
    """
    bullets = []
    for logical, label in [("administrator","Administrator"), ("portfolio","Portfolio"),
                           ("team_manager","Team manager"), ("location","Location")]:
        dim_col = _available_dim(df_raw, logical, col_map)
        if not dim_col:
            continue

        latest_tab, latest, prev = _latest_pass_by_dim(df_raw, dim_col)
        if latest_tab.empty:
            continue

        # top performer (latest)
        top_name = latest_tab.index[0]
        top_val = latest_tab.iloc[0, 0]

        # MoM movers
        delta = _mom_delta_by_dim(df_raw, dim_col, latest, prev)
        mover_up = None
        mover_dn = None
        if not delta.empty:
            mover_up = (delta.index[0], float(delta.iloc[0]))
            mover_dn = (delta.index[-1], float(delta.iloc[-1]))

        parts = [f"**{label}**: Top = *{top_name}* (**{top_val:.0f}%**)"]
        if mover_up:
            parts.append(f"  - Biggest ↑: *{mover_up[0]}* (+{mover_up[1]:.0f} pp MoM)")
        if mover_dn and mover_dn[1] < 0:
            parts.append(f"  - Biggest ↓: *{mover_dn[0]}* ({mover_dn[1]:.0f} pp MoM)")
        bullets.append("\n".join(parts))

        if len(bullets) == 4:
            break
    return bullets[:4]

# ======================
# Streamlit entry
# ======================
def run(store: Dict, params: Dict, user_text: str = "") -> Tuple[str, pd.DataFrame]:
    # Load
    try:
        df_raw, col_map = _load_fpa()
    except FileNotFoundError as e:
        st.error(str(e)); return ("", pd.DataFrame())
    except KeyError as e:
        st.error(f"FPA file found, but a required column is missing: {e}")
        return ("", pd.DataFrame())

    # Overview computations
    mom = _series_mom(df_raw)
    if mom.empty:
        st.info("No First-Pass Accuracy rows found from Jan-25 onward.")
        return ("", pd.DataFrame())

    df_raw = df_raw.assign(_m=_coerce_month(pd.to_datetime(df_raw["date"], errors="coerce", dayfirst=True)))
    latest = df_raw["_m"].max()
    piv_portfolio_mom = _table_portfolio_mom(df_raw)

    # Sidebar filters for Overview row-2 table
    st.sidebar.header("Filters (2025) — Fail reasons")
    fails_all = _label_all(df_raw)
    if not fails_all.empty:
        start_2025 = pd.Period("2025-01")
        fails_2025 = fails_all[fails_all["_m"] >= start_2025].copy()
        all_reasons = sorted(fails_2025["reason"].unique().tolist())
        all_portfolios = sorted(fails_2025["portfolio"].dropna().unique().tolist())
    else:
        all_reasons, all_portfolios = [], []
    sel_reasons = st.sidebar.multiselect("Fail reasons", options=all_reasons, default=all_reasons)
    sel_portfolios = st.sidebar.multiselect("Portfolios", options=all_portfolios, default=all_portfolios)

    # Tabs
    tab_overview, tab_comparisons = st.tabs(["Overview", "Comparisons"])

    # ===================== Tab 1: Overview (existing UI preserved) =====================
    with tab_overview:
        # Insights (global)
        _render_insights(mom, df_raw, fails_all)

        # Row 1
        c1, c2 = st.columns((1.1, 1.0), gap="large")
        with c1:
            st.pyplot(_fig_mom(mom, f"First-Pass Accuracy — Jan–{pd.Period(latest).to_timestamp().strftime('%b %y')}"))
        with c2:
            st.markdown(
                f"<h4 style='color:{_DARK_BLUE};margin:0 0 .5rem 0;'>FPA % by Portfolio — Month on Month</h4>",
                unsafe_allow_html=True
            )
            if not piv_portfolio_mom.empty:
                st.dataframe(piv_portfolio_mom, use_container_width=True)

        # Row 2
        reasons_latest, lastp = _label_all_latest(df_raw)
        matrix_2025 = _pivot_fail_matrix(fails_all)

        st.markdown(
            f"<h4 style='color:{_DARK_BLUE};margin:1rem 0 .5rem 0;'>Reasons for Fail — {pd.Period(lastp).to_timestamp().strftime('%b-%y')}</h4>",
            unsafe_allow_html=True
        )
        r1, r2 = st.columns((1.0, 1.2), gap="large")
        with r1:
            if not reasons_latest.empty:
                st.pyplot(_fig_pareto_full(reasons_latest))
            else:
                st.info("No fail reasons available for the latest month.")
        with r2:
            if not matrix_2025.empty:
                if sel_reasons:
                    matrix_2025 = matrix_2025.loc[matrix_2025.index.get_level_values("reason").isin(sel_reasons)]
                if sel_portfolios:
                    matrix_2025 = matrix_2025.loc[matrix_2025.index.get_level_values("portfolio").isin(sel_portfolios)]
                st.markdown(
                    f"<h4 style='color:{_DARK_BLUE};margin:0 0 .5rem 0;'>Fail Reasons × Portfolio — Month on Month (2025)</h4>",
                    unsafe_allow_html=True
                )
                st.dataframe(matrix_2025, use_container_width=True)
            else:
                st.info("No 2025 fail reason data available to populate the matrix.")

    # ===================== Tab 2: Comparisons (new) =====================
    with tab_comparisons:
        st.markdown(f"<h4 style='color:{_DARK_BLUE};margin:0 0 .5rem 0;'>Comparison analysis — Accuracy (Pass %)</h4>", unsafe_allow_html=True)

        # ---- Insights specific to Comparisons (4 bullets; one per table if available)
        bullets = _comparison_insights_block(df_raw, col_map)
        if bullets:
            for b in bullets:
                st.markdown(f"- {b}")
        else:
            st.caption("No comparison fields available in this dataset.")

        st.divider()

        # Render 4 single-column tables (latest month only)
        dims = [
            ("administrator", "Administrator"),
            ("portfolio", "Portfolio"),
            ("team_manager", "Team manager"),
            ("location", "Location"),
        ]
        for logical, label in dims:
            dim_col = _available_dim(df_raw, logical, col_map)
            if not dim_col:
                continue
            st.markdown(f"**{label} — Latest month Pass %**")
            latest_tab, _, _ = _latest_pass_by_dim(df_raw, dim_col)
            if not latest_tab.empty:
                st.dataframe(latest_tab.rename(columns={"pass_%": "pass_%"}), use_container_width=True)
            else:
                st.info(f"No latest-month data for {label}.")

    return ("", pd.DataFrame())
