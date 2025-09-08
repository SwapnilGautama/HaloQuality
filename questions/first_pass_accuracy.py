# questions/first_pass_accuracy.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple, Optional, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import altair as alt
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
    """
    Portfolio (rows) × Month (columns) FPA% from Jan-25 to latest.
    """
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
# Reasons — data (ALL months for the matrix; latest for Pareto)
# ======================
def _label_all(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return FAIL rows across all months with a 'reason' column (rule-based).
    """
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

def _reasons_latest(fails_all: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Period]:
    """
    Pareto-ready counts for the latest month only.
    """
    latest = fails_all["_m"].max()
    sub = fails_all[fails_all["_m"] == latest]
    vc = sub["reason"].value_counts().rename_axis("reason").reset_index(name="count")
    if vc.empty:
        return pd.DataFrame(columns=["reason","count","cum_percent"]), latest
    vc = vc.sort_values("count", ascending=False).reset_index(drop=True)
    total = int(vc["count"].sum()) or 1
    vc["percent"] = vc["count"] * 100.0 / total
    vc["cum_percent"] = vc["percent"].cumsum().clip(upper=100.0).round(1)
    return vc[["reason","count","cum_percent"]], latest

def _reason_portfolio_month_matrix(fails_all: pd.DataFrame) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """
    Build a matrix (rows: reason × portfolio; columns: months) with counts.
    Returns (matrix_df, ordered_reasons, ordered_month_labels)
    """
    # Month range + label
    months = pd.period_range(fails_all["_m"].min(), fails_all["_m"].max(), freq="M")
    month_labels = [pd.Period(m).to_timestamp().strftime("%b-%y") for m in months]

    # Group
    g = fails_all.groupby(["reason", "portfolio", "_m"]).size().reset_index(name="count")
    mat = g.pivot_table(index=["reason","portfolio"], columns="_m", values="count", fill_value=0)
    mat = mat.reindex(columns=months, fill_value=0)
    mat.columns = month_labels
    mat = mat.sort_index()

    ordered_reasons = list(fails_all["reason"].value_counts().index)
    return mat, ordered_reasons, month_labels

# ======================
# Plots (Row 1 remains Matplotlib line; Row 2 is Altair interactive)
# ======================
def _fig_mom(df: pd.DataFrame, title: str):
    """
    Smooth pastel line, soft baseline, no y-axis.
    """
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

def _altair_reason_pareto_plus_matrix(vc_latest: pd.DataFrame,
                                      mat: pd.DataFrame,
                                      ordered_reasons: List[str],
                                      month_labels: List[str]) -> alt.Chart:
    """
    Build a composition:
      - Left: Pareto bars (latest month) + smooth cumulative line
      - Right: Heatmap 'table' (months × portfolios) filtered by selected reason
    Clicking a bar filters the heatmap.
    """
    # --- Data prep for charts ---
    vc = vc_latest.copy()
    vc["reason"] = vc["reason"].astype("string")
    vc["order"] = np.arange(1, len(vc) + 1)  # ordering for the line if needed

    # Melt matrix for Altair heatmap
    mat_disp = mat.reset_index().melt(id_vars=["reason","portfolio"], var_name="month", value_name="count")

    # Selection (single reason)
    sel = alt.selection_single(fields=["reason"], on="click", empty="all")

    # Bars
    bars = (
        alt.Chart(vc)
        .mark_bar()
        .encode(
            x=alt.X("reason:N", sort=ordered_reasons, axis=alt.Axis(title=None, labelColor=_DARK_GREY, labelAngle=90)),
            y=alt.Y("count:Q", axis=None),
            color=alt.condition(
                sel,
                alt.value(_RCA1_BARS[0]),
                alt.value("#DCE6F2")
            ),
            tooltip=["reason:N","count:Q","cum_percent:Q"]
        )
        .add_selection(sel)
    )

    # Count labels
    labels = (
        alt.Chart(vc)
        .mark_text(dy=-6, color=_DARK_GREY, size=11)
        .encode(x=alt.X("reason:N", sort=ordered_reasons), y="count:Q", text="count:Q")
    )

    # Smooth cumulative line (use dense interpolation in pandas already -> here simple line)
    line = (
        alt.Chart(vc)
        .mark_line(color=_RCA1_CUM_LINE, strokeWidth=2.8)
        .encode(x=alt.X("reason:N", sort=ordered_reasons, axis=alt.Axis(title=None, labels=False, ticks=False)),
                y=alt.Y("cum_percent:Q", scale=alt.Scale(domain=[0, 100]), axis=None))
    )
    # Line labels
    llabels = (
        alt.Chart(vc)
        .mark_text(dy=-8, color=_DARK_GREY, size=10)
        .encode(x=alt.X("reason:N", sort=ordered_reasons), y="cum_percent:Q", text=alt.Text("cum_percent:Q", format=".0f"))
    )

    left = (bars + labels + line + llabels).properties(width=420, height=280)

    # Heatmap table filtered by selection
    base_heat = (
        alt.Chart(mat_disp)
        .transform_filter(sel)
        .encode(
            x=alt.X("month:N", sort=month_labels, axis=alt.Axis(title=None, labelColor=_DARK_GREY)),
            y=alt.Y("portfolio:N", sort=alt.SortField(field="portfolio", order="ascending"),
                    axis=alt.Axis(title=None, labelColor=_DARK_GREY)),
        )
        .properties(width=420, height=280)
    )
    rects = base_heat.mark_rect().encode(
        color=alt.Color("count:Q", scale=alt.Scale(scheme="blues"), legend=None)
    )
    texts = base_heat.mark_text(color=_DARK_GREY, size=11).encode(text="count:Q")

    right = (rects + texts)

    # Combine side-by-side
    combo = alt.hconcat(left, right).resolve_scale(y="independent")
    return combo

# ======================
# Streamlit entry
# ======================
def run(store: Dict, params: Dict, user_text: str = "") -> Tuple[str, pd.DataFrame]:
    # Row 1: FPA MoM + table
    try:
        df_raw, _ = _load_fpa()
    except FileNotFoundError as e:
        st.error(str(e)); return ("", pd.DataFrame())
    except KeyError as e:
        st.error(f"FPA file found, but a required column is missing: {e}")
        return ("", pd.DataFrame())

    mom = _series_mom(df_raw)
    if mom.empty:
        st.info("No First-Pass Accuracy rows found from Jan-25 onward.")
        return ("", pd.DataFrame())

    df_raw = df_raw.assign(_m=_coerce_month(pd.to_datetime(df_raw["date"], errors="coerce", dayfirst=True)))
    latest = df_raw["_m"].max()

    piv_portfolio_mom = _table_portfolio_mom(df_raw)

    c1, c2 = st.columns((1.1, 1.0), gap="large")
    with c1:
        st.pyplot(_fig_mom(mom, f"First-Pass Accuracy — Jan–{pd.Period(latest).to_timestamp().strftime('%b %y')}"))
    with c2:
        st.markdown(
            f"<h4 style='color:{_DARK_BLUE};margin:0 0 .5rem 0;'>"
            f"FPA % by Portfolio — Month on Month"
            f"</h4>", unsafe_allow_html=True)
        if not piv_portfolio_mom.empty:
            st.dataframe(piv_portfolio_mom, use_container_width=True)

    # Row 2: Reasons — interactive chart + cross-filtered matrix table
    st.markdown(
        f"<h4 style='color:{_DARK_BLUE};margin:1rem 0 .5rem 0;'>"
        f"Reasons for Fail — {pd.Period(latest).to_timestamp().strftime('%b-%y')}"
        f"</h4>", unsafe_allow_html=True)

    fails_all = _label_all(df_raw)  # all months, only FAILs with reason labels
    vc_latest, _ = _reasons_latest(fails_all)
    if vc_latest.empty:
        st.info("No fail reasons available.")
        return ("", pd.DataFrame())

    mat, ordered_reasons, month_labels = _reason_portfolio_month_matrix(fails_all)

    # Interactive composition (bars + cumulative line) ⟷ (matrix heatmap)
    chart = _altair_reason_pareto_plus_matrix(vc_latest, mat, ordered_reasons, month_labels)
    st.altair_chart(chart, use_container_width=True)

    return ("", pd.DataFrame())
