# questions/first_pass_accuracy.py
from __future__ import annotations
import re
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

PASTEL = ["#9ec5fe", "#a3d2a3", "#f6c48f", "#f7a3a3", "#b8b8ff", "#ffd6a5", "#b9e6ff"]

def _load_latest(data_root: Path) -> Path:
    folder = data_root / "first_pass_accuracy"
    files = sorted(folder.glob("FirstPassAccuracy*.xlsx"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        raise FileNotFoundError("Could not find a FirstPassAccuracy workbook (FirstPassAccuracy*.xlsx).")
    return files[0]

def _pick(df: pd.DataFrame, *cands: str) -> str|None:
    low = {c.lower(): c for c in df.columns}
    for c in cands:
        if c in low: return low[c]
    return None

def _coerce(df: pd.DataFrame) -> pd.DataFrame:
    res = _pick(df,"review result","result","reviewresult")
    dt  = _pick(df,"activity date","date","activitydate")
    pf  = _pick(df,"portfolio")
    sch = _pick(df,"scheme","plan","plan name")
    cmt = _pick(df,"case comment","comment","comments","note","notes")
    miss = [n for n,v in {"review result":res,"activity date":dt,"portfolio":pf}.items() if v is None]
    if miss: raise KeyError(f"Missing columns: {miss}")
    out = pd.DataFrame({
        "review_result": df[res],
        "activity_date": pd.to_datetime(df[dt], errors="coerce"),
        "portfolio": df[pf],
        "scheme": df[sch] if sch else "",
        "comment": df[cmt] if cmt else "",
    })
    return out

def _month_index(s: pd.Series) -> pd.DatetimeIndex:
    s = s.dropna()
    if s.empty: return pd.date_range("2025-01-01","2025-01-01",freq="MS")
    start = pd.Timestamp(2025,1,1)
    end = s.max()
    start = min(start, s.min())  # still clamp display later
    return pd.date_range(pd.Timestamp(2025,1,1), end, freq="MS")

def _draw_mom(ax, months, values):
    y = values.reindex(months, fill_value=0.0)
    line, = ax.plot(months, y, marker="o", linewidth=2.5, color="#5f8cff")
    for x,v in zip(months, y.values):
        ax.text(x, v, f"{v:.0f}%", ha="center", va="bottom", fontsize=10)
    ax.set_xticks(months)
    ax.set_xticklabels([m.strftime("%b-%y") for m in months])
    ax.spines["left"].set_visible(False); ax.yaxis.set_visible(False)
    for s in ("top","right"): ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color("#d0d4dd")
    ax.grid(False)

_REASON_PATTERNS: List[Tuple[str,List[str]]] = [
    ("Bank / payment", ["bank","payment","bacs","cheque","refund","funds","payout","pay-out"]),
    ("Trustee / AVC", ["trustee","avc","authoris","approval","governance"]),
    ("Data entry / setup", ["data","setup","key","input","update","address","dob","name","record","typo","amend"]),
    ("Postal / dispatch", ["post","postal","mail","dispatch","letter","courier","sent","receive"]),
    ("Manual calculation", ["manual calc","manual","calc","calculation","recalc"]),
    ("System", ["system","technical","workflow","automation","server","bug","down"]),
    ("Waiting on member/TPA", ["waiting","awaiting","member response","no response","chase","tpa","third party"]),
]

def _label_reason(text: pd.Series) -> pd.Series:
    s = text.fillna("").astype(str).str.lower()
    out = pd.Series(index=s.index, dtype="object")
    hit = pd.Series(False, index=s.index)
    for name, kws in _REASON_PATTERNS:
        pat = "|".join([re.escape(k) for k in kws])
        m = s.str.contains(pat, regex=True)
        out[m & ~hit] = name
        hit |= m
    out[~hit] = "Other"
    return out

def _pareto_top80(counts: pd.Series, title: str):
    df = counts.sort_values(ascending=False).rename_axis("reason").reset_index(name="count")
    total = df["count"].sum()
    df["percent"] = df["count"]/max(total,1)*100
    df["cum_percent"] = df["percent"].cumsum()

    core = df[df["cum_percent"] <= 80]
    rest = df[df["cum_percent"] > 80]
    if not rest.empty:
        other = pd.DataFrame([{
            "reason":"Other",
            "count": int(rest["count"].sum()),
            "percent": rest["percent"].sum(),
            "cum_percent": 100.0
        }])
        pareto = pd.concat([core, other], ignore_index=True)
    else:
        pareto = df

    fig, ax1 = plt.subplots(figsize=(7,4))
    bars = ax1.bar(pareto["reason"], pareto["count"], color=PASTEL[0])
    ax1.bar_label(bars, labels=[f"{int(v)}" for v in pareto["count"]], padding=3)
    ax1.yaxis.set_visible(False)
    for s in ("top","right","left"): ax1.spines[s].set_visible(False)
    ax1.spines["bottom"].set_color("#d0d4dd")
    ax1.set_xticklabels(pareto["reason"], rotation=90)
    ax1.grid(False)

    ax2 = ax1.twinx()
    c = pareto["percent"].cumsum()
    ax2.plot(pareto["reason"], c, marker="o", color="#4f6cdf", linewidth=2)
    for x,v in zip(pareto["reason"], c): ax2.text(x, v, f"{v:.0f}%", ha="center", va="bottom", fontsize=9)
    ax2.set_ylim(0,104); ax2.yaxis.set_visible(False); ax2.spines["right"].set_visible(False)

    ax1.set_title(title, color="#0d3b82", fontsize=14, pad=8)
    fig.tight_layout()
    return pareto, fig

def run(store: Dict, params: Dict, q: str):
    root = store.get("root", Path(__file__).parents[1])
    data_root = store.get("data", root / "data")

    f = _load_latest(data_root)
    df = _coerce(pd.read_excel(f))
    df["passed"] = df["review_result"].astype(str).str.contains("pass", case=False, na=False)
    df["month"] = df["activity_date"].dt.to_period("M").dt.to_timestamp()

    months = _month_index(df["month"])
    mom = (df.groupby("month")["passed"].mean()*100).reindex(months, fill_value=0.0)

    st.markdown(f"## First-Pass Accuracy — {months[0].strftime('%b-%y')}–{months[-1].strftime('%b-%y')}")
    c1, c2 = st.columns([1.05,1.25])

    with c1:
        fig, ax = plt.subplots(figsize=(7,3.4))
        _draw_mom(ax, months, mom)
        st.pyplot(fig, use_container_width=True)

    with c2:
        latest = months[-1]
        cur = df[df["month"]==latest]
        table = (cur.groupby(["portfolio","scheme"])["passed"].mean()*100).round(0).reset_index(name="pass_%")
        st.markdown(f"#### Pass % by Portfolio × Scheme — {latest.strftime('%b-%y')}")
        st.dataframe(table.sort_values(["portfolio","scheme"]), use_container_width=True, hide_index=True)

    st.markdown(f"### Reasons for Fail — {months[-1].strftime('%b-%y')}")
    cur_fail = df[(df["month"]==months[-1]) & (~df["passed"])]
    if cur_fail.empty:
        st.info("No failed records for the most recent month.")
        return

    cur_fail["reason"] = _label_reason(cur_fail["comment"])
    counts = cur_fail["reason"].value_counts()

    r1, r2 = st.columns([1.05,1.25])
    with r1:
        _, fig = _pareto_top80(counts, "Fail reasons — Pareto (top 80%)")
        st.pyplot(fig, use_container_width=True)

    with r2:
        tbl = (counts.rename_axis("reason").reset_index(name="count")
               .assign(percent=lambda d: d["count"]/d["count"].sum()*100)
               .assign(cum_percent=lambda d: d["percent"].cumsum()))
        st.markdown(f"#### Reason breakdown — {months[-1].strftime('%b-%y')}")
        st.dataframe(tbl, use_container_width=True, hide_index=True)

# --- utilities that are safe for Q2 only -----------------------------------
def _matplotlib_sandbox():
    """Context manager to isolate Matplotlib state inside Q2."""
    import contextlib, matplotlib as mpl, matplotlib.pyplot as plt
    @contextlib.contextmanager
    def _ctx():
        old = mpl.rcParams.copy()
        try:
            mpl.rcParams.update(mpl.rcParamsDefault)  # reset to defaults
            yield plt
        finally:
            mpl.rcParams.update(old)
    return _ctx()

def _bar_with_labels(ax, x, y, **bar_kw):
    container = ax.bar(x, y, **bar_kw)  # BarContainer
    ax.bar_label(container, labels=[f"{int(v)}" for v in y], padding=3)
    return container

def _contains_any(text_series, keywords):
    import re
    patt = "|".join(re.escape(k) for k in keywords if k)
    if not patt:
        return text_series*False
    return text_series.str.contains(rf"\b({patt})\b", case=False, regex=True)

# --- Stable entrypoint shim for Q2 (paste at bottom of first_pass_accuracy.py) ---
def _call_first_existing(_names, *args, **kwargs):
    for _n in _names:
        _f = globals().get(_n)
        if callable(_f):
            return _f(*args, **kwargs)
    raise RuntimeError(
        "Q2 entrypoint not found. Expected one of: "
        "'run_q2','entry','render','main','build','page','render_page'."
    )

def run(store, params, q):
    """
    Stable entrypoint used by app.py. Do not rename or change the signature.
    """
    # Is there already a proper run()? If yes, call it.
    if globals().get("run") and getattr(globals()["run"], "__name__", "") != "run":
        return globals()["run"](store, params, q)  # an alternate run existed

    # Otherwise, call your existing implementation.
    return _call_first_existing(
        ["run_q2", "entry", "render", "main", "build", "page", "render_page"],
        store, params, q
    )

