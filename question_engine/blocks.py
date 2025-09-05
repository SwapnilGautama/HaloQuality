# question_engine/blocks.py
from __future__ import annotations
import pandas as pd
import plotly.express as px

# ---------- small utils ----------

def _is_empty(x) -> bool:
    return x is None or (hasattr(x, "empty") and bool(getattr(x, "empty")))

def _between_months(df: pd.DataFrame, col: str, m_from: str | None, m_to: str | None) -> pd.DataFrame:
    """Filter df between 'YYYY-MM' month strings (inclusive)."""
    if m_from is None and m_to is None:
        return df
    # make a sortable integer YYYYMM
    z = df[col].astype(str).str.replace("-", "", regex=False)
    zi = pd.to_numeric(z, errors="coerce")
    lo = pd.to_numeric(str(m_from).replace("-", ""), errors="coerce") if m_from else None
    hi = pd.to_numeric(str(m_to).replace("-", ""),   errors="coerce") if m_to else None
    if lo is not None:
        df = df[zi >= lo]
    if hi is not None:
        df = df[zi <= hi]
    return df

def _title(s: str | None) -> str:
    return (s or "").strip().title()

# ---------- handlers ----------

def complaints_per_1000_by_process(store, portfolio: str | None, m_from: str | None, m_to: str | None):
    """
    Uses store.joined_summary (Month, Portfolio_std, Process_std,
    Unique_Cases, Complaints, Complaints_per_1000).
    Aggregates across the selected months & (optionally) portfolio.
    """
    df = getattr(store, "joined_summary", None)
    if _is_empty(df):
        return {"error": "No joined cases/complaints data available."}

    q = df.copy()
    if portfolio:
        p = portfolio.strip().lower()
        q = q[q["Portfolio_std"].str.lower().str.contains(p, na=False)]

    q = _between_months(q, "Month", m_from, m_to)

    if _is_empty(q):
        return {"error": "No rows after applying filters."}

    # Aggregate across months: sum Complaints & Unique_Cases, recompute per 1000
    agg = (
        q.groupby(["Process_std"], dropna=False)
          .agg(Complaints=("Complaints", "sum"),
               Unique_Cases=("Unique_Cases", "sum"))
          .reset_index()
    )
    with pd.option_context("mode.use_inf_as_na", True):
        agg["Complaints_per_1000"] = (agg["Complaints"] * 1000.0) / agg["Unique_Cases"].replace(0, pd.NA)
    agg["Complaints_per_1000"] = agg["Complaints_per_1000"].round(2)
    agg = agg.sort_values("Complaints_per_1000", ascending=False)

    title = "Complaints per 1000 by process"
    if portfolio:
        title += f" — Portfolio: {_title(portfolio)}"
    if m_from or m_to:
        title += f" — Period: {m_from or '...'} to {m_to or '...'}"

    fig = px.bar(
        agg.head(30),  # keep chart readable
        x="Process_std", y="Complaints_per_1000",
        hover_data=["Complaints","Unique_Cases"],
        title=title
    )
    fig.update_layout(xaxis_title="Process", yaxis_title="Complaints per 1000 cases")

    return {"title": title, "df": agg, "fig": fig}


def rca1_by_portfolio_for_process(store, process: str, m_from: str | None, m_to: str | None):
    """
    Uses store.rca_table (Month, Portfolio_std, Process_std, RCA1, RCA1_Count).
    For a given process, shows RCA1 mix by portfolio (aggregated across months).
    """
    rca = getattr(store, "rca_table", None)
    if _is_empty(rca):
        return {"error": "No RCA table available (RCA1 not present in complaints)."}

    q = rca.copy()

    if process:
        proc = process.strip().lower()
        q = q[q["Process_std"].str.lower().str.contains(proc, na=False)]

    q = _between_months(q, "Month", m_from, m_to)

    if _is_empty(q):
        return {"error": "No rows after applying filters."}

    agg = (
        q.groupby(["Portfolio_std", "RCA1"], dropna=False)["RCA1_Count"]
         .sum().reset_index()
    )

    # compute shares within portfolio
    totals = agg.groupby("Portfolio_std")["RCA1_Count"].sum().rename("Total").reset_index()
    out = agg.merge(totals, on="Portfolio_std", how="left")
    out["Share_%"] = (out["RCA1_Count"] / out["Total"] * 100).round(1)
    out = out.sort_values(["Portfolio_std","Share_%"], ascending=[True, False])

    title = f"RCA1 mix by portfolio — Process: {_title(process)}"
    if m_from or m_to:
        title += f" — Period: {m_from or '...'} to {m_to or '...'}"

    fig = px.bar(
        out, x="Portfolio_std", y="Share_%", color="RCA1",
        text="Share_%", barmode="stack", title=title
    )
    fig.update_layout(xaxis_title="Portfolio", yaxis_title="RCA1 share (%)")

    return {"title": title, "df": out, "fig": fig}
