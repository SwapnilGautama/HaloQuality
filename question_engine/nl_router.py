# question_engine/nl_router.py
from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import pandas as pd
from .parser import (
    parse_month_range, parse_group_by, parse_filters, parse_topn, parse_sort,
    infer_domain, resolve_metric
)
from .aggregate import aggregate_generic
from .drivers import drivers_of_fails

def run_nl(text: str, store: Dict[str, pd.DataFrame]) -> Dict:
    """
    Main entrypoint. Parses the text and returns a payload with:
      - 'insights' (list[str])
      - 'tables' (dict[name -> DataFrame])
      - 'figs' (dict[name -> plotly fig])
    """
    txt = (text or "").strip()
    if not txt:
        return {"insights": ["Please type a question."], "tables": {}, "figs": {}}

    # 1) parse intent
    domain = infer_domain(txt)  # complaints | cases | fpa
    m_from, m_to = parse_month_range(txt)
    group_by = parse_group_by(txt)
    filters = parse_filters(txt)
    top_n = parse_topn(txt)
    sort_order = parse_sort(txt)

    # special skill: "drivers of (case )?fails"
    if "driver" in txt.lower() and "fail" in txt.lower():
        out = drivers_of_fails(store.get("fpa", pd.DataFrame()), group_by=["ProcessName","TeamName"],
                               month_from=m_from, month_to=m_to, top_n=top_n or 10)
        insights = ["Top segments by fails shown; expand ‘reasons’ to see dominant failure categories."]

        return {"insights": insights, "tables": out, "figs": {}}

    # metric inference (default per domain)
    metric = None
    # try explicit mentions
    for token in ["complaints per 1000", "complaints/1000", "fail rate", "fails", "reviewed", "cases", "avg days", "complaints"]:
        if token in txt.lower():
            metric = resolve_metric(token)
            break
    # sensible defaults
    if not metric:
        metric = {"complaints": "Complaints", "cases": "Unique_Cases", "fpa": "Fail_Rate"}[domain]

    # 2) aggregate
    table, fig = aggregate_generic(domain=domain, metric=metric, group_by=group_by, store=store,
                                   month_from=m_from, month_to=m_to, filters=filters)

    # 3) post-process: topN / sorting
    if not table.empty and (top_n or sort_order):
        value_col = metric if metric in table.columns else table.columns[-1]
        ascending = (sort_order == "asc") if sort_order else False
        # If Month in group_by, sort within latest month
        if "Month" in table.columns and table["Month"].nunique() > 1:
            latest = table["Month"].max()
            sel = table[table["Month"] == latest].sort_values(value_col, ascending=ascending)
            table = sel.groupby([c for c in table.columns if c not in [value_col]]).apply(lambda x: x).reset_index(drop=True)
        table = table.sort_values(value_col, ascending=ascending)
        if top_n:
            table = table.head(top_n)

    # Insights
    insights = []
    if not table.empty:
        insights.append(f"Showing **{metric}** for **{domain}** grouped by {', '.join(group_by) or 'Month'}"
                        + (f" from {m_from} to {m_to}" if m_from and m_to else ""))

    return {
        "insights": insights or ["No data returned — try changing the time window or filters."],
        "tables": {"result": table},
        "figs": {"chart": fig} if fig else {}
    }
