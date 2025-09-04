import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml
import pandas as pd
from jinja2 import Template

from . import blocks

@dataclass
class KPIResult:
    df: pd.DataFrame
    meta: Dict[str, Any]

def _deep_replace(val, params: Dict[str, Any]):
    if isinstance(val, str):
        if val.startswith("$"):
            key = val[1:]
            return params.get(key)
        return val
    if isinstance(val, list):
        return [_deep_replace(v, params) for v in val]
    if isinstance(val, dict):
        return {k: _deep_replace(v, params) for k, v in val.items()}
    return val

def _prev_month(month: str) -> str:
    return (pd.Period(month, freq="M") - 1).strftime("%Y-%m")

def _call_func(func_path: str, kwargs: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    module_path, func_name = func_path.rsplit(".", 1)
    mod = importlib.import_module(module_path)
    fn = getattr(mod, func_name)

    res = fn(**kwargs)
    if isinstance(res, tuple) and len(res) == 2 and isinstance(res[0], pd.DataFrame):
        second = res[1]
        meta = {}
        if isinstance(second, str) and len(second) == 7 and second[4] == "-":
            meta["prev_month"] = second
        elif isinstance(second, str):
            meta["info"] = second
        elif isinstance(second, dict):
            meta.update(second)
        return res[0], meta
    elif isinstance(res, tuple) and len(res) == 3 and isinstance(res[0], pd.DataFrame):
        meta = {}
        if isinstance(res[1], str):
            meta["prev_month"] = res[1]
        if isinstance(res[2], dict):
            meta.update(res[2])
        return res[0], meta
    elif isinstance(res, pd.DataFrame):
        return res, {}
    else:
        try:
            df = pd.DataFrame(res)
            return df, {}
        except Exception:
            return pd.DataFrame(), {"raw": res}

def run_question(spec_path: Path, params: Dict[str, Any], *, store_data: Dict[str, Any]) -> Dict[str, Any]:
    spec = yaml.safe_load(spec_path.read_text())

    defaults = spec.get("defaults", {})
    merged_params = {**defaults, **params}

    # normalize group_by to list
    gb = merged_params.get("group_by")
    if gb is None:
        merged_params["group_by"] = []
    elif isinstance(gb, str):
        merged_params["group_by"] = [c.strip() for c in gb.split(",") if c.strip()]

    results: Dict[str, KPIResult] = {}
    for call in (spec.get("kpi_calls") or []):
        call_id = call["id"]
        func_path = call["func"]
        raw_args = call.get("args", {})
        args = _deep_replace(raw_args, merged_params)

        # inject store data by convention
        if "mom_overview" in func_path or "top_contributors" in func_path:
            args = {"complaints_df": store_data["complaints_df"],
                    "cases_df": store_data["cases_df"],
                    "survey_df": store_data["survey_df"],
                    **args}
        elif "reason_mix_percent" in func_path or "complaint_heatmap" in func_path or "reason_drilldown" in func_path:
            args = {"complaints_df": store_data["complaints_df"], **args}

        df, meta = _call_func(func_path, args)
        results[call_id] = KPIResult(df=df, meta=meta)

    # Build blocks
    payload_cards = {}
    prev_month = None
    if "mom" in results:
        payload_cards = blocks.make_metric_cards(results["mom"].df)
        prev_month = results["mom"].meta.get("prev_month") or _prev_month(merged_params["month"])

    out_cards = [{"name": "headline", "title": "Headline metrics", "data": payload_cards}] if payload_cards else []

    out_tables = []
    out_charts = []

    def resolve_token(tok: Any):
        if isinstance(tok, str) and tok.startswith("$group_by[") and tok.endswith("]"):
            idx = int(tok[len("$group_by["):-1])
            gb = merged_params.get("group_by", [])
            return gb[idx] if 0 <= idx < len(gb) else None
        return tok

    for b in (spec.get("layout", {}).get("blocks") or []):
        btype = b.get("type")
        source_id = b.get("source")
        res = results.get(source_id)
        if res is None:
            continue

        if btype == "table":
            cols = [resolve_token(c) for c in (b.get("columns") or [])]
            cols = [c for c in cols if c]
            out_tables.append({
                "name": b.get("name", source_id),
                "title": b.get("title", source_id),
                "data": blocks.table_from_df(res.df, columns=cols)
            })

        elif btype == "chart":
            out_charts.append(blocks.chart_spec(
                b.get("name", source_id),
                b.get("chart"),
                resolve_token(b.get("x")),
                resolve_token(b.get("y")),
                data_ref=source_id,
                sort=b.get("sort")
            ))

        elif btype == "metric_cards":
            # already added above
            pass

        elif btype == "heatmap":
            out_tables.append({
                "name": b.get("name", "heatmap"),
                "title": b.get("title", "Heatmap"),
                "data": blocks.table_from_df(res.df)
            })

    templ_s = (spec.get("narrative") or {}).get("template", "")
    jenv = {
        "month": merged_params["month"],
        "prev_month": prev_month,
        "group_by": merged_params["group_by"],
        "headline": payload_cards,
        "signed": blocks.signed
    }
    narrative = Template(templ_s).render(**jenv) if templ_s else ""

    return {
        "id": spec.get("id"),
        "version": spec.get("version", 1),
        "params": {"month": merged_params["month"], "group_by": merged_params["group_by"]},
        "insights": narrative.strip(),
        "cards": out_cards,
        "tables": out_tables,
        "charts": out_charts,
        "dataRefs": {k: v.df.to_dict(orient="records") for k, v in results.items()}
    }
