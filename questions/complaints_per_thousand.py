# complaints_per_thousand.py
import pandas as pd

def run(store, params=None, user_text=""):
    cases = store["cases"].copy()
    comp  = store["complaints"].copy()

    # Aggregate base tables
    g_cases = (cases.groupby(["portfolio","process","month_dt"])
                    .agg(cases=("case_id","nunique")).reset_index())
    g_comp  = (comp.groupby(["portfolio","process","month_dt"])
                    .agg(complaints=("complaint_id","nunique")).reset_index())

    # Tolerant join
    out = pd.merge(g_cases, g_comp, on=["portfolio","process","month_dt"], how="outer")
    out[["cases","complaints"]] = out[["cases","complaints"]].fillna(0).astype(int)
    out["per_1000"] = (out["complaints"] * 1000 / out["cases"].replace(0, pd.NA)).fillna(0)
    out["month"] = out["month_dt"].dt.strftime("%b %y")

    # Optional filters
    if params:
        p = params.get("portfolio")
        if p:
            out = out[out["portfolio"].eq(str(p).lower().strip())]
        pr = params.get("process")
        if pr:
            out = out[out["process"].eq(str(pr).lower().strip())]
        s = params.get("start_month"); e = params.get("end_month")
        if s and e:
            sm = pd.to_datetime(s).to_period("M").to_timestamp()
            em = pd.to_datetime(e).to_period("M").to_timestamp()
            out = out[(out["month_dt"]>=sm) & (out["month_dt"]<=em)]

    return (out.sort_values(["month_dt","portfolio","process"])
              [["month","process","cases","complaints","per_1000"]])
