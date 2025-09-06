# unique_cases_mom.py
import pandas as pd

def run(store, params=None, user_text=""):
    cases = store["cases"].copy()
    if cases.empty:
        return pd.DataFrame(columns=["_month","unique_cases"])

    # parse window
    if params and params.get("start_month") and params.get("end_month"):
        sm = pd.to_datetime(params["start_month"]).to_period("M").to_timestamp()
        em = pd.to_datetime(params["end_month"]).to_period("M").to_timestamp()
    else:
        last = cases["month_dt"].max()
        sm = (last.to_period("M")-2).to_timestamp()
        em = last

    df = cases[(cases["month_dt"]>=sm) & (cases["month_dt"]<=em)]
    g = (df.groupby("month_dt")
           .agg(unique_cases=("case_id","nunique"))
           .reset_index()
           .sort_values("month_dt"))
    g["_month"] = g["month_dt"].dt.strftime("%b %y")
    return g[["_month","unique_cases"]]
