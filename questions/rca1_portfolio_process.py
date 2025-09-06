# rca1_portfolio_process.py
import pandas as pd
import numpy as np

def _auto_label_from_comment(s: pd.Series) -> pd.Series:
    s = s.fillna("").str.lower()
    return np.select(
        [
            s.str.contains("manual", na=False),
            s.str.contains("standard|timescale", na=False),
            s.str.contains("scheme|rules", na=False),
            s.str.contains("pension set up|set up", na=False),
            s.str.contains("requirement not checked|not checked", na=False),
            s.str.contains("case not created", na=False),
        ],
        ["Manual", "Standard Timescale", "Scheme Rules", "Pension set up", "Requirement not checked", "Case not created"],
        default="Other",
    )

def run(store, params=None, user_text=""):
    comp = store["complaints"].copy()
    if comp.empty:
        return pd.DataFrame(columns=["portfolio","process","rca1","complaints"])

    # ensure RCA labels present
    if "rca1" not in comp.columns or comp["rca1"].isna().all():
        comp["rca1"] = _auto_label_from_comment(comp.get("comment"))

    # window: last 3 months from latest complaint
    last = comp["month_dt"].max()
    if pd.isna(last):
        return pd.DataFrame(columns=["portfolio","process","rca1","complaints"])
    window_start = (last.to_period("M") - 2).to_timestamp()
    df = comp[(comp["month_dt"]>=window_start) & (comp["month_dt"]<=last)]

    # filters
    if params:
        p = params.get("portfolio")
        if p:
            df = df[df["portfolio"].eq(str(p).lower().strip())]
        pr = params.get("process")
        if pr:
            df = df[df["process"].eq(str(pr).lower().strip())]

    out = (df.groupby(["portfolio","process","rca1"])
             .agg(complaints=("complaint_id","nunique"))
             .reset_index()
             .sort_values(["portfolio","process","complaints"], ascending=[True,True,False]))
    return out
