from __future__ import annotations
import re, glob
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
CASES_DIR = DATA_DIR / "cases"
PROC_DIR = DATA_DIR / "processed"
PROC_DIR.mkdir(parents=True, exist_ok=True)

def _to_month_str(v):
    if pd.isna(v): return None
    d = pd.to_datetime(v, errors="coerce", dayfirst=True)
    if pd.isna(d): return None
    return d.to_period("M").strftime("%Y-%m")

def _pick_first(df: pd.DataFrame, preferred: list[str], contains_any: list[str] | None = None):
    # exact
    for c in preferred:
        if c in df.columns: return c
    # case-insensitive exact
    low_map = {c.lower(): c for c in df.columns}
    for c in preferred:
        if c.lower() in low_map: return low_map[c.lower()]
    # substring
    if contains_any:
        for c in df.columns:
            cl = c.lower()
            if any(sub in cl for sub in contains_any): return c
    return None

def _std_portfolio(s):
    if pd.isna(s): return s
    t = str(s).strip().lower()
    t = t.replace("leatherhead - baes", "baes leatherhead")
    t = t.replace("baes-leatherhead", "baes leatherhead")
    t = t.replace("north west", "northwest")
    return t.title()

def _load_complaints() -> pd.DataFrame:
    pats = [
        str(DATA_DIR / "complaints" / "*.xls*"),
        str(DATA_DIR / "*.xls*"),  # fallback if someone drops at top-level
    ]
    files = []
    for p in pats: files += glob.glob(p)
    if not files: return pd.DataFrame()

    frames = []
    for f in sorted(files):
        try:
            df = pd.read_excel(f)
        except Exception:
            continue
        if df.empty: continue

        date_col = _pick_first(
            df,
            preferred=[
                "Date Complaint Received - DD/MM/YY", "Date Complaint Received",
                "Complaint Received Date", "Received Date", "Date", "Complaint Date"
            ],
            contains_any=["date","received"]
        )
        if date_col:
            df["month"] = df[date_col].apply(_to_month_str)
        else:
            df["month"] = np.nan

        port_col = _pick_first(
            df, preferred=["Portfolio","Portfolio Name"],
            contains_any=["portfolio","scheme","office","site","location","team"]
        )
        if port_col is None:
            port_col = "Portfolio"
            df[port_col] = np.nan

        # optional reason column if present
        reason_col = _pick_first(df, preferred=["Reason","Complaint Reason"], contains_any=["reason","category"])
        if reason_col is None and "reason" not in df.columns:
            df["Reason"] = np.nan
        elif reason_col and reason_col != "Reason":
            df = df.rename(columns={reason_col:"Reason"})

        df["Portfolio_std"] = df[port_col].apply(_std_portfolio)
        frames.append(df[["month","Portfolio_std","Reason"]].copy())

    if not frames: return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out = out.dropna(subset=["month"])
    return out

def _load_cases() -> pd.DataFrame:
    files = glob.glob(str(CASES_DIR / "*.xls*"))
    if not files: return pd.DataFrame()
    frames = []
    for f in sorted(files):
        try:
            d = pd.read_excel(f)
        except Exception:
            continue
        if "Case ID" not in d.columns: continue

        mcol = _pick_first(d, preferred=["Report_Date","Month"], contains_any=["date","month"])
        if mcol:
            d["month"] = d[mcol].apply(_to_month_str)
        elif "month" not in d.columns:
            d["month"] = np.nan

        port_col = _pick_first(
            d, preferred=["Portfolio","Portfolio Name"],
            contains_any=["portfolio","scheme","office","site","location","team"]
        )
        if port_col is None:
            port_col = "Portfolio"
            d[port_col] = np.nan

        d["Portfolio_std"] = d[port_col].apply(_std_portfolio)
        d = d.dropna(subset=["Case ID"])
        d["Case ID"] = d["Case ID"].astype(str)
        frames.append(d[["Case ID","month","Portfolio_std"]].copy())

    if not frames: return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out = out.dropna(subset=["month"])
    out = out.drop_duplicates(subset=["month","Case ID"], keep="first")
    return out

def _load_survey() -> pd.DataFrame:
    pats = [
        str(DATA_DIR / "surveys" / "*.xls*"),
        str(DATA_DIR / "*.xls*"),
    ]
    files = []
    for p in pats: files += glob.glob(p)
    # prefer likely survey files
    files = [f for f in files if re.search(r"survey|overall raw data|nps", f, re.I)]
    if not files: return pd.DataFrame()

    frames = []
    for f in sorted(files):
        try:
            df = pd.read_excel(f)
        except Exception:
            continue
        if df.empty: continue

        # month
        mcol = _pick_first(df, preferred=["month","Month"], contains_any=["month","date"])
        if mcol:
            df["month"] = df[mcol].apply(_to_month_str)
        elif "month" not in df.columns:
            df["month"] = np.nan

        # portfolio
        port_col = _pick_first(
            df, preferred=["Portfolio","Portfolio Name"],
            contains_any=["portfolio","scheme","office","site","location","team"]
        )
        if port_col is None:
            port_col = "Portfolio"; df[port_col] = np.nan
        df["Portfolio_std"] = df[port_col].apply(_std_portfolio)

        # NPS direct or derive
        nps_col = _pick_first(df, preferred=["NPS","Net Promoter Score"], contains_any=["nps"])
        if nps_col:
            frames.append(df[["month","Portfolio_std",nps_col]].rename(columns={nps_col:"NPS"}))
            continue

        low = {c.lower(): c for c in df.columns}
        prom = next((low[k] for k in low if k in {"promoters","promoters_count"} or ("promoter" in k and "count" in k)), None)
        detr = next((low[k] for k in low if k in {"detractors","detractors_count"} or ("detractor" in k and "count" in k)), None)
        pasv = next((low[k] for k in low if k in {"passives","passives_count"} or ("passive" in k and "count" in k)), None)
        if prom and detr and pasv:
            x = df[["month","Portfolio_std",prom,detr,pasv]].copy()
            x["total"] = x[prom] + x[detr] + x[pasv]
            x = x[x["total"] > 0]
            x["NPS"] = ((x[prom] - x[detr]) / x["total"]) * 100.0
            frames.append(x[["month","Portfolio_std","NPS"]])

    if not frames: return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out = out.dropna(subset=["month"])
    # if multiple sources overlap, keep mean NPS per month/portfolio
    out = out.groupby(["month","Portfolio_std"], dropna=False)["NPS"].mean().reset_index()
    return out

def build_processed_datasets() -> dict[str, pd.DataFrame]:
    comp = _load_complaints()
    cases = _load_cases()
    surv  = _load_survey()

    if not comp.empty:
        comp.to_parquet(PROC_DIR / "complaints.parquet", index=False)
    if not cases.empty:
        cases.to_parquet(PROC_DIR / "cases.parquet", index=False)
    if not surv.empty:
        surv.to_parquet(PROC_DIR / "survey.parquet", index=False)

    return {"complaints": comp, "cases": cases, "survey": surv}

