from __future__ import annotations
from pathlib import Path
import pandas as pd
from __future__ import annotations
import pandas as pd

import core.ingest as ingest
from core.join_cases_complaints import build_cases_complaints_join


ROOT = Path(__file__).resolve().parents[1]
PROC_DIR = ROOT / "data" / "processed"

def _read_or_empty(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()

def load_processed(rebuild_if_missing: bool = True) -> dict[str, pd.DataFrame]:
    comps = _read_or_empty(PROC_DIR / "complaints.parquet")
    cases = _read_or_empty(PROC_DIR / "cases.parquet")
    surv  = _read_or_empty(PROC_DIR / "survey.parquet")

    if not rebuild_if_missing:
        return {"complaints": comps, "cases": cases, "survey": surv}

    if comps.empty or cases.empty or surv.empty:
        return ingest.build_processed_datasets()

    return {"complaints": comps, "cases": cases, "survey": surv}

def latest_month(df: pd.DataFrame) -> str | None:
    if df is None or df.empty or "month" not in df.columns: return None
    vals = df["month"].dropna().astype(str).tolist()
    return max(vals) if vals else None

def available_months(*dfs: pd.DataFrame) -> list[str]:
    months = set()
    for d in dfs:
        if d is not None and not d.empty and "month" in d.columns:
            months.update(d["month"].dropna().astype(str).tolist())
    return sorted(months)

# ---------- NEW: common-month helpers ----------
def _to_month_set(df: pd.DataFrame) -> set[str]:
    if df is None or df.empty or "month" not in df.columns: return set()
    return set(df["month"].dropna().astype(str).tolist())

def common_months(*dfs: pd.DataFrame) -> list[str]:
    sets = [ _to_month_set(d) for d in dfs if _to_month_set(d) ]
    if not sets: return []
    common = set.intersection(*sets) if len(sets) > 1 else sets[0]
    return sorted(common)

def latest_common_month(*dfs: pd.DataFrame) -> str | None:
    cm = common_months(*dfs)
    return cm[-1] if cm else None
