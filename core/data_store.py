from __future__ import annotations
from pathlib import Path
import pandas as pd
from . import ingest

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

