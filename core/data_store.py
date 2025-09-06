# data_store.py
from __future__ import annotations

from pathlib import Path
import re
import pandas as pd


# Where your folders live in Streamlit Cloud
DATA_DIR = Path("/mount/src/haloquality/data")  # change if yours is different

def _read_any(path: Path, sheet_name=0) -> pd.DataFrame:
    """Read Excel (first sheet) or Parquet with a tiny PQ cache next to the file."""
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    pq = path.with_suffix(".parquet")
    if pq.exists() and pq.stat().st_mtime >= path.stat().st_mtime:
        return pd.read_parquet(pq)
    df = pd.read_excel(path, sheet_name=sheet_name, engine="openpyxl")
    try:
        df.to_parquet(pq, index=False)
    except Exception:
        pass
    return df


def _coerce_month(df: pd.DataFrame, candidates: list[str]) -> pd.Series:
    """
    Find the first date-like column in `candidates` that exists in df, coerce to datetime,
    then return the month start (Timestamp). Never returns object dtype.
    """
    for col in candidates:
        if col in df.columns:
            s = pd.to_datetime(df[col], errors="coerce", dayfirst=True)
            if s.notna().any():
                return s.dt.to_period("M").dt.to_timestamp()
    # If nothing matched, try any column containing 'date'
    for col in df.columns:
        if re.search(r"date", col, re.I):
            s = pd.to_datetime(df[col], errors="coerce", dayfirst=True)
            if s.notna().any():
                return s.dt.to_period("M").dt.to_timestamp()
    # Fallback: all-NaT month series
    return pd.to_datetime(pd.Series([pd.NaT] * len(df))).dt.to_period("M").dt.to_timestamp()


def _clean_str(s: pd.Series) -> pd.Series:
    return (
        s.astype(str)
        .str.strip()
        .replace({"nan": None, "NaN": None, "None": None})
    )


def load_cases(dir_path: Path) -> pd.DataFrame:
    """Load & normalize cases."""
    files = sorted(list(dir_path.glob("*.xlsx"))) + sorted(list(dir_path.glob("*.parquet")))
    frames = []
    for f in files:
        df = _read_any(f)
        # Normalize column names (case-insensitive, keep originals if not found)
        cols = {c.lower(): c for c in df.columns}
        # IDs
        case_id = cols.get("case id") or next((c for c in df.columns if re.fullmatch(r"(?i)case\s*id", c)), None)
        if case_id is None:
            continue
        df = df.rename(columns={case_id: "Case ID"})
        # Process & Portfolio
        proc_col = cols.get("process name") or next((c for c in df.columns if re.search(r"(?i)^process(\s|_)name$", c)), None)
        if proc_col:
            df = df.rename(columns={proc_col: "Process"})
        port_col = cols.get("portfolio") or next((c for c in df.columns if re.fullmatch(r"(?i)portfolio", c)), None)
        if port_col:
            df = df.rename(columns={port_col: "Portfolio"})
        # Month
        df["_month_dt"] = _coerce_month(
            df,
            [
                "Case Created Date",
                "Created Date",
                "Open Date",
                "Start Date",
                "Report Date",
            ],
        )
        frames.append(df)

    if not frames:
        return pd.DataFrame(columns=["Case ID", "Process", "Portfolio", "_month_dt"])

    out = pd.concat(frames, ignore_index=True)
    out["Case ID"] = _clean_str(out["Case ID"])
    if "Process" in out.columns:
        out["Process"] = _clean_str(out["Process"])
    else:
        out["Process"] = None
    if "Portfolio" in out.columns:
        out["Portfolio"] = _clean_str(out["Portfolio"])
    else:
        out["Portfolio"] = None
    # Guarantee datetime dtype
    out["_month_dt"] = pd.to_datetime(out["_month_dt"], errors="coerce")
    return out[["Case ID", "Process", "Portfolio", "_month_dt"]]


def load_complaints(dir_path: Path, cases: pd.DataFrame) -> pd.DataFrame:
    """Load & normalize complaints; map Process and Portfolio per your rules."""
    files = sorted(list(dir_path.glob("*.xlsx"))) + sorted(list(dir_path.glob("*.parquet")))
    frames = []
    for f in files:
        df = _read_any(f)
        cols = {c.lower(): c for c in df.columns}

        # The two key columns you specified
        parent_type = cols.get("parent case type") or next((c for c in df.columns if re.fullmatch(r"(?i)parent\s*case\s*type", c)), None)
        affected_id = cols.get("original process affected case id") or next((c for c in df.columns if re.search(r"(?i)affected.*case\s*id", c)), None)
        if not affected_id:
            # Try common alternates
            affected_id = next((c for c in df.columns if re.fullmatch(r"(?i)original\s*case\s*id", c)), None)

        if parent_type:
            df = df.rename(columns={parent_type: "Process"})
        else:
            df["Process"] = None

        if affected_id:
            df = df.rename(columns={affected_id: "Case ID"})
        else:
            df["Case ID"] = None

        # Month (the sheet label you shared earlier)
        date_col = cols.get("report date dd/mm/yy format only") or cols.get("report date") or \
                   next((c for c in df.columns if re.search(r"(?i)report.*date", c)), None)
        df["_month_dt"] = _coerce_month(df, [date_col] if date_col else [])

        frames.append(df)

    if not frames:
        return pd.DataFrame(columns=["Case ID", "Process", "Portfolio", "_month_dt"])

    cmp = pd.concat(frames, ignore_index=True)
    cmp["Case ID"] = _clean_str(cmp["Case ID"])
    cmp["Process"] = _clean_str(cmp["Process"])
    cmp["_month_dt"] = pd.to_datetime(cmp["_month_dt"], errors="coerce")

    # Join to cases to fetch Portfolio (and optionally backstop Process if missing)
    if not cases.empty:
        joined = cmp.merge(
            cases[["Case ID", "Process", "Portfolio"]],
            on="Case ID",
            how="left",
            suffixes=("", "_cases"),
        )
        # Process preference: Parent Case Type (complaints) if present, else cases Process
        joined["Process"] = joined["Process"].where(joined["Process"].notna() & (joined["Process"] != ""), joined["Process_cases"])
        joined["Portfolio"] = joined["Portfolio"]  # already from cases merge
        cmp = joined.drop(columns=[c for c in ["Process_cases"] if c in joined.columns])

    # Final shape
    return cmp[["Case ID", "Process", "Portfolio", "_month_dt"]]


def load_store() -> dict:
    """Entry point used by the app."""
    cases_dir = DATA_DIR / "cases"
    complaints_dir = DATA_DIR / "complaints"
    fpa_dir = DATA_DIR / "first_pass_accuracy"  # optional

    cases = load_cases(cases_dir) if cases_dir.exists() else pd.DataFrame(columns=["Case ID","Process","Portfolio","_month_dt"])
    complaints = load_complaints(complaints_dir, cases) if complaints_dir.exists() else pd.DataFrame(columns=["Case ID","Process","Portfolio","_month_dt"])

    store = {
        "cases": cases,
        "complaints": complaints,
        "cases_rows": int(cases.shape[0]),
        "complaints_rows": int(complaints.shape[0]),
    }
    return store
