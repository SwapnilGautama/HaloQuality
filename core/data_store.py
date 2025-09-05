# core/data_store.py
from __future__ import annotations
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
from .rca_labeller import label_complaints_rca

DATA_DIR = Path("data")

# ---------- helpers ----------

def _read_xlsx_all(path: Path, sheet_name=0) -> pd.DataFrame:
    # read_excel handles xlsx robustly; keep it simple and reliable
    return pd.read_excel(path, sheet_name=sheet_name, engine="openpyxl")

def _gather_xlsx(folder: Path) -> List[Path]:
    return sorted([p for p in folder.glob("*.xlsx") if p.is_file()])

def _std_text(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.replace(r"\s+", " ", regex=True)

def _make_month_cols(dt: pd.Series) -> pd.DataFrame:
    # dt is datetime64[ns] with NaT allowed
    mdt = dt.dt.to_period("M").dt.to_timestamp()
    return pd.DataFrame({"month_dt": mdt, "month": mdt.dt.strftime("%b %y")})

def _first_present(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None

def _to_datetime(df: pd.DataFrame, col: str, dayfirst=True) -> pd.Series:
    return pd.to_datetime(df[col], errors="coerce", dayfirst=dayfirst)

def _std_key(s: pd.Series) -> pd.Series:
    return _std_text(s).str.lower()

# ---------- loaders ----------

def load_cases(path: Path) -> pd.DataFrame:
    """
    Cases:
      Date column: 'Create Date'
      Uniques: 'Case ID' (fallback to first id-like column present)
      Keys: Portfolio_std, ProcessName_std
    """
    frames = []
    for f in _gather_xlsx(path):
        df = _read_xlsx_all(f, sheet_name=0)
        if df.empty:
            continue

        # date
        if "Create Date" not in df.columns:
            # try common variants
            alt = _first_present(df, ["Create_Date", "Create date", "Created", "Start Date"])
            if not alt:
                continue
            df.rename(columns={alt: "Create Date"}, inplace=True)

        dt = _to_datetime(df, "Create Date", dayfirst=True)
        mcols = _make_month_cols(dt)
        df = df.join(mcols)

        # portfolio: prefer Portfolio, fallback Site/Location
        pcol = _first_present(df, ["Portfolio", "portfolio", "Site", "Location"])
        if pcol:
            df["Portfolio_std"] = _std_key(df[pcol])
        else:
            df["Portfolio_std"] = ""

        # process name
        pn = _first_present(df, ["Process Name", "Process", "Process_Name", "Processname"])
        if pn:
            df["ProcessName_std"] = _std_key(df[pn])
        else:
            df["ProcessName_std"] = ""

        # id
        idcol = _first_present(df, ["Case ID", "CaseID", "Case Id"])
        if idcol and idcol not in df.columns:
            df.rename(columns={idcol: "Case ID"}, inplace=True)
        elif idcol != "Case ID" and idcol:
            df["Case ID"] = df[idcol]

        frames.append(df)

    if not frames:
        return pd.DataFrame()

    cases = pd.concat(frames, ignore_index=True)
    return cases


def load_complaints(path: Path) -> pd.DataFrame:
    """
    Complaints:
      Date column: 'Date Complaint Received - DD/MM/YY'
      Keys: Portfolio_std (fallback Site), Parent_Case_Type_std
      RCA1/RCA2 computed during load
    """
    frames = []
    for f in _gather_xlsx(path):
        df = _read_xlsx_all(f, sheet_name=0)
        if df.empty:
            continue

        # date (explicit per spec)
        dcol = "Date Complaint Received - DD/MM/YY"
        if dcol not in df.columns:
            # permissive fallback if file has slight header variation
            alt = _first_present(df, [
                "Date Complaint Received - DD/MM/YY",
                "Date Complaint Received",
                "Complaint Received Date",
                "Date Received",
            ])
            if not alt:
                continue
            dcol = alt

        dt = _to_datetime(df, dcol, dayfirst=True)
        mcols = _make_month_cols(dt)
        df = df.join(mcols)

        # portfolio/site
        pcol = _first_present(df, ["Portfolio", "portfolio", "Site", "Location"])
        df["Portfolio_std"] = _std_key(df[pcol]) if pcol else ""

        # parent case type (the join key to cases' process)
        pct = _first_present(df, ["Parent Case Type", "Parent_Case_Type", "Parent case type"])
        df["Parent_Case_Type_std"] = _std_key(df[pct]) if pct else ""

        # ensure RCA labels exist
        df = label_complaints_rca(df)

        frames.append(df)

    if not frames:
        return pd.DataFrame()

    complaints = pd.concat(frames, ignore_index=True)
    return complaints


def load_fpa(path: Path) -> pd.DataFrame:
    """
    First Pass Accuracy:
      Date column: 'Activity Date'
      Label column: 'Review Result' (Pass/Fail)
      Keys: Portfolio_std, ProcessName_std, Team, Manager, Scheme, Location (if available)
    """
    frames = []
    for f in _gather_xlsx(path):
        df = _read_xlsx_all(f, sheet_name=0)
        if df.empty:
            continue

        if "Activity Date" not in df.columns:
            alt = _first_present(df, ["ActivityDate", "Date", "Processed Date"])
            if not alt:
                continue
            df.rename(columns={alt: "Activity Date"}, inplace=True)

        dt = _to_datetime(df, "Activity Date", dayfirst=True)
        mcols = _make_month_cols(dt)
        df = df.join(mcols)

        pcol = _first_present(df, ["Portfolio", "portfolio", "Site", "Location"])
        df["Portfolio_std"] = _std_key(df[pcol]) if pcol else ""

        pn = _first_present(df, ["Process Name", "Process", "Process_Name"])
        df["ProcessName_std"] = _std_key(df[pn]) if pn else ""

        frames.append(df)

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)


def load_checker(path: Path) -> pd.DataFrame:
    """
    Checker accuracy:
      Date column: prefer 'Date Completed', fallback 'Review Date'
    """
    frames = []
    for f in _gather_xlsx(path):
        df = _read_xlsx_all(f, sheet_name=0)
        if df.empty:
            continue

        dcol = "Date Completed" if "Date Completed" in df.columns else _first_present(df, ["Review Date"])
        if not dcol:
            continue

        dt = _to_datetime(df, dcol, dayfirst=True)
        mcols = _make_month_cols(dt)
        df = df.join(mcols)

        pcol = _first_present(df, ["Portfolio", "portfolio", "Site", "Location"])
        df["Portfolio_std"] = _std_key(df[pcol]) if pcol else ""

        pn = _first_present(df, ["Process Name", "Process", "Process_Name"])
        df["ProcessName_std"] = _std_key(df[pn]) if pn else ""

        frames.append(df)

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)


def load_surveys(path: Path) -> pd.DataFrame:
    """
    Surveys:
      Date column: 'Month_received'
    """
    frames = []
    for f in _gather_xlsx(path):
        df = _read_xlsx_all(f, sheet_name=0)
        if df.empty:
            continue

        dcol = "Month_received" if "Month_received" in df.columns else _first_present(df, ["Month Received"])
        if not dcol:
            continue

        dt = _to_datetime(df, dcol, dayfirst=True)
        mcols = _make_month_cols(dt)
        df = df.join(mcols)

        pcol = _first_present(df, ["Portfolio", "portfolio", "Site", "Location"])
        df["Portfolio_std"] = _std_key(df[pcol]) if pcol else ""

        frames.append(df)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


# ---------- master loader ----------

def load_store() -> Dict[str, pd.DataFrame]:
    """
    Loads all tables with normalized keys & months.
    Returns dict usable by questions:
      store["cases"], store["complaints"], store["fpa"], store["checker"], store["surveys"]
    """
    cases = load_cases(DATA_DIR / "cases")
    complaints = load_complaints(DATA_DIR / "complaints")
    fpa = load_fpa(DATA_DIR / "first_pass_accuracy")
    checker = load_checker(DATA_DIR / "checker_accuracy")
    surveys = load_surveys(DATA_DIR / "surveys")

    return {
        "cases": cases,
        "complaints": complaints,
        "fpa": fpa,
        "checker": checker,
        "surveys": surveys,
    }
