# kpi/kpi_complaints.py
from __future__ import annotations
from typing import Dict, Iterable, Optional, Tuple, List
from pathlib import Path
import re
import pandas as pd
import numpy as np

try:
    import yaml  # optional, for pattern config
    _HAVE_YAML = True
except Exception:
    _HAVE_YAML = False


# -------------------- RCA Standardization --------------------

_DEFAULT_RCA_PATTERNS = {
    # RCA1: { RCA2_label: [regexes...] }
    "Data entry": {
        "Wrong entry": [r"\bwrong data\b", r"\bincorrect\b", r"\btypo\b", r"\bmis-?key(ed)?\b"],
        "Missing/blank field": [r"\bmissing\b", r"\bblank\b", r"\bnot provided\b"],
    },
    "Documentation": {
        "Incorrect document": [r"\bwrong doc(ument)?\b", r"\bincorrect doc\b", r"\binvalid doc\b"],
        "Document missing": [r"\bmissing doc\b", r"\bno document\b"],
    },
    "Process": {
        "Followed wrong process": [r"\bwrong process\b", r"\bincorrect process\b", r"\bnot as per process\b"],
        "Training gap": [r"\btraining\b", r"\bknowledge gap\b"],
        "SLA miss": [r"\bsla\b", r"\bbreach\b"],
    },
    "System": {
        "System error": [r"\bsystem (error|issue|down)\b", r"\btool (issue|error)\b"],
        "Integration": [r"\bintegration\b", r"\binterface\b"],
    },
    "Customer provided": {
        "Incorrect info from customer": [r"\bcustomer.*(wrong|incorrect)\b", r"\bprovided wrong\b"],
        "Missing info from customer": [r"\bcustomer.*missing\b", r"\bawaiting info\b"],
    },
    "Control": {
        "Control failure": [r"\bcontrol (gap|fail(ure)?)\b", r"\bno control\b"],
    },
}

def _load_rca_patterns(cfg_path: str | Path = "data/rca_patterns.yml") -> Dict[str, Dict[str, List[str]]]:
    p = Path(cfg_path)
    if _HAVE_YAML and p.exists():
        with open(p, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        return cfg
    return _DEFAULT_RCA_PATTERNS

def standardize_rca(text: str, patterns: Dict[str, Dict[str, List[str]]]) -> Tuple[str, str]:
    """
    Map free-text RCA to (RCA1, RCA2) via regex patterns.
    If no pattern hits, return ("Other", top_keyword) using a simple fallback.
    """
    t = (text or "").lower()
    # strict matching first
    for rca1, buckets in patterns.items():
        for rca2, regs in buckets.items():
            for rg in regs:
                if re.search(rg, t, flags=re.IGNORECASE):
                    return rca1, rca2

    # fallback: pull top token/bigram as hint
    toks = re.findall(r"[a-z]{3,}", t)
    stop = {"the","and","for","with","from","that","this","were","have","has","had","not","but","are","was"}
    toks = [w for w in toks if w not in stop]
    top = ""
    if toks:
        # simple frequency
        from collections import Counter
        top = Counter(toks).most_common(1)[0][0]
    return "Other", (top or "Unclassified").title()


def add_rca_labels(complaints_df: pd.DataFrame, cfg_path: str | Path = "data/rca_patterns.yml") -> pd.DataFrame:
    if complaints_df.empty:
        return complaints_df
    pats = _load_rca_patterns(cfg_path)
    df = complaints_df.copy()
    if "RCA_Text" not in df.columns:
        df["RCA_Text"] = (df.get("RCA_Brief","").astype(str) + " " + df.get("RCA_Why","").astype(str)).str.strip()

    rca1, rca2 = [], []
    for t in df["RCA_Text"].fillna(""):
        a,b = standardize_rca(t, pats)
        rca1.append(a); rca2.append(b)
    df["RCA1"] = rca1
    df["RCA2"] = rca2
    return df


# -------------------- Optional mappings --------------------

def load_parentcase_to_process(map_path: str | Path = "data/mappings/parent_case_to_process.csv") -> pd.DataFrame:
    """
    Optional mapping table: columns [ParentCaseType, ProcessName]
    If not present, return empty DF; we'll keep ParentCaseType as its own dim.
    """
    p = Path(map_path)
    if not p.exists():
        return pd.DataFrame(columns=["ParentCaseType","ProcessName"])
    m = pd.read_csv(p)
    m = m[["ParentCaseType","ProcessName"]].dropna().copy()
    m["ParentCaseType"] = m["ParentCaseType"].astype(str).str.strip()
    m["ProcessName"] = m["ProcessName"].astype(str).str.strip()
    return m


# -------------------- Aggregations --------------------

def complaints_summary(
    complaints_df: pd.DataFrame,
    group_by: Iterable[str] = ("ParentCaseType", "report_month_ym", "Portfolio_std", "Scheme",
                               "ReceiptMethod", "ParentTeam", "AptiaError", "Control", "RCA1", "RCA2"),
) -> pd.DataFrame:
    """
    Count complaints at requested grain (default includes all dims + RCA1/2).
    """
    if complaints_df.empty:
        return pd.DataFrame()

    gb = [c for c in group_by if c in complaints_df.columns]
    if not gb:
        gb = ["ParentCaseType", "report_month_ym"]

    out = (
        complaints_df
        .groupby(gb, dropna=False)
        .size()
        .reset_index(name="Complaints")
        .sort_values(gb)
    )
    return out


def complaints_with_process(
    complaints_df: pd.DataFrame,
    mapping_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Attach ProcessName via mapping if provided; otherwise just keep ParentCaseType.
    """
    if mapping_df is None or mapping_df.empty:
        return complaints_df.copy()
    df = complaints_df.copy()
    df = df.merge(mapping_df, on="ParentCaseType", how="left")  # may be NaN if not mapped
    return df


def complaints_per_1000(
    cases_df: pd.DataFrame,
    complaints_df: pd.DataFrame,
    map_parent_to_process: pd.DataFrame | None = None,
    denom_group: Iterable[str] = ("Portfolio_std", "Scheme", "ParentTeam"),  # ParentTeam will align to TeamName (see note)
    numer_group: Iterable[str] = ("ParentCaseType", "report_month_ym", "Portfolio_std", "Scheme", "ParentTeam"),
    join_on_process: bool = True,
) -> pd.DataFrame:
    """
    Compute Complaints per 1,000 Cases by month and business dims.

    Join keys:
      - month: cases.month_ym  <-> complaints.report_month_ym
      - dims: overlap between denom_group & numer_group
      - if `join_on_process` and mapping available, also align ParentCaseType -> ProcessName
        (this gives a ProcessName column in the output).

    Returns columns:
      dims..., report_month_ym, Complaints, Unique_Cases, Complaints_per_1000
    """
    if cases_df.empty or complaints_df.empty:
        return pd.DataFrame()

    # ensure month keys exist
    if "month_ym" not in cases_df.columns:
        raise ValueError("cases_df must contain 'month_ym' (derived from Create_Date).")
    if "report_month_ym" not in complaints_df.columns:
        raise ValueError("complaints_df must contain 'report_month_ym' (derived from Report_Date).")

    # Optionally attach ProcessName via mapping
    comp = complaints_df.copy()
    if join_on_process and map_parent_to_process is not None and not map_parent_to_process.empty:
        comp = comp.merge(map_parent_to_process, on="ParentCaseType", how="left")

    # --- Numerator (Complaints) ---
    numer_gb = [c for c in numer_group if c in comp.columns]
    numer = comp.groupby([*numer_gb], dropna=False).size().reset_index(name="Complaints")

    # --- Denominator (Unique Cases) ---
    # We’ll align ParentTeam (complaints) with TeamName (cases) by simple text match
    cases = cases_df.copy()
    if "TeamName" in cases.columns:
        cases["TeamName"] = cases["TeamName"].astype(str).str.strip()
    # Provide a lightweight alias to join by team if requested
    if "ParentTeam" in numer_gb and "TeamName" in cases.columns:
        cases["ParentTeam"] = cases["TeamName"]

    denom_gb = [c for c in denom_group if c in cases.columns]
    # Always include month in both sides for the join window
    denom = (
        cases
        .dropna(subset=["Case_ID"])
        .drop_duplicates(subset=["Case_ID"] + ([c for c in denom_gb if c != "ParentTeam"] if denom_gb else []))
        .groupby([*denom_gb, "month_ym"], dropna=False)["Case_ID"]
        .nunique()
        .reset_index(name="Unique_Cases")
    )

    # --- Join numerator to denominator on overlapping dims + month ---
    # Identify common dim columns across numer & denom
    common_dims = [c for c in numer_gb if c in denom_gb]
    left = numer.rename(columns={"report_month_ym": "month_ym"})
    join_cols = [*common_dims, "month_ym"]
    out = left.merge(denom, on=join_cols, how="left")

    # Sometimes denom might be missing (no cases) → leave NaN; avoid div/0
    out["Complaints_per_1000"] = (out["Complaints"] / out["Unique_Cases"].replace({0: np.nan})) * 1000.0
    out["Complaints_per_1000"] = out["Complaints_per_1000"].round(2)

    # restore report_month_ym name for clarity
    out = out.rename(columns={"month_ym": "report_month_ym"})

    # order nicely
    sel = [*join_cols]
    if "ParentCaseType" in out.columns and "ParentCaseType" not in sel:
        sel.insert(0, "ParentCaseType")
    if "ProcessName" in out.columns and "ProcessName" not in sel:
        sel.insert(1, "ProcessName")
    sel = [c for c in sel if c in out.columns]
    return out.sort_values(sel).reset_index(drop=True)
