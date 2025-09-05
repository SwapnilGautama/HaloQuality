# core/rca_labeller.py
from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Tuple
import re
import yaml
import pandas as pd

# We’ll accept BOTH formats:
# A) simple: { category: [regex, ...], ... }
# B) nested:
#    categories:
#      data_error:
#        patterns: [regex, ...]
#        sub:
#          miskey: [regex, ...]
#          missing_fields: [regex, ...]
#
# RCA1 = top-level category; RCA2 = matched sub-category (if any)

DEFAULT_PATTERNS: Dict = {
    "categories": {
        "data_error": {
            "patterns": [
                r"\bwrong data\b", r"\bincorrect\b", r"\btypo\b",
                r"\bmis-?key(ed)?\b", r"\btranspos(e|ed)\b",
                r"\bdata entry\b"
            ],
            "sub": {
                "miskey": [r"\bmis-?key(ed)?\b", r"\btranspos(e|ed)\b"],
                "missing_fields": [r"\bmissing\b.*\b(field|info|information|value)\b"]
            }
        },
        "missing_document": {
            "patterns": [
                r"\bmissing doc(ument)?\b", r"\bno document\b",
                r"\bdoc not (received|available)\b"
            ]
        },
        "sla_delay": {
            "patterns": [
                r"\bdelay(ed)?\b", r"\blate\b", r"\bmiss(ed)? sla\b", r"\bbreach(ed)? sla\b"
            ]
        },
        "process_noncompliance": {
            "patterns": [
                r"\bdid not follow\b", r"\bnot as per process\b",
                r"\bprocess deviation\b", r"\bnon-?compliance\b"
            ]
        },
        "system_issue": {
            "patterns": [
                r"\bsystem (down|error|issue)\b", r"\bportal (error|issue)\b",
                r"\brobot(ic|ics)\b", r"\brpa\b"
            ]
        },
        "client_input": {
            "patterns": [
                r"\bclient provided wrong\b", r"\bmember wrong\b",
                r"\bincorrect client data\b"
            ]
        },
    }
}

def _load_yaml(path: Path) -> Dict:
    if not path.exists():
        return DEFAULT_PATTERNS
    with open(path, "r", encoding="utf-8") as f:
        y = yaml.safe_load(f) or {}
    # normalise: allow simple flat mapping {cat: [regex]} too
    if "categories" not in y:
        y = {"categories": y}
    cats = y.get("categories") or {}
    norm: Dict[str, Dict] = {}
    for cat, spec in cats.items():
        if isinstance(spec, list):
            norm[cat] = {"patterns": spec, "sub": {}}
        elif isinstance(spec, dict):
            norm[cat] = {
                "patterns": spec.get("patterns", []) if isinstance(spec.get("patterns", []), list) else [],
                "sub": spec.get("sub", {}) if isinstance(spec.get("sub", {}), dict) else {},
            }
        else:
            norm[cat] = {"patterns": [], "sub": {}}
    return {"categories": norm}

def _compile(cats: Dict[str, Dict]) -> Dict[str, Dict[str, List[re.Pattern]]]:
    comp: Dict[str, Dict[str, List[re.Pattern]]] = {}
    for cat, spec in cats.items():
        comp[cat] = {
            "patterns": [re.compile(p, re.IGNORECASE) for p in spec.get("patterns", [])],
            "sub": {sc: [re.compile(p, re.IGNORECASE) for p in pats]
                    for sc, pats in spec.get("sub", {}).items()}
        }
    return comp

def _match(text: str, comp) -> Tuple[str, str, List[str]]:
    """
    Returns: (RCA1, RCA2, all_tags)
    """
    if not isinstance(text, str) or not text.strip():
        return "", "", []
    all_tags: List[str] = []
    primary = ""
    subcat = ""
    for cat, spec in comp.items():
        # any top-level hit?
        if any(r.search(text) for r in spec["patterns"]):
            if not primary:
                primary = cat
            all_tags.append(cat)
            # check subcategories
            for sc, regs in spec["sub"].items():
                if any(r.search(text) for r in regs):
                    if not subcat:
                        subcat = sc
                    all_tags.append(f"{cat}:{sc}")
        else:
            # even if top-level didn't match, allow sub hit to imply top-level
            for sc, regs in spec["sub"].items():
                if any(r.search(text) for r in regs):
                    if not primary:
                        primary = cat
                    if not subcat:
                        subcat = sc
                    all_tags.append(cat)
                    all_tags.append(f"{cat}:{sc}")
                    break
    if not primary:
        primary = "other"
    return primary, subcat, sorted(set(all_tags))

def _ensure_rca_text(df: pd.DataFrame) -> pd.DataFrame:
    """Combine the admin RCA + Why into a single free-text field to label."""
    # Accept many header variants robustly
    candidates = [
        "Brief Description - RCA done by admin", "Brief Description – RCA done by admin",
        "Brief_Description_-_RCA_done_by_admin", "Admin_RCA", "RCA_Admin",
        "Why", "Root Cause", "Root_Cause", "Why?"  # common alternates
    ]
    # Gather any available columns
    cols = [c for c in candidates if c in df.columns]
    if not cols:
        df["RCA_text"] = ""
        return df
    # Build text: join non-null bits with separator
    df["RCA_text"] = (
        df[cols]
        .astype(str)
        .replace({"nan": "", "None": ""})
        .apply(lambda r: " · ".join([x for x in r if x and x.strip()]), axis=1)
        .str.strip()
    )
    return df

def label_complaints_rca(
    complaints_df: pd.DataFrame,
    patterns_file: str | Path = "data/rca_patterns.yml",
    text_col: str = "RCA_text",
    fail_only: bool = False  # if you only want to label rows that meet some boolean flag
) -> pd.DataFrame:
    """
    Add RCA labels to Complaints using free-text fields (Admin RCA + Why).
    Adds: RCA1 (primary), RCA2 (subcategory if matched), RCA_AllTags (joined)
    """
    if complaints_df is None or complaints_df.empty:
        return complaints_df

    df = complaints_df.copy()
    df = _ensure_rca_text(df)

    cfg = _load_yaml(Path(patterns_file))
    comp = _compile(cfg["categories"])

    df["RCA1"] = ""
    df["RCA2"] = ""
    df["RCA_AllTags"] = ""

    # Apply matcher
    def _apply_row(txt: str):
        rca1, rca2, all_tags = _match(txt, comp)
        return pd.Series([rca1, rca2, ";".join(all_tags)], index=["RCA1","RCA2","RCA_AllTags"])

    if fail_only and "FailFlag" in df.columns:
        mask = df["FailFlag"] == True
        df.loc[mask, ["RCA1","RCA2","RCA_AllTags"]] = df.loc[mask, text_col].apply(_apply_row)
    else:
        df[["RCA1","RCA2","RCA_AllTags"]] = df[text_col].apply(_apply_row)

    return df
