# core/fpa_labeller.py
from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Tuple
import re
import yaml
import pandas as pd

# Default patterns (used if data/fpa_patterns.yml is missing)
DEFAULT_PATTERNS: Dict[str, List[str]] = {
    "data_error": [
        r"\bwrong data\b", r"\bincorrect\b", r"\btypo\b", r"\bspelling\b",
        r"\bmis-?key(ed)?\b", r"\btranspos(e|ed)\b"
    ],
    "missing_document": [
        r"\bmissing doc(ument)?\b", r"\bno document\b", r"\bdoc not (received|available)\b"
    ],
    "sla_delay": [
        r"\bdelay(ed)?\b", r"\blate\b", r"\bmiss(ed)? sla\b", r"\bbreach(ed)? sla\b"
    ],
    "process_noncompliance": [
        r"\bdid not follow\b", r"\bnot as per process\b", r"\bprocess deviation\b", r"\bnon-?compliance\b"
    ],
    "wrong_calculation": [
        r"\bcalc(ulation)? error\b", r"\bwrong calc\b", r"\bincorrect amount\b", r"\bamount mismatch\b"
    ],
    "communication": [
        r"\bwrong email\b", r"\bnot informed\b", r"\bincorrect note\b", r"\bno update\b"
    ],
}

def _load_patterns(path: Path) -> Dict[str, List[str]]:
    if not path.exists():
        return DEFAULT_PATTERNS
    with open(path, "r", encoding="utf-8") as f:
        y = yaml.safe_load(f) or {}
    cats = y.get("categories") or y  # support both shapes
    # ensure list of regex strings
    out = {}
    for k, v in cats.items():
        if isinstance(v, list):
            out[k] = v
    return out or DEFAULT_PATTERNS

def _compile_patterns(patterns: Dict[str, List[str]]) -> Dict[str, List[re.Pattern]]:
    compiled = {}
    for cat, pats in patterns.items():
        compiled[cat] = [re.compile(p, re.IGNORECASE) for p in pats]
    return compiled

def label_fpa_comments(
    fpa_df: pd.DataFrame,
    patterns_file: str | Path = "data/fpa_patterns.yml",
    comment_col: str = "CaseComment",
    fail_flag_col: str = "FailFlag"
) -> pd.DataFrame:
    """
    Add FPA labels for failed rows using regex/keyword patterns.
    Adds:
      - FPA_AllTags: ';'-joined list of matched categories
      - FPA_PrimaryTag: first matched category (or 'other')
    """
    if fpa_df is None or fpa_df.empty:
        return fpa_df

    pats = _load_patterns(Path(patterns_file))
    comp = _compile_patterns(pats)

    df = fpa_df.copy()
    df["FPA_AllTags"] = ""
    df["FPA_PrimaryTag"] = ""

    if comment_col not in df.columns:
        return df

    def tag_row(txt: str) -> Tuple[str, str]:
        if not isinstance(txt, str) or not txt.strip():
            return "", ""
        matches = []
        for cat, regs in comp.items():
            for rg in regs:
                if rg.search(txt):
                    matches.append(cat)
                    break  # one hit per category is enough
        if not matches:
            return "", "other"
        return ";".join(matches), matches[0]

    mask = df[fail_flag_col] if fail_flag_col in df.columns else True
    df.loc[mask, ["FPA_AllTags","FPA_PrimaryTag"]] = df.loc[mask, comment_col].apply(
        lambda x: pd.Series(tag_row(x))
    )

    return df
