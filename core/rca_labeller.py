# core/rca_labeller.py
from __future__ import annotations
import re
import pandas as pd
from typing import Optional

# very lightweight rules; you can extend these or keep your YAML approach
RULES = [
    # ("label", [patterns...])
    ("Data issue", [r"wrong data", r"incorrect (?:nino|dob|name)", r"mismatch", r"duplicate"]),
    ("Delay / SLA miss", [r"delay", r"late", r"missed sla", r"took too long"]),
    ("Process gap", [r"no process", r"missing step", r"gap", r"not covered"]),
    ("Training / SOP", [r"not trained", r"didn'?t know", r"sop", r"guideline"]),
    ("System / Access", [r"system", r"access", r"d365", r"robotics", r"rpa"]),
    ("Communication", [r"email", r"call", r"not informed", r"poor communication"]),
]

def _first_match(text: str) -> Optional[str]:
    if not text:
        return None
    t = text.lower()
    for label, pats in RULES:
        for p in pats:
            if re.search(p, t):
                return label
    return None

def _std(text: str) -> str:
    return (text or "").strip()

def label_complaints_rca(df: pd.DataFrame) -> pd.DataFrame:
    """
    Builds RCA1 and RCA2 from free text:
      - 'Brief Description - RCA done by admin'
      - 'Why'
    Adds columns if missing; leaves existing values intact.
    """
    if "RCA1" in df.columns and "RCA2" in df.columns:
        return df

    src1 = "Brief Description - RCA done by admin"
    src2 = "Why"

    if src1 not in df.columns and src2 not in df.columns:
        # nothing to label, create empty cols
        df["RCA1"] = pd.NA
        df["RCA2"] = pd.NA
        return df

    t1 = df.get(src1, "").astype(str)
    t2 = df.get(src2, "").astype(str)

    df["RCA1"] = (t1.fillna("") + " " + t2.fillna("")).map(_first_match)
    # simple RCA2: second pass by blanking the first label term (keeps light)
    def rca2(row):
        combined = f"{row.get(src1,'')} {row.get(src2,'')}".lower()
        l1 = row.get("RCA1")
        if not combined or not l1:
            return None
        # remove any words from label and try again
        cleaned = re.sub("|".join([re.escape(w) for w in l1.lower().split()]), " ", combined)
        return _first_match(cleaned)

    if "RCA2" not in df.columns:
        df["RCA2"] = df.apply(rca2, axis=1)

    # tidy
    df["RCA1"] = df["RCA1"].astype("string")
    df["RCA2"] = df["RCA2"].astype("string")
    return df
