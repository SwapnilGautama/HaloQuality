# core/reason_labeller.py
from __future__ import annotations
import re
import pandas as pd

# High-level buckets as per your slide
BUCKETS = {
    "Delay": [
        r"\bdelay\b", r"\bmanual calc", r"\bmanual calculation", r"\bpostal\b",
        r"\b2(nd|nd)? review\b", r"\btimescale", r"\bslow\b", r"\blate\b",
    ],
    "Procedure": [
        r"\bscheme rules?\b", r"\brule\b", r"\bstandard timescale\b", r"\bSLA\b",
        r"\bprocess\b", r"\bprocedure\b",
    ],
    "Communication": [
        r"\bletter\b", r"\bcommunication\b", r"\bnot (informed|told|clear)",
        r"\bno reply\b", r"\bno response\b", r"\bupdate\b",
    ],
    "System": [
        r"\bsystem\b", r"\bit issue\b", r"\bworkflow\b", r"\bplatform\b",
        r"\bbug\b", r"\berror\b",
    ],
    "Incorrect/Incomplete information": [
        r"\bincorrect\b", r"\bwrong\b", r"\bincomplete\b", r"\bmissing\b",
        r"\bnot provided\b", r"\bno evidence\b",
    ],
}

# Sub-reasons from the slide (used only when we can be sure)
SUBREASONS = {
    "Delay Manual calculation": [r"\bmanual calc(ulation)?\b"],
    "Aptia standard Timescale": [r"\bstandard timescale\b", r"\btimescale\b", r"\bSLA\b"],
    "Delay Pension set up": [r"\bpension set up\b", r"\bsetup\b"],
    "Delay Postal Delay": [r"\bpostal\b", r"\bpost\b", r"\bmail\b"],
    "Delay – AVC": [r"\bAVC\b"],
    "Delay Requirement not checked": [r"\brequirement not checked\b", r"\bnot checked\b"],
    "Delay Case not created": [r"\bcase not created\b"],
    "Delay 2nd Review": [r"\b2(nd)? review\b", r"\bsecond review\b"],
    "Delay – Trustee": [r"\btrustee\b"],
    "Scheme Rules": [r"\bscheme rules?\b"],
    "Drop in value/ factor change": [r"\bfactor change\b", r"\bdrop in value\b"],
    "Death benefits payout": [r"\bdeath benefit(s)?\b"],
    "Overpayment": [r"\boverpayment\b"],
    "Pension Increase": [r"\bpension increase\b"],
    "Transfer Documentation": [r"\btransfer doc(umentation)?\b"],
}

def _first_match(text: str, patterns: list[str]) -> bool:
    for pat in patterns:
        if re.search(pat, text, flags=re.I):
            return True
    return False

def label_reasons(df: pd.DataFrame, text_col: str) -> pd.DataFrame:
    """
    Returns df with two new columns:
      - reason_bucket
      - reason_detail (best sub-reason where confidently matched)
    Unmatched → reason_bucket='Other', reason_detail=None
    """
    out = df.copy()
    text = out[text_col].fillna("").astype(str)

    # bucket
    buckets = []
    for t in text:
        t2 = t.lower()
        assigned = None
        for bucket, pats in BUCKETS.items():
            if _first_match(t2, pats):
                assigned = bucket
                break
        buckets.append(assigned or "Other")
    out["reason_bucket"] = buckets

    # sub-reason
    details = []
    for t in text:
        t2 = t.lower()
        chosen = None
        for label, pats in SUBREASONS.items():
            if _first_match(t2, pats):
                chosen = label
                break
        details.append(chosen)
    out["reason_detail"] = details

    return out

def summarize_reasons(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns (bucket_summary, detail_summary) with counts and %.
    """
    if df.empty:
        return (pd.DataFrame(columns=["reason_bucket","count","pct"]),
                pd.DataFrame(columns=["reason_detail","count","pct"]))

    bucket = (df["reason_bucket"]
              .value_counts(dropna=False)
              .rename_axis("reason_bucket")
              .reset_index(name="count"))
    bucket["pct"] = (bucket["count"] / bucket["count"].sum() * 100).round(1)

    details = (df.dropna(subset=["reason_detail"])
                 .groupby("reason_detail", dropna=False)
                 .size()
                 .reset_index(name="count")
                 .sort_values("count", ascending=False))
    if not details.empty:
        details["pct"] = (details["count"] / details["count"].sum() * 100).round(1)
    else:
        details = pd.DataFrame(columns=["reason_detail","count","pct"])

    return bucket, details
