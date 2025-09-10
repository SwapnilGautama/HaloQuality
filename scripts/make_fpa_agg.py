# scripts/make_fpa_agg.py
from __future__ import annotations

from pathlib import Path
import hashlib
import pandas as pd
import numpy as np

JAN_2025 = pd.Period("2025-01")

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "data" / "first_pass_accuracy"
OUT_DIR = SRC_DIR / "agg"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def _pick(df: pd.DataFrame, names):
    lut = {c.lower(): c for c in df.columns}
    for n in names:
        if n.lower() in lut:
            return lut[n.lower()]
    return None

def _month(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce", dayfirst=True).dt.to_period("M")

def _read_any(path: Path) -> pd.DataFrame:
    return pd.read_excel(path)

def _hash_id(*bits: str) -> str:
    h = hashlib.sha1()
    for b in bits:
        if b:
            h.update(str(b).encode("utf-8", "ignore"))
            h.update(b"\0")
    return h.hexdigest()

def _label_reasons(df_fail: pd.DataFrame) -> pd.Series:
    """
    Lightweight fallback reason labeller.
    If you have core.reason_labeller, you can import and use it here instead.
    """
    txt = (df_fail.get("Case Comment") or df_fail.get("comment") or pd.Series([""]*len(df_fail))).fillna("").str.lower()

    # simple keyword bins (keep these in sync with your rca_patterns.yml if you want)
    bins = [
        ("Documentation", ["document", "paperwork", "evidence", "form"]),
        ("Process & Policy", ["process", "policy", "procedure", "guideline"]),
        ("Communication & Support", ["call", "email", "contact", "support"]),
        ("Staff & Behavior", ["rude", "attitude", "behaviour", "behavior"]),
        ("Charges & Billing", ["charge", "bill", "fee", "invoice"]),
        ("Product / Features", ["feature", "app", "portal", "function"]),
        ("Turnaround Time / Speed", ["delay", "slow", "time", "wait"]),
    ]
    lab = pd.Series(["Other"] * len(txt), index=txt.index)
    for tag, kws in bins:
        hit = txt.str.contains("|".join([rf"\b{k}\b" for k in kws]), regex=True)
        lab[hit] = tag
    return lab

def main():
    files = sorted(SRC_DIR.glob("FirstPassAccuracy*.xls*"))
    if not files:
        raise SystemExit("No FPA files found.")

    rows = []
    fails = []

    for f in files:
        df = _read_any(f)
        dcol = _pick(df, ["Activity Date", "ActivityDate", "Date", "Activity date"])
        rcol = _pick(df, ["Review Result", "Review result", "Result"])
        pcol = _pick(df, ["Portfolio", "portfolio"])
        tcol = _pick(df, ["Team manager", "Team Manager", "Manager", "Team", "Department", "TeamManager"])
        ccol = _pick(df, ["Case Comment", "Comments", "Reviewer Comment", "Comment"])

        if dcol is None or rcol is None:
            continue

        m = _month(df[dcol])
        res = df[rcol].astype(str).str.strip().str.lower()
        is_pass = res.str.startswith("pass")
        portfolio = df[pcol].astype(str) if pcol else "Unknown"
        team = df[tcol].astype(str) if tcol else ""

        keep = pd.DataFrame({
            "_m": m, "portfolio": portfolio, "is_pass": is_pass, "team": team
        }).dropna(subset=["_m"])
        keep = keep[keep["_m"] >= JAN_2025]
        rows.append(keep)

        # fail comments (for sample index + reason monthly)
        if ccol is not None:
            fail = pd.DataFrame({
                "_m": m, "portfolio": portfolio, "team": team,
                "comment": df[ccol].fillna("").astype(str),
                "is_pass": is_pass
            })
            fail = fail[(~fail["is_pass"]) & (fail["_m"] >= JAN_2025)].copy()
            if not fail.empty:
                fails.append(fail[["_m", "portfolio", "team", "comment"]])

    if not rows:
        raise SystemExit("No rows after normalisation.")

    df = pd.concat(rows, ignore_index=True)

    # portfolio × month metrics
    g = (df.groupby(["portfolio", "_m"])
           .agg(total_cases=("is_pass", "size"), passed_cases=("is_pass", "sum"))
           .reset_index())
    g["pass_pct"] = (g["passed_cases"] * 100.0 / g["total_cases"].replace(0, np.nan)).fillna(0.0)
    g["_m"] = g["_m"].astype(str)
    g = g.rename(columns={"_m": "month", "portfolio": "portfolio"})
    g = g.sort_values(["portfolio", "month"]).reset_index(drop=True)

    # write 1) portfolio × month metrics
    (OUT_DIR / "fpa_portfolio_month.parquet").unlink(missing_ok=True)
    g.to_parquet(OUT_DIR / "fpa_portfolio_month.parquet", index=False)

    # reasons + comments index (optional but tiny)
    if fails:
        f = pd.concat(fails, ignore_index=True)
        if not f.empty:
            # reasons
            reasons = _label_reasons(f)
            f["_m"] = f["_m"].astype(str)
            f["reason"] = reasons

            r = (f.groupby(["portfolio", "_m", "reason"])
                  .size().reset_index(name="count"))\
                .rename(columns={"_m": "month"})

            (OUT_DIR / "fpa_reasons_portfolio_month.parquet").unlink(missing_ok=True)
            r.to_parquet(OUT_DIR / "fpa_reasons_portfolio_month.parquet", index=False)

            # comment index (short excerpts)
            EXCERPT = f["comment"].str.slice(0, 200).str.replace(r"\s+", " ", regex=True)
            cidx = pd.DataFrame({
                "comment_id": [_hash_id(m, p, t, c) for m, p, t, c in zip(f["_m"], f["portfolio"], f["team"], f["comment"])],
                "month": f["_m"].astype(str),
                "portfolio": f["portfolio"].astype(str),
                "team": f["team"].astype(str),
                "reason": f["reason"].astype(str),
                "excerpt": EXCERPT
            })
            (OUT_DIR / "fpa_comments_index.parquet").unlink(missing_ok=True)
            cidx.to_parquet(OUT_DIR / "fpa_comments_index.parquet", index=False)

    print("✓ Aggregates written to", OUT_DIR)

if __name__ == "__main__":
    main()
