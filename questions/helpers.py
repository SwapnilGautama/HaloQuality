# helpers.py
import re
import pandas as pd

def _norm(name: str) -> str:
    return re.sub(r'[^a-z0-9]+', '', str(name).strip().lower())

def pick_col(df: pd.DataFrame, candidates=None, regex: str | None = None) -> str | None:
    """
    Pick a column from df using case/space-insensitive matching.
    - candidates: list of exact names to try (leniently matched)
    - regex: optional regex as a fallback
    """
    if candidates is None:
        candidates = []
    norm_map = {_norm(c): c for c in df.columns}
    # try candidates
    for c in candidates:
        key = _norm(c)
        if key in norm_map:
            return norm_map[key]
    # fallback regex
    if regex:
        pat = re.compile(regex, re.I)
        for c in df.columns:
            if pat.search(str(c)):
                return c
    return None

def ensure_datetime(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce", dayfirst=True, infer_datetime_format=True)

def portfolio_selector(series: pd.Series, wanted: str) -> pd.Series:
    """Exact match first; fallback to word-boundary contains."""
    s = series.fillna("").astype(str)
    w = (wanted or "").strip()
    exact = s.str.casefold().eq(w.casefold())
    if exact.any():
        return exact
    return s.str.contains(rf"\b{re.escape(w)}\b", case=False, na=False)
