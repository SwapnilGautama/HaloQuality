# core/reason_labeller.py
# Hybrid (rules → RCA2 → ML) reason classification for Fail rows
# Safe to import from Q2 (FPA) and Q3 (FRA)

from __future__ import annotations

import os
import re
from typing import Dict, Optional, Tuple, Iterable

import pandas as pd

# Optional ML imports (guarded)
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.svm import LinearSVC
    import joblib  # scikit-learn joblib or standalone joblib is fine
except Exception:  # pragma: no cover
    TfidfVectorizer = None
    LinearSVC = None
    joblib = None


# -----------------------------
# Canonical RCA2 / reason set
# -----------------------------

CANONICAL_REASONS: Tuple[str, ...] = (
    "Incorrect Data Input",
    "Knowledge Gap–Onshore",
    "Incorrect Scheme Rule",
    "Incorrect Calculator",
    "Incorrect Calc Estimate",
    "Knowledge Gap–Offshore",
)

# Basic normalisation map for RCA2 text → canonical bucket
_NORMALIZE_RCA2: Dict[str, str] = {
    # exact matches first
    "Incorrect Data Input": "Incorrect Data Input",
    "Incorrect Calculator": "Incorrect Calculator",
    "Knowledge Gap–Onshore": "Knowledge Gap–Onshore",
    "Knowledge Gap-Onshore": "Knowledge Gap–Onshore",
    "Knowledge Gap–Offshore": "Knowledge Gap–Offshore",
    "Knowledge Gap-Offshore": "Knowledge Gap–Offshore",
    "Incorrect Scheme Rule": "Incorrect Scheme Rule",
    "Incorrect Calc Estimate": "Incorrect Calc Estimate",
    "Incorrect Calculator Logic": "Incorrect Calculator",        # map variants
    "Incorrect Calc": "Incorrect Calculator",
    "Incorrect Calc Estimate ": "Incorrect Calc Estimate",
}

def _normalize_rca2(val: Optional[str]) -> Optional[str]:
    if not val or not isinstance(val, str):
        return None
    v = val.strip()
    if v in _NORMALIZE_RCA2:
        return _NORMALIZE_RCA2[v]
    # fuzzy-ish quick paths
    lv = v.lower()
    if "data" in lv and ("input" in lv or "entry" in lv):
        return "Incorrect Data Input"
    if "scheme" in lv and "rule" in lv:
        return "Incorrect Scheme Rule"
    if "calc" in lv and ("logic" in lv or "incorrect" in lv):
        return "Incorrect Calculator"
    if "estimate" in lv or "quote" in lv or "illustration" in lv:
        return "Incorrect Calc Estimate"
    if "offshore" in lv:
        return "Knowledge Gap–Offshore"
    if "onshore" in lv or "training" in lv or "procedure" in lv:
        return "Knowledge Gap–Onshore"
    return None


# -----------------------------
# Rules (keyword/regex) fallback
# -----------------------------

def _rx(words: Iterable[str]) -> re.Pattern:
    """
    Build a case-insensitive word-boundary regex from a list of tokens/phrases.
    """
    parts = []
    for w in words:
        w = w.strip()
        if not w:
            continue
        parts.append(rf"(?:\b{re.escape(w)}\b)")
    if not parts:
        parts = [r"a^"]  # match nothing
    return re.compile("|".join(parts), flags=re.I)

_RULES: Dict[str, re.Pattern] = {
    # 1) Incorrect Data Input
    "Incorrect Data Input": _rx([
        "incorrect input", "data entry", "mistyped", "typo", "wrong value",
        "entered", "amended", "updated", "keyed", "mis-key", "miskey",
        "ni number", "national insurance", "dob", "date of birth", "postcode",
        "address",
    ]),
    # 2) Knowledge Gap – Onshore
    "Knowledge Gap–Onshore": _rx([
        "not aware", "missed step", "training", "procedure", "guidance",
        "interpretation", "didn't follow", "did not follow",
    ]),
    # 3) Incorrect Scheme Rule
    "Incorrect Scheme Rule": _rx([
        "scheme rule", "rule applied", "annuity factor", "late retirement",
        "pension increase", "revaluation", "commutation", "step up", "gmp",
    ]),
    # 4) Incorrect Calculator
    "Incorrect Calculator": _rx([
        "calc incorrect", "calculator incorrect", "arrears calculated",
        "calc run", "increase applied", "formula",
    ]),
    # 5) Incorrect Calc Estimate
    "Incorrect Calc Estimate": _rx([
        "estimate", "quote", "illustration", "projection", "reval rate",
        "basis", "assumption",
    ]),
    # 6) Knowledge Gap – Offshore
    "Knowledge Gap–Offshore": _rx([
        "rejected comment", "need to", "how to", "split", "pre", "age", "gmp",  # generic coaching-ish signals
    ]),
}

def _rules_reason(text: str) -> Optional[str]:
    if not text or not isinstance(text, str):
        return None
    for label, pattern in _RULES.items():
        if pattern.search(text):
            return label
    return None


# -----------------------------
# ML utilities (optional)
# -----------------------------

def _candidate_model_paths() -> Tuple[str, ...]:
    return (
        "core/artifacts/rca2_text_model.joblib",
        "data/models/rca2_text_model.joblib",
        "/mnt/data/rca2_text_model.joblib",
        "/tmp/rca2_text_model.joblib",
    )

def load_model_bundle() -> Optional[dict]:
    """
    Returns {'vec': TfidfVectorizer, 'clf': LinearSVC, 'labels': list} or None.
    """
    if joblib is None:
        return None
    for p in _candidate_model_paths():
        try:
            if os.path.exists(p):
                return joblib.load(p)
        except Exception:
            continue
    return None

def maybe_fit_model(df: pd.DataFrame,
                    text_col: str = "Case Comment",
                    rca2_col: str = "RCA2",
                    min_rows: int = 50) -> Optional[dict]:
    """
    Train a tiny TF-IDF + LinearSVC on rows with RCA2 present.
    Returns model bundle or None if not enough labels or sklearn not available.
    """
    if TfidfVectorizer is None or LinearSVC is None:
        return None

    if text_col not in df.columns or rca2_col not in df.columns:
        return None

    df_lab = df[[text_col, rca2_col]].dropna()
    df_lab[rca2_col] = df_lab[rca2_col].map(_normalize_rca2)
    df_lab = df_lab[df_lab[rca2_col].isin(CANONICAL_REASONS)]
    df_lab[text_col] = df_lab[text_col].astype(str)

    if len(df_lab) < min_rows:
        return None

    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=30000)
    X = vec.fit_transform(df_lab[text_col])
    y = df_lab[rca2_col].values
    clf = LinearSVC()
    clf.fit(X, y)

    bundle = {"vec": vec, "clf": clf, "labels": sorted(pd.unique(y))}
    # Try to persist (best-effort)
    if joblib is not None:
        for p in _candidate_model_paths():
            try:
                os.makedirs(os.path.dirname(p), exist_ok=True)
                joblib.dump(bundle, p)
                break
            except Exception:
                continue
    return bundle


# -----------------------------
# Public labelling API
# -----------------------------

def classify_text(text: Optional[str],
                  rca2: Optional[str] = None,
                  model_bundle: Optional[dict] = None) -> str:
    """
    Rules → RCA2 → ML → Other. Returns a canonical reason string.
    """
    # 1) Rules
    r = _rules_reason(text or "")
    if r:
        return r

    # 2) RCA2 direct
    r2 = _normalize_rca2(rca2)
    if r2:
        return r2

    # 3) ML
    try:
        if model_bundle and text and isinstance(text, str):
            vec = model_bundle.get("vec")
            clf = model_bundle.get("clf")
            if vec is not None and clf is not None:
                pred = clf.predict(vec.transform([text]))[0]
                if pred in CANONICAL_REASONS:
                    return str(pred)
    except Exception:
        pass

    # 4) Fallback
    return "Other"


def label_dataframe(df: pd.DataFrame,
                    text_col: str = "Case Comment",
                    rca2_col: str = "RCA2",
                    model_bundle: Optional[dict] = None) -> pd.Series:
    """
    Vectorised classification for a DataFrame of fail rows.
    """
    txt = df[text_col] if text_col in df.columns else pd.Series(index=df.index, dtype=object)
    r2 = df[rca2_col] if rca2_col in df.columns else pd.Series(index=df.index, dtype=object)

    # Ensure string-ish
    txt = txt.fillna("").astype(str)
    r2 = r2.fillna("")

    return pd.Series(
        (classify_text(t, rr, model_bundle) for t, rr in zip(txt, r2)),
        index=df.index,
        name="reason",
    )


def get_or_fit_model(df_all: pd.DataFrame,
                     text_col: str = "Case Comment",
                     rca2_col: str = "RCA2") -> Optional[dict]:
    """
    1) Try load a saved model.
    2) If not available and we have enough RCA2 labels, fit one quickly.
    """
    bundle = load_model_bundle()
    if bundle is not None:
        return bundle
    return maybe_fit_model(df_all, text_col=text_col, rca2_col=rca2_col, min_rows=50)
