# semantic_router.py
from __future__ import annotations
from typing import Dict, Tuple

def route(q: str) -> Tuple[str | None, Dict]:
    """
    Extremely small, stable router.
    Returns (slug, params)
    """
    if not q:
        return None, {}

    ql = q.lower().strip()

    # Q1 — complaints
    if ("complaint" in ql and "portfolio" in ql) or "complaints_june_by_portfolio" in ql:
        return "complaints_june_by_portfolio", {}

    # Q2 — first pass accuracy
    if ("first pass" in ql and "accuracy" in ql) or "first_pass_accuracy" in ql:
        return "first_pass_accuracy", {}

    # Default: None (UI will show hint)
    return None, {}
