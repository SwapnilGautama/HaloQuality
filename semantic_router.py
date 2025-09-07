# semantic_router.py
from __future__ import annotations
from typing import Dict, Tuple

def route(q: str) -> Tuple[str|None, Dict]:
    """
    Return ("complaints_june_by_portfolio" | "first_pass_accuracy" | None, params)
    This file must stay tiny/stable so Q1 cannot be broken by Q2 updates.
    """
    if not q:
        return None, {}

    t = q.lower().strip()

    # Q1 (complaints) — extremely permissive to avoid misrouting
    if ("complaint" in t and ("portfolio" in t or "june" in t)) or "complaints_june_by_portfolio" in t:
        return "complaints_june_by_portfolio", {}

    # Q2 (first pass accuracy)
    if ("first pass" in t and "accuracy" in t) or "first_pass_accuracy" in t:
        return "first_pass_accuracy", {}

    return None, {}
