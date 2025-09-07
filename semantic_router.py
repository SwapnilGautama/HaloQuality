# semantic_router.py
"""
Very small router: map a free-text question to a question "slug"
and optional params. Keep this additive and safe.
"""

from __future__ import annotations
from typing import Dict

def match(q: str) -> Dict:
    ql = (q or "").lower().strip()

    # --- First Pass Accuracy intents
    fpa_terms = ["first pass", "first-pass", "fpa", "pass accuracy", "pass rate", "pass %", "first pass accuracy"]
    if any(t in ql for t in fpa_terms):
        return {"slug": "first_pass_accuracy", "params": {}}

    # --- Complaints analysis (default/fallback)
    comp_terms = ["complaint", "complaints", "rca", "per 1000", "portfolio"]
    if any(t in ql for t in comp_terms):
        return {"slug": "complaints_june_by_portfolio", "params": {}}

    # default to complaints to remain backwards compatible
    return {"slug": "complaints_june_by_portfolio", "params": {}}
