# semantic_router.py
from __future__ import annotations
import re
from typing import Dict, Optional

# Minimal, robust router that will never break Q1.
# If we don't see "first pass" or "fpa", we default to Q1.

_Q2_PAT = re.compile(r"\b(first\s*pass|fpa)\b", re.I)

def match(q: str) -> Dict:
    q = (q or "").strip()
    if _Q2_PAT.search(q):
        return {"slug": "first_pass_accuracy", "params": {}}
    # default to Q1
    return {"slug": "complaints_june_by_portfolio", "params": {}}
