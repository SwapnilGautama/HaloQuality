# semantic_router.py
import re
from typing import Dict

def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower()).strip()

def match(q: str) -> Dict:
    """
    Extremely small router: picks between
    - questions.complaints_june_by_portfolio
    - questions.first_pass_accuracy
    Returns {'slug': ..., 'params': {...}}
    """
    text = _norm(q)

    # FPA intent keywords
    if any(k in text for k in [
        "first pass accuracy", "first-pass accuracy", "fpa", "pass %", "pass percent", "pass rate"
    ]):
        return {"slug": "first_pass_accuracy", "params": {}}

    # Complaints intent keywords (default)
    if any(k in text for k in [
        "complaint analysis", "complaints analysis", "complaints —", "complaints -",
        "complaints by portfolio", "rca", "reasons"
    ]):
        return {"slug": "complaints_june_by_portfolio", "params": {}}

    # Fallback: prefer FPA if user typed 'pass'
    if "pass" in text:
        return {"slug": "first_pass_accuracy", "params": {}}

    return {"slug": "complaints_june_by_portfolio", "params": {}}
