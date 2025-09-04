import re

def route(prompt: str) -> str | None:
    """Return a question id based on the prompt; keep this fast & reliable.
       We can upgrade to embeddings later."""
    if not prompt: return None
    q = prompt.lower()
    # v1: complaints vs NPS correlation
    if ("nps" in q or "csat" in q) and any(k in q for k in ["corr", "correlation", "relationship", "impact", "association"]):
        return "corr_nps"
    if "complaint" in q and "correlation" in q:
        return "corr_nps"
    return None

def extract_month(prompt: str, default_month: str | None) -> str | None:
    m = re.search(r"(20\d{2})[-/](0[1-9]|1[0-2])", (prompt or ""))
    return f"{m.group(1)}-{m.group(2)}" if m else default_month

