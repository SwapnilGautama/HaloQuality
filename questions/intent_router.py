# ... existing imports / helpers ...

MOM_KEYWORDS = [
    "month on month", "mom", "mo m", "trend", "trends",
    "time series", "over time", "by month", "monthly"
]

def route(prompt: str) -> str | None:
    p = (prompt or "").lower().strip()

    # existing rule for correlation
    if "nps" in p and "complaint" in p and "correlation" in p:
        return "corr_nps"

    # NEW: month-on-month view
    if any(k in p for k in MOM_KEYWORDS) or ("complaint" in p and "month" in p):
        return "mom_overview"

    # fallback
    return None
