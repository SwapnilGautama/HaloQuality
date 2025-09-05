# question_engine/lexicon.py
from __future__ import annotations

# Canonical dimensions (what’s in your dataframes)
DIM_CANON = {
    "portfolio": "Portfolio_std",
    "portfolio_std": "Portfolio_std",
    "process": "ProcessName",
    "processname": "ProcessName",
    "parent case type": "Parent_Case_Type",
    "rca1": "RCA1",
    "rca2": "RCA2",
    "scheme": "Scheme",
    "team": "TeamName",
    "team name": "TeamName",
    "team manager": "TeamManager",
    "location": "Location",
    "receipt method": "Receipt_Method",
    "month": "Month",
}

# Metric names used in outputs
METRIC_CANON = {
    # cases
    "cases": "Unique_Cases",
    "unique cases": "Unique_Cases",
    "avg days": "Avg_NoOfDays",
    "average days": "Avg_NoOfDays",
    # complaints
    "complaints": "Complaints",
    "complaints per 1000": "Complaints_per_1000",
    "complaints/1000": "Complaints_per_1000",
    # fpa
    "reviewed": "Reviewed",
    "fails": "Fails",
    "fail rate": "Fail_Rate",
}

# Soft synonyms → canonical dimension/metric
DIM_SYNONYMS = {
    "office": "Portfolio_std",
    "portfolio name": "Portfolio_std",
    "process type": "ProcessName",
    "process group": "ProcessGroup",  # if present
    "parent": "Parent_Case_Type",
    "reason": "RCA1",
}

METRIC_SYNONYMS = {
    "volume": "Complaints",
    "rate": "Complaints_per_1000",
    "per 1000": "Complaints_per_1000",
    "error rate": "Fail_Rate",
    "accuracy fail rate": "Fail_Rate",
    "fail%": "Fail_Rate",
}

# Domain selection keywords
DOMAIN_HINTS = {
    "complaints": ["complaint", "rca", "reason", "aptia", "receipt"],
    "cases":      ["case", "unique case", "avg days", "sla", "within sla"],
    "fpa":        ["first pass", "fpa", "review", "fail", "review result"],
}
