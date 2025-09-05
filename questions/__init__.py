# questions/__init__.py
"""
Mark `questions` as a package so `import questions.<slug>` works.
Optionally expose slugs via __all__ (not required for importlib).
"""
__all__ = [
    "complaints_per_thousand",
    "complaint_volume_rate",
    "unique_cases_mom",
    "mom_overview",
    "fpa_fail_rate",
    "fpa_fail_drivers",
    "rca1_portfolio_process",
    "corr_nps",
]
