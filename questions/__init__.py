# questions/__init__.py
"""
Expose question modules by name so the app can resolve slugs safely.
Also provide a lazy-import fallback so new modules do not break loading.
"""

from importlib import import_module as _imp

# Keep your existing entries and make sure Q1 + Q2 are included here.
__all__ = [
    # --- your existing question names (keep them exactly as-is) ---
    "complaints_per_thousand",
    "complaint_volume_rate",
    "unique_cases_mom",
    "mom_overview",
    "fpa_fail_rate",
    "fpa_fail_drivers",
    "rca1_portfolio_process",
    "corr_nps",
    "fail_reasons_analysis",

    # Ensure these two are ALWAYS present
    "complaints_june_by_portfolio",   # Q1
    "first_pass_accuracy",            # Q2
]

def __getattr__(name):
    """
    Lazy import so 'from questions import *' + globals()[slug] always works,
    even if a new question wasn't added to __all__ yet.
    """
    try:
        return _imp(f"{__name__}.{name}")
    except Exception as e:
        raise AttributeError(f"questions package has no module '{name}': {e}")
