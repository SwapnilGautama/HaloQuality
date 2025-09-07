# questions/__init__.py
"""
Expose question modules by name so the app can resolve slugs safely.
We also add a lazy import fallback to avoid future breakages if a module
is added but not listed in __all__ yet.
"""

__all__ = [
    # --- keep your existing entries exactly as-is ---
    "complaints_per_thousand",
    "complaint_volume_rate",
    "unique_cases_mom",
    "mom_overview",
    "fpa_fail_rate",
    "fpa_fail_drivers",
    "rca1_portfolio_process",
    "corr_nps",

    # Q1 and Q2 slugs so they are always available:
    "complaints_june_by_portfolio",
    "first_pass_accuracy",
]

# Optional: robust lazy import so future modules don’t break loading
def __getattr__(name):
    import importlib
    try:
        return importlib.import_module(f"{__name__}.{name}")
    except Exception as e:
        raise AttributeError(f"questions package has no module '{name}': {e}")
