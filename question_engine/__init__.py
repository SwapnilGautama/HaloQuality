# question_engine/__init__.py
"""
Exports the single entrypoint `run_nl(store)` used by app.py.
"""
from .nl_router import run_nl

__all__ = ["run_nl"]
