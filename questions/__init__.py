# questions/__init__.py
"""
Marks `questions` as a package and exposes your question modules.
Also ensures the project root is on sys.path so `import questions.<slug>` works
on Streamlit Cloud.
"""
from pathlib import Path
import sys

PKG_DIR = Path(__file__).resolve().parent
ROOT = PKG_DIR.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

__all__ = [
    "complaint_volume_rate",
    "complaints_per_thousand",
    "corr_nps",
    "fpa_fail_rate",        # shim -> fpa_fail_drivers.run
    "fpa_fail_drivers",     # the real implementation
    "mom_overview",
    "rca1_portfolio_process",
    "unique_cases_mom",
]
