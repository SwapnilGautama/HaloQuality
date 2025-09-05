# questions/__init__.py
# Mark the folder as a package so `import questions.xxx` works.

from pathlib import Path
import sys

# Ensure project root is on sys.path
PKG_DIR = Path(__file__).resolve().parent
ROOT = PKG_DIR.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# (Optional) explicitly export known modules to help IDEs/static tools
__all__ = [
    "complaint_volume_rate",
    "complaints_per_thousand",
    "corr_nps",
    "fpa_fail_rate",
    "fpa_fail_drivers",
    "mom_overview",
    "rca1_portfolio_process",
    "unique_cases_mom",
]
