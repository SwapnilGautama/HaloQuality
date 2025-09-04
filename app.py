# add at top
import glob
from pathlib import Path

# in __init__ keep as-is; in reload() replace the "Cases" block with:
cases_files = []
cases_input = cases_path or DEFAULT_CASES_XLSX
p = Path(cases_input)
if p.is_dir():
    cases_files = sorted(glob.glob(str(p / "*.xlsx")))
elif p.exists():
    cases_files = [str(p)]
else:
    # also support HALO_DATA_DIR + /cases if user only sets HALO_DATA_DIR
    default_dir = Path(DEFAULT_DATA_DIR) / "cases"
    cases_files = sorted(glob.glob(str(default_dir / "*.xlsx")))

if not cases_files:
    raise HTTPException(status_code=400, detail="No case files found (expected .xlsx under data/cases)")

frames = []
for f in cases_files:
    df = pd.read_excel(f, sheet_name=0)
    for c in ["Report_Date", "Case ID"]:
        if c not in df.columns:
            raise HTTPException(status_code=400, detail=f"Cases file missing '{c}': {Path(f).name}")
    df["month"] = df["Report_Date"].apply(_to_month_str)
    df["Portfolio_std"] = df.get("Portfolio", np.nan).apply(_std_portfolio) if "Portfolio" in df.columns else np.nan
    frames.append(df)

cases = pd.concat(frames, ignore_index=True)
# dedupe by month + case id (+ optional group cols after we compute them in KPI)
cases = cases.dropna(subset=["Case ID"])
cases = cases.drop_duplicates(subset=["month", "Case ID"], keep="first")
self.cases = cases
