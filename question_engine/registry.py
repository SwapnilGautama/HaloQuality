from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
QUESTIONS_DIR = ROOT / "questions"

def list_questions():
    return [p.stem for p in QUESTIONS_DIR.glob("*.yml")]

def get_spec_path(question_id: str) -> Path:
    p = QUESTIONS_DIR / f"{question_id}.yml"
    if not p.exists():
        raise FileNotFoundError(f"Question spec not found: {p.name}")
    return p
