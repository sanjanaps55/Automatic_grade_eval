"""Build kaggle_marks.zip: everything the marks-reader bake-off needs (code, cells.npz, labels, split, records, marks.pt warm start).

Usage: python scripts/package_kaggle_marks.py   ->  kaggle_marks.zip (upload as a Kaggle dataset, open kaggle_marks_bakeoff.ipynb)
"""
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILES = ["scripts/train_marks.py", "scripts/train_marks_sum.py", "scripts/train_marks_trocr.py", "scripts/train_marks_hard.py", "data/label_work/hard_notation.txt", "models/marks_sum.pt",
         "data/cells.npz", "data/cells_deg.npz", "data/labels_all.csv", "data/split.csv", "data/label_work/records.txt", "models/marks.pt", "models/hard_b_s0.pt"]


def main():
    out = ROOT / "kaggle_marks.zip"
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for f in FILES:
            src = ROOT / f
            if not src.exists():                       # moved aside when no longer used by the app
                src = ROOT / "models" / "not_used_models" / Path(f).name
            z.write(src, f)
        for p in (ROOT / "grade_eval").rglob("*.py"):
            z.write(p, p.relative_to(ROOT).as_posix())
    print(out, round(out.stat().st_size / 1e6, 1), "MB")


if __name__ == "__main__":
    main()
