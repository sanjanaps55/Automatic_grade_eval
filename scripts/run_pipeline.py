"""Run the full pipeline over a folder of sheet photos and write marks.xlsx.

Usage: python scripts/run_pipeline.py <folder> [--roster class.xlsx|csv] [--db data/grades.db] [--xlsx marks.xlsx]
Prints one line per photo; sheets needing review are listed with reasons and are NOT written to Excel.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grade_eval.pipeline import GradePipeline
from grade_eval.roster import load_roster
from grade_eval.store.excel import export_excel


def arg(name, default=None):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


def main():
    folder = Path(next(a for a in sys.argv[1:] if not a.startswith("--") and a not in (arg("--roster"), arg("--db"), arg("--xlsx"))))
    roster = load_roster(arg("--roster")) if arg("--roster") else None
    pipe = GradePipeline(db_path=arg("--db", ROOT / "data" / "grades.db"), roster=roster,
                         save_aligned=ROOT / "data" / "aligned_new")
    counts = {}
    for p in sorted(list(folder.glob("*.jp*g")) + list(folder.glob("*.png"))):
        r = pipe.process(p.read_bytes(), p.name)
        counts[r["status"]] = counts.get(r["status"], 0) + 1
        marks = " ".join(f"{q}:{v['a']:g}" for q, v in r["questions"].items())
        print(f"{p.name:28s} {r['status']:13s} {r['reg_no'] or r.get('reg_raw', ''):10s} {marks}")
        for why in r["reasons"]:
            print(f"      - {why}")
    out = export_excel(pipe.conn, arg("--xlsx", ROOT / "marks.xlsx"))
    print(counts, "->", out)


if __name__ == "__main__":
    main()
