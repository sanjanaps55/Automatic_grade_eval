"""Score joint roster matching from a predictions CSV (any engine).

Usage: python scripts/score_roster.py [preds.csv] [roster.csv|xlsx] [--require-name]
For each sheet, compares: reg no. alone (nearest roster row) vs reg + name jointly. Reports auto-correct,
auto-WRONG (the number that must stay ~0) and review counts, per split and overall.
"""
import csv
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grade_eval.roster import load_roster, match_sheet


def outcome(m, true_reg, require_name):
    if m.status == "review" or (require_name and m.status != "auto"): return "review"
    return "correct" if m.row.reg_no == true_reg else "WRONG"


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    preds = Path(args[0]) if args else ROOT / "data" / "preds_trocr.csv"
    roster = load_roster(args[1] if len(args) > 1 else ROOT / "data" / "roster_test.csv")
    req = "--require-name" in sys.argv
    rows = list(csv.DictReader(open(preds, encoding="utf-8")))
    print(f"preds {preds.name} | roster {len(roster)} | sheets {len(rows)}"
          f" | raw reg exact {sum(r['reg_pred'] == r['reg_true'] for r in rows)}/{len(rows)}"
          f" | raw name exact {sum(r['name_pred'].lower() == r['name_true'].lower() for r in rows)}/{len(rows)}")
    for mode, use_name in (("reg only      ", False), ("reg + name    ", True)):
        for split in ("val", "test", "all"):
            sel = [r for r in rows if split == "all" or r["split"] == split]
            c = Counter(outcome(match_sheet(r["reg_pred"], r["name_pred"] if use_name else "", roster), r["reg_true"], req and use_name)
                        for r in sel)
            print(f"  {mode} {split:4s} n={len(sel):2d}: auto-correct {c['correct']:2d} | auto-WRONG {c['WRONG']:2d} | review {c['review']:2d}")
    print("\nsheets sent to review / wrong (reg + name):")
    for r in rows:
        m = match_sheet(r["reg_pred"], r["name_pred"], roster)
        o = outcome(m, r["reg_true"], req)
        if o != "correct":
            print(f"  [{o}] true {r['reg_true']} {r['name_true']!r} | read {r['reg_pred']} {r['name_pred']!r} | {m.status} {m.row.reg_no if m.row else '-'} {m.reasons}")


if __name__ == "__main__":
    main()
