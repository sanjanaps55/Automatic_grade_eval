"""Score the pipeline readers against data/labels_all.csv (register no. + per-question marks).

Usage: python scripts/score_pipeline.py [--split val|test|train|all] [--out data/score_rows.csv]
Counts, per question row: auto-correct / auto-WRONG (the number that must stay ~0) / held for review.
Aligned images are cached in data/aligned_eval/.
"""
import csv
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from grade_eval.pipeline import GradePipeline
from grade_eval.vision import layout
from grade_eval.vision.preprocess import load_image


def arg(n, d):
    return sys.argv[sys.argv.index(n) + 1] if n in sys.argv else d


def num(s):
    s = (s or "").strip()
    return float(s) if s else None


def main():
    want = arg("--split", "all")
    labels = list(csv.DictReader(open(ROOT / "data" / "labels_all.csv", encoding="utf-8-sig")))
    split = {r["file"]: r["split"] for r in csv.DictReader(open(ROOT / "data" / "split.csv", encoding="utf-8-sig"))}
    cache = ROOT / "data" / "aligned_eval"; cache.mkdir(exist_ok=True)
    pipe = GradePipeline(db_path=":memory:")
    tot = dict(sheets=0, aligned=0, reg_ok=0, rows=0, auto_ok=0, auto_wrong=0, review=0, sheets_clean=0, sheets_clean_wrong=0)
    wrong_rows = []
    from collections import Counter
    tally = Counter()
    for r in labels:
        sp = split.get(r["file"], "train")
        if want != "all" and sp != want:
            continue
        tot["sheets"] += 1
        c = cache / (Path(r["file"]).stem + ".png")
        if c.exists():
            a = cv2.imread(str(c))
        else:
            res = layout.align(load_image(ROOT / "data" / "samples" / r["file"]))
            if res is None:
                continue
            a = res.image; cv2.imwrite(str(c), a)
        tot["aligned"] += 1
        tot["reg_ok"] += pipe.read_reg(a) == r["reg_no"]
        qs, sub, confs, reasons, cross = pipe.read_marks(a)
        flagged = {int(x.split(":")[0][1:]) for x in reasons if x.startswith("Q") and ":" in x}
        for x in reasons:
            tally['sum/ab' if 'written sum' in x else 'unclear' if 'unclear' in x else 'beyond Q10' if 'beyond' in x else 'other'] += 1
        tally['sheets_with_beyond'] += any('beyond' in x for x in reasons)
        sheet_ok, any_flag = True, bool(flagged) or any("beyond" in x for x in reasons)
        for q in range(1, 11):
            truth = num(r[f"q{q}"])
            tot["rows"] += 1
            if q in flagged:
                tot["review"] += 1; continue
            pred = qs.get(q, {}).get("a")
            if (pred or 0.0) == (truth or 0.0):          # blank and 0 are the same mark
                tot["auto_ok"] += 1
            else:
                tot["auto_wrong"] += 1; sheet_ok = False
                wrong_rows.append((r['file'][6:9], q, truth, qs.get(q, {}).get('raw'), qs.get(q, {}).get('conf'), 'sub' if sub is not None else 'nosub'))
        # sheet-level acceptance policies
        has_sub = sub is not None and not any('sub-total' in x for x in reasons)
        cleanread = not any_flag
        pol = {'A_sub_matches': cleanread and has_sub,
               'B_A_or_singles_hi': cleanread and (has_sub or (all('+' not in v['raw'] for v in qs.values()) and min([v['conf'] for v in qs.values()] or [1]) >= 0.97)),
               'C_A_or_singles': cleanread and (has_sub or all('+' not in v['raw'] for v in qs.values()))}
        for k, v in pol.items():
            if v:
                tot[k] = tot.get(k, 0) + 1
                tot[k + '_wrong'] = tot.get(k + '_wrong', 0) + (not sheet_ok)
        if not any_flag:
            tot["sheets_clean"] += 1
            tot["sheets_clean_wrong"] += (not sheet_ok)
    print(f"split={want}", tot)
    print("flag reasons:", dict(tally))
    print("auto-wrong rows:", wrong_rows[:40])
    if tot["sheets"]:
        print(f"reg exact {tot['reg_ok']}/{tot['aligned']} | rows auto-correct {tot['auto_ok']}, AUTO-WRONG {tot['auto_wrong']}, "
              f"review {tot['review']} of {tot['rows']} | sheets with no flags: {tot['sheets_clean']} (of which wrong: {tot['sheets_clean_wrong']})")


if __name__ == "__main__":
    main()
