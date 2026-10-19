"""Build the training set for the total readers from data/overnight/total_loc.csv + labels_all.csv + split.csv.

Targets: the true total is the sum of the labelled question marks. Where the total is written as an expression (24+3) the
expression is the target. Sheets whose written total disagrees with the labelled rows, scribbled sheets (X) and expressions
cut off at the right edge of the aligned image are excluded from training (trust = 0) but kept for evaluation notes.

Usage: python scripts/build_total_data.py  ->  data/overnight/total_data.npz
"""
import csv
import re
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grade_eval import totals
from grade_eval.vision import cells

ALIGNED = ROOT / "data" / "aligned_eval"
MISMATCH = {"021", "072", "073", "019", "127", "133", "049", "120", "132"}     # written total != labelled rows (checked by eye or by the labeller)
EXPR = re.compile(r"(\d{1,2}\+\d{1,2})")


def fmt(v: float) -> str:
    return str(int(v)) if float(v).is_integer() else f"{v:g}"


def main():
    labels = {r["file"]: r for r in csv.DictReader(open(ROOT / "data" / "labels_all.csv", encoding="utf-8-sig"))}
    split = {r["file"]: r["split"] for r in csv.DictReader(open(ROOT / "data" / "split.csv", encoding="utf-8-sig"))}
    locs = list(csv.DictReader(open(ROOT / "data" / "overnight" / "total_loc.csv")))
    S, G, meta = [], [], []
    for r in locs:
        f = f"sheet_{r['sheet']}.jpeg"
        im = cv2.imread(str(ALIGNED / f"sheet_{r['sheet']}.png"))
        if im is None or f not in labels:
            continue
        mask = cells.ink_mask(im)
        S.append(totals.sub_strip(mask)); G.append(totals.grand_bands(mask))
        total = sum(float(labels[f][f"q{i}"] or 0) for i in range(1, 11))
        loc, note = r["loc"], r["note"]
        expr = (EXPR.search(note) or [None])[0] if "+" in note else None
        trust = int(loc in "SGBN" and r["sheet"] not in MISMATCH)
        tsub = tgr = ""
        if loc in ("S", "B"):
            tsub = expr or fmt(total)
            if "struck" in note:
                tsub = ""
        if loc in ("G", "B"):
            tgr = fmt(total)
            if loc == "G" and expr:
                trust_g = 0
        meta.append((r["sheet"], loc, split[f], total, tsub, tgr, trust, int(loc == "G" and bool(expr))))
    np.savez_compressed(ROOT / "data" / "overnight" / "total_data.npz", S=np.stack(S), G=np.stack(G),
                        sheet=np.array([m[0] for m in meta]), loc=np.array([m[1] for m in meta]), split=np.array([m[2] for m in meta]),
                        total=np.array([m[3] for m in meta]), tsub=np.array([m[4] for m in meta]), tgr=np.array([m[5] for m in meta]),
                        trust=np.array([m[6] for m in meta]), grand_skip=np.array([m[7] for m in meta]))
    import collections
    print(len(meta), collections.Counter((m[2], m[1]) for m in meta), "trusted", sum(m[6] for m in meta))


if __name__ == "__main__":
    main()
