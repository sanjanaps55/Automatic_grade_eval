"""Align every labelled sheet, cache the aligned image, and write per-row ink strips + labels.

Usage: python scripts/build_cells.py   -> data/aligned/*.jpg, data/cells.npz
Rows are split into train/val by student (reg_no) so re-photos / repeat sheets never straddle the split.
"""
import csv
import hashlib
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grade_eval.vision import cells, layout
from grade_eval.vision.preprocess import load_image

ALIGNED = ROOT / "data" / "aligned"


def is_val(reg_no: str) -> bool:
    return int(hashlib.md5(reg_no.encode()).hexdigest(), 16) % 5 == 0   # ~20% of students


def main():
    ALIGNED.mkdir(exist_ok=True)
    rows = list(csv.DictReader(open(ROOT / "data" / "labels_all.csv", encoding="utf-8-sig")))
    X, y, meta = [], [], []
    failed = []
    for r in rows:
        cache = ALIGNED / (Path(r["file"]).stem + ".png")
        if cache.exists():
            a = cv2.imread(str(cache))
        else:
            res = layout.align(load_image(ROOT / "data" / "samples" / r["file"]))
            if res is None:
                failed.append(r["file"]); continue
            a = res.image
            cv2.imwrite(str(cache), a)
        mask = cells.ink_mask(a)
        for i in range(11):
            label = r["sub_total"] if i == 10 else r[f"q{i + 1}"]
            if i == 10 and not label.strip():
                continue   # sub-total not transcribed: unknown, not "blank"
            X.append(cells.row_strip(mask, i))
            y.append(label.strip())
            meta.append((r["file"], r["reg_no"], i, int(cells.is_simple(mask, i)), int(is_val(r["reg_no"]))))
    np.savez_compressed(ROOT / "data" / "cells.npz", X=np.array(X), y=np.array(y),
                        sheet=np.array([m[0] for m in meta]), reg=np.array([m[1] for m in meta]),
                        row=np.array([m[2] for m in meta]), simple=np.array([m[3] for m in meta]),
                        val=np.array([m[4] for m in meta]))
    print("unaligned (skipped):", failed)
    print("rows:", len(y), "| students val:", len({m[1] for m in meta if m[4]}), "of", len({m[1] for m in meta}))
    print("filled rows:", sum(1 for v in y if v), "| simple among filled:", sum(1 for v, m in zip(y, meta) if v and m[3]))


if __name__ == "__main__":
    main()
