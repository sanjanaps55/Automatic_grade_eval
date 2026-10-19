"""Phone-quality copies of the labelled sheets -> row strips (data/cells_deg.npz), for compression-robust training.

A gallery upload is often a smaller / more compressed copy of the camera photo and the marks reader gets less sure on those
(see PLAN.md). Each sheet is re-encoded (long side + JPEG quality), aligned exactly like a real upload, and cut into the same
row strips as data/cells.npz. Labels are NOT stored: they are the labels of the same (sheet, row) in cells.npz.
  train sheets: 4 random variants each (used as augmentation)      val/test sheets: 2 fixed variants (used only for scoring)
    python scripts/build_cells_degraded.py
"""
import csv
import sys
import zlib
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grade_eval.vision import cells, layout
from grade_eval.vision.preprocess import load_image

FIXED = [(2000, 85), (1600, 75)]          # scoring variants for val/test (same as the PLAN.md robustness check)


def degrade(raw: bytes, side: int, quality: int) -> bytes:
    im = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    s = min(1.0, side / max(im.shape[:2]))
    if s < 1:
        im = cv2.resize(im, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    return cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, quality])[1].tobytes()


def main():
    split = {r["file"]: r["split"] for r in csv.DictReader(open(ROOT / "data" / "split.csv", encoding="utf-8-sig"))}
    X, sheet, row, var, sp = [], [], [], [], []
    for f, s in split.items():
        raw = (ROOT / "data" / "samples" / f).read_bytes()
        rng = np.random.default_rng(zlib.crc32(f.encode()))
        cfgs = [(int(rng.integers(1200, 3001)), int(rng.integers(55, 96))) for _ in range(4)] if s == "train" else FIXED
        for v, (side, q) in enumerate(cfgs):
            res = layout.align(load_image(degrade(raw, side, q)))
            if res is None:
                print("not aligned:", f, side, q); continue
            mask = cells.ink_mask(res.image)
            for i in range(10):
                X.append(cells.row_strip(mask, i)); sheet.append(f); row.append(i); var.append(v); sp.append(s)
        print(f, s, len(cfgs), "variants", flush=True)
    np.savez_compressed(ROOT / "data" / "cells_deg.npz", X=np.array(X), sheet=np.array(sheet), row=np.array(row), variant=np.array(var), split=np.array(sp))
    print("strips:", len(X))


if __name__ == "__main__":
    main()
