"""Cut name strips from the aligned sheets -> data/name_real.npz (X uint8, y names, reg_no, split) + data/name_preview.png.

Usage: python scripts/export_name_data.py   (needs data/aligned/ from build_cells.py)
Look at data/name_preview.png before training: if names are cut off, edit header.NAME_CROP.
"""
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grade_eval.split import NAMES, split_of
from grade_eval.vision import header


def main():
    X, y, reg, sp = [], [], [], []
    for r in csv.DictReader(open(ROOT / "data" / "labels_all.csv", encoding="utf-8-sig")):
        p = ROOT / "data" / "aligned" / (Path(r["file"]).stem + ".png")
        if p.exists() and r["name"].strip():
            X.append(header.name_strip(cv2.imread(str(p)))); y.append(r["name"].strip()); reg.append(r["reg_no"])
            sp.append(NAMES[split_of(r["reg_no"])])
    X = (np.stack(X) * 255).astype(np.uint8)
    np.savez_compressed(ROOT / "data" / "name_real.npz", X=X, y=np.array(y), reg_no=np.array(reg), split=np.array(sp))
    cv2.imwrite(str(ROOT / "data" / "name_preview.png"), np.vstack(list(X[:12])))
    print(len(y), "name strips:", {s: sp.count(s) for s in set(sp)})


if __name__ == "__main__":
    main()
