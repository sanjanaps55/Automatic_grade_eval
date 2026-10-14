"""Turn the header npz files into PaddleOCR recognition data (PNG strips + tab-separated label lists).

Usage: python scripts/prep_paddle_data.py [--syn 6000] [--out paddle_data]
Writes <out>/{reg,name}/{train,val,test}.txt and <out>/{reg,name}/img/*.png  (lines: "img/xxx.png<TAB>text").
Same student split as finetune_header.py (grade_eval/split.py). Register number: real train strips x4 + synthetic.
Name: real train strips x3 (no synthetic cursive).
"""
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from finetune_header import load


def arg(name, default):
    a = sys.argv
    return type(default)(a[a.index(name) + 1]) if name in a else default


def write(out: Path, field: str, n_syn: int):
    X, y, sp, SX, SY = load(field)
    d = out / field; (d / "img").mkdir(parents=True, exist_ok=True)
    X = (X[:, 0].numpy() * 255).astype(np.uint8)
    lists = {"train": [], "val": [], "test": []}

    def save(img, text, key, name):
        # PP-OCR rec resizes to height 48; upscale small strips so thin strokes survive
        img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(str(d / "img" / f"{name}.png"), img)
        lists[key].append(f"img/{name}.png\t{text}")

    rep = {0: 4 if field == "reg" else 3, 1: 1, 2: 1}
    for i in range(len(y)):
        key = ["train", "val", "test"][sp[i]]
        for r in range(rep[sp[i]]):
            save(X[i], y[i], key, f"{key}_{i}_{r}")
    if field == "reg" and n_syn:
        S = (SX[:, 0].numpy() * 255).astype(np.uint8)
        for j in np.random.RandomState(0).choice(len(SY), n_syn, replace=False):
            save(S[j], SY[j], "train", f"syn_{j}")
    for k, v in lists.items():
        (d / f"{k}.txt").write_text("\n".join(v) + "\n", encoding="utf-8")
    print(field, {k: len(v) for k, v in lists.items()})


if __name__ == "__main__":
    out = Path(arg("--out", "paddle_data")); n_syn = arg("--syn", 6000)
    for f in ("reg", "name"):
        write(out, f, n_syn)
