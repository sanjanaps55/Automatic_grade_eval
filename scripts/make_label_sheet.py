"""Build label-helper panels: header strip + marks table per sheet, several sheets per image.

Usage: python scripts/make_label_sheet.py [out_dir] [--all] [--per-image N]
By default only sheets missing from data/labels.csv are included.
"""
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from grade_eval.vision import layout
from grade_eval.vision.preprocess import load_image

ROOT = Path(__file__).resolve().parents[1]
HEADER = (100, 95, 760, 205)       # register no + name, template coords
TABLE = (100, 520, 1135, 1835)     # full marks table incl. sub-total row
W = 900


def panel(path: Path) -> np.ndarray:
    img = load_image(path)
    res = layout.align(img)
    if res is None:  # unaligned: show the whole photo so it can still be read by eye
        h = int(img.shape[0] * W / img.shape[1])
        body = cv2.resize(img, (W, h))
        tag = "UNALIGNED"
    else:
        a = res.image
        hx0, hy0, hx1, hy1 = HEADER
        tx0, ty0, tx1, ty1 = TABLE
        head = a[hy0:hy1, hx0:hx1]
        tab = a[ty0:ty1, tx0:tx1]
        head = cv2.resize(head, (W, int(head.shape[0] * W / head.shape[1])))
        tab = cv2.resize(tab, (W, int(tab.shape[0] * W / tab.shape[1])))
        body = np.vstack([head, tab])
        tag = ""
    cv2.putText(body, f"{path.stem} {tag}", (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    return body


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    out = Path(args[0]) if args else ROOT / "data" / "label_panels"
    per = int(sys.argv[sys.argv.index("--per-image") + 1]) if "--per-image" in sys.argv else 2
    out.mkdir(parents=True, exist_ok=True)
    done = set()
    lab = ROOT / "data" / "labels.csv"
    if "--all" not in sys.argv and lab.exists():
        done = {r["file"] for r in csv.DictReader(open(lab, encoding="utf-8-sig"))}
    files = [p for p in sorted((ROOT / "data" / "samples").glob("*.jpeg")) if p.name not in done]
    for i in range(0, len(files), per):
        ps = [panel(p) for p in files[i:i + per]]
        H = max(p.shape[0] for p in ps)
        ps = [cv2.copyMakeBorder(p, 0, H - p.shape[0], 0, 8, cv2.BORDER_CONSTANT, value=(255, 255, 255)) for p in ps]
        cv2.imwrite(str(out / f"panel_{i // per:03d}.jpg"), np.hstack(ps), [cv2.IMWRITE_JPEG_QUALITY, 90])
    print(len(files), "sheets ->", out)


if __name__ == "__main__":
    main()
