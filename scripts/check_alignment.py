"""Align every sheet photo to the template; report failures and save cell-box overlays.

Usage: python scripts/check_alignment.py [samples_dir] [out_dir] [--overlay-every N]
"""
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from grade_eval.vision import layout
from grade_eval.vision.preprocess import load_image

ROOT = Path(__file__).resolve().parents[1]


def draw_overlay(aligned):
    out = aligned.copy()
    for (q, part), (x0, y0, x1, y1) in layout.cell_boxes().items():
        color = (0, 160, 0) if part == "a" else (200, 0, 200)
        cv2.rectangle(out, (x0, y0), (x1, y1), color, 1)
    for box in (layout.REGISTER_BOX, layout.NAME_BOX):
        cv2.rectangle(out, box[:2], box[2:], (0, 0, 255), 2)
    return out


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    samples = Path(args[0]) if args else ROOT / "data" / "samples"
    out_dir = Path(args[1]) if len(args) > 1 else ROOT / "data" / "alignment_check"
    every = int(sys.argv[sys.argv.index("--overlay-every") + 1]) if "--overlay-every" in sys.argv else 1
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(samples.glob("*.jpeg"))
    failed, rows = [], []
    for i, path in enumerate(files):
        try:
            res = layout.align(load_image(path))
        except Exception as e:  # unreadable image
            failed.append((path.name, f"error: {e}"))
            continue
        if res is None:
            failed.append((path.name, "no match"))
            continue
        rows.append((path.name, res.inliers))
        if i % every == 0:
            cv2.imwrite(str(out_dir / f"{path.stem}.jpg"), draw_overlay(res.image))

    print(f"{len(rows)}/{len(files)} aligned")
    if rows:
        inl = sorted(r[1] for r in rows)
        print(f"inliers min {inl[0]}, median {inl[len(inl)//2]}, max {inl[-1]}")
        print("weakest:", [r for r in sorted(rows, key=lambda r: r[1])[:8]])
    for name, why in failed:
        print("FAILED", name, why)


if __name__ == "__main__":
    main()
