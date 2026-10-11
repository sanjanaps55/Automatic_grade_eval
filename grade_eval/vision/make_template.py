"""One-off: build the reference form image used to align every photo.

Sample 0 was shot nearly head-on. Its four outer grid corners (read off the image by hand,
in `load_image` coordinates) are mapped to an upright rectangle, with a margin kept around the
grid so header text and side boxes contribute alignment features.
"""
import sys
from pathlib import Path

import cv2
import numpy as np

from .preprocess import GRID_H, GRID_W, load_image

MARGIN = dict(l=120, t=420, r=420, b=160)
TEMPLATE_PATH = Path(__file__).resolve().parent.parent / "assets" / "template.png"
# tl, tr, br, bl of the marks table in sample 0
SRC_CORNERS = np.array([[238, 628], [1036, 626], [1036, 1770], [228, 1768]], np.float32)


def build(sample_path: str, out: Path = TEMPLATE_PATH):
    img = load_image(sample_path)
    ml, mt = MARGIN["l"], MARGIN["t"]
    dst = np.array([[ml, mt], [ml + GRID_W, mt], [ml + GRID_W, mt + GRID_H], [ml, mt + GRID_H]], np.float32)
    size = (GRID_W + ml + MARGIN["r"], GRID_H + mt + MARGIN["b"])
    M = cv2.getPerspectiveTransform(SRC_CORNERS, dst)
    cv2.imwrite(str(out), cv2.warpPerspective(img, M, size, flags=cv2.INTER_CUBIC))


if __name__ == "__main__":
    build(sys.argv[1])
