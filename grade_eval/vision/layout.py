"""Align a photo of the cover page to the reference template and cut out the marks cells."""
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

TEMPLATE_PATH = Path(__file__).resolve().parent.parent / "assets" / "template.png"
MIN_INLIERS = 30

# Measured on the template image (see make_template.py). Row boundaries: 10 question rows + sub-total row.
ROW_Y = [539, 656, 772, 888, 1005, 1122, 1239, 1356, 1472, 1588, 1706, 1825]
# Per group of 10 questions: x of (Q-No left edge, a left, b left, b right).
GROUP_X = [(122, 208, 290, 373), (373, 456, 539, 622), (622, 705, 788, 871), (871, 954, 1037, 1121)]
# Header band (printed name/register fields) in template coordinates, as (x0, y0, x1, y1).
REGISTER_BOX = (190, 105, 700, 150)
NAME_BOX = (360, 150, 700, 190)

_sift = cv2.SIFT_create(nfeatures=5000)
_template = None


def _gray(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(g)


def _get_template():
    global _template
    if _template is None:
        img = cv2.imread(str(TEMPLATE_PATH))
        kp, des = _sift.detectAndCompute(_gray(img), None)
        _template = (img, kp, des)
    return _template


@dataclass
class Alignment:
    image: np.ndarray   # photo warped into template coordinates
    inliers: int
    H: np.ndarray | None = None   # photo -> template homography


def align(img: np.ndarray) -> Alignment | None:
    """Warp the photo into the template frame, or None if the page can't be matched."""
    tpl, tkp, tdes = _get_template()
    kp, des = _sift.detectAndCompute(_gray(img), None)
    if des is None or len(kp) < MIN_INLIERS:
        return None
    pairs = cv2.BFMatcher().knnMatch(des, tdes, k=2)
    good = [m for m, n in (p for p in pairs if len(p) == 2) if m.distance < 0.75 * n.distance]
    if len(good) < MIN_INLIERS:
        return None
    src = np.float32([kp[m.queryIdx].pt for m in good])
    dst = np.float32([tkp[m.trainIdx].pt for m in good])
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 4.0)
    if H is None or int(mask.sum()) < MIN_INLIERS:
        return None
    th, tw = tpl.shape[:2]
    return Alignment(cv2.warpPerspective(img, H, (tw, th), flags=cv2.INTER_CUBIC), int(mask.sum()), H)


def cell_boxes() -> dict:
    """{(q_no, part): (x0, y0, x1, y1)} for every question mark cell; q_no 0 = group sub-totals use ('sub', g)."""
    boxes = {}
    for g, (ql, al, bl, br) in enumerate(GROUP_X):
        for r in range(10):
            y0, y1 = ROW_Y[r], ROW_Y[r + 1]
            q = g * 10 + r + 1
            boxes[(q, "a")] = (al, y0, bl, y1)
            boxes[(q, "b")] = (bl, y0, br, y1)
        boxes[(("sub", g), "a")] = (al, ROW_Y[10], bl, ROW_Y[11])
        boxes[(("sub", g), "b")] = (bl, ROW_Y[10], br, ROW_Y[11])
    return boxes


def crop(img: np.ndarray, box, pad: int = 0) -> np.ndarray:
    x0, y0, x1, y1 = box
    return img[y0 - pad:y1 + pad, x0 - pad:x1 + pad]
