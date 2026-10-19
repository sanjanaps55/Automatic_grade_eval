"""Cut mark cells out of an aligned sheet and turn them into small ink-only images for the recognizer."""
import cv2
import numpy as np

from . import layout

H, W = 64, 192            # recognizer input size
SPILL_X = (206, 450)      # row strip: column a, column b and the spill-over into the next group's Q-no column
A_X = (layout.GROUP_X[0][1], layout.GROUP_X[0][2])
B_X = (layout.GROUP_X[0][2], layout.GROUP_X[0][3])


def ink_mask(aligned: np.ndarray) -> np.ndarray:
    """Boolean mask of pen ink.

    Main rule: pixels whose red-green (Lab a*) value departs from the local paper colour; works for pale red/pink pen
    where a plain saturation threshold fails, and ignores the blue printed grid.
    Second rule (compression-robust): JPEG/downscaling washes the colour out of thin red strokes but they stay darker than the
    paper, so pixels that are clearly darker AND show a (blurred) red shift, and are not blue, count as ink too. It only ever
    adds pixels, so photos at full quality give practically the same mask as before.
    """
    L, a, b = cv2.split(cv2.cvtColor(aligned, cv2.COLOR_BGR2LAB))
    d = a.astype(np.int16) - cv2.medianBlur(a, 51).astype(np.int16)
    main = (d > 7) | (d < -12)   # red/pink ink, green ink
    dl = L.astype(np.int16) - cv2.medianBlur(L, 51).astype(np.int16)
    a_s, b_s = cv2.GaussianBlur(a, (0, 0), 1.5), cv2.GaussianBlur(b, (0, 0), 1.5)
    da = a_s.astype(np.int16) - cv2.medianBlur(a_s, 51).astype(np.int16)
    db = b_s.astype(np.int16) - cv2.medianBlur(b_s, 51).astype(np.int16)
    return main | ((dl < -22) & (da > 3) & (db > -6))


def row_box(row: int, x_range=SPILL_X, pad: int = 6):
    """Box for question row 0..9 (10 = sub-total row)."""
    return (x_range[0], layout.ROW_Y[row] + pad, x_range[1], layout.ROW_Y[row + 1] - pad)


def row_strip(mask: np.ndarray, row: int) -> np.ndarray:
    """Ink-only strip for one row, white background / dark ink, resized to (H, W) uint8."""
    x0, y0, x1, y1 = row_box(row)
    m = mask[y0:y1, x0:x1].astype(np.uint8) * 255
    return 255 - cv2.resize(m, (W, H), interpolation=cv2.INTER_AREA)


def ink_fraction(mask: np.ndarray, row: int, x_range) -> float:
    x0, y0, x1, y1 = row_box(row, x_range)
    return float(mask[y0:y1, x0:x1].mean())


def is_simple(mask: np.ndarray, row: int) -> bool:
    """True when the mark sits in column a only (no b entry or written sum spilling to the right)."""
    return ink_fraction(mask, row, (B_X[0] + 8, SPILL_X[1])) < 0.004
