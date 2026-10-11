import cv2
import numpy as np

MAX_SIDE = 2000


def load_image(path_or_bytes) -> np.ndarray:
    """Read an image (path or raw bytes), apply EXIF-free decode, and cap the long side."""
    if isinstance(path_or_bytes, (bytes, bytearray)):
        img = cv2.imdecode(np.frombuffer(path_or_bytes, np.uint8), cv2.IMREAD_COLOR)
    else:
        img = cv2.imdecode(np.fromfile(str(path_or_bytes), np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("could not decode image")
    h, w = img.shape[:2]
    scale = MAX_SIDE / max(h, w)
    if scale < 1:
        img = cv2.resize(img, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
    return img


def line_mask(img: np.ndarray) -> np.ndarray:
    """Binary mask of printed grid lines (long horizontal/vertical strokes)."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 12)
    h, w = bw.shape
    hk = cv2.getStructuringElement(cv2.MORPH_RECT, (max(w // 25, 25), 1))
    vk = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(h // 25, 25)))
    horiz = cv2.morphologyEx(bw, cv2.MORPH_OPEN, hk)
    vert = cv2.morphologyEx(bw, cv2.MORPH_OPEN, vk)
    return cv2.bitwise_or(horiz, vert)


def _split_line_masks(img: np.ndarray):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 12)
    h, w = bw.shape
    hk = cv2.getStructuringElement(cv2.MORPH_RECT, (max(w // 25, 25), 1))
    vk = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(h // 25, 25)))
    return cv2.morphologyEx(bw, cv2.MORPH_OPEN, hk), cv2.morphologyEx(bw, cv2.MORPH_OPEN, vk)


def _hline_segments(horiz: np.ndarray, min_w: float, max_w: float):
    """Long horizontal rules as (left_pt, right_pt) pairs, filtered by width."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(horiz)
    segs = []
    for i in range(1, n):
        x, y, w, h, _ = stats[i]
        if min_w <= w <= max_w:
            ys, xs = np.where(labels[y:y + h, x:x + w] == i)
            l, r = xs.argmin(), xs.argmax()
            segs.append(((x + xs[l], y + ys[l]), (x + xs[r], y + ys[r])))
    return segs


def find_grid_corners(img: np.ndarray) -> np.ndarray | None:
    """Corners (tl, tr, br, bl) of the marks table, from its long horizontal rules.

    The table rules share one width and left edge; the side boxes and page border differ.
    """
    horiz, _ = _split_line_masks(img)
    W = img.shape[1]
    segs = _hline_segments(horiz, 0.30 * W, 0.62 * W)
    if len(segs) < 6:
        return None
    lefts = np.array([s[0][0] for s in segs], float)
    widths = np.array([s[1][0] - s[0][0] for s in segs], float)
    # keep the rules that agree with the most common (left edge, width)
    best = max(range(len(segs)), key=lambda i: np.sum(
        (abs(lefts - lefts[i]) < 0.04 * W) & (abs(widths - widths[i]) < 0.06 * W)))
    keep = [s for s, l, w in zip(segs, lefts, widths)
            if abs(l - lefts[best]) < 0.04 * W and abs(w - widths[best]) < 0.06 * W]
    if len(keep) < 6:
        return None
    keep.sort(key=lambda s: s[0][1])
    top, bot = keep[0], keep[-1]
    return np.array([top[0], top[1], bot[1], bot[0]], np.float32)


GRID_W, GRID_H = 1000, 1420


def warp_grid(img: np.ndarray, corners: np.ndarray) -> np.ndarray:
    dst = np.array([[0, 0], [GRID_W, 0], [GRID_W, GRID_H], [0, GRID_H]], np.float32)
    M = cv2.getPerspectiveTransform(corners, dst)
    return cv2.warpPerspective(img, M, (GRID_W, GRID_H), flags=cv2.INTER_CUBIC)
