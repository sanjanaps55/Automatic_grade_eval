"""Alignment success on re-compressed copies, with and without upscaling small photos before feature matching."""
import csv, sys
from pathlib import Path
import cv2, numpy as np
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from grade_eval.vision import layout
from grade_eval.vision.preprocess import load_image
labels = [r["file"] for r in csv.DictReader(open(ROOT / "data" / "labels_all.csv", encoding="utf-8-sig"))]
def degrade(path, side, q):
    im = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR); h, w = im.shape[:2]; s = min(1.0, side / max(h, w))
    if s < 1: im = cv2.resize(im, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
    return cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, q])[1].tobytes()
def close(a, b):
    g = lambda x: cv2.resize(cv2.cvtColor(x, cv2.COLOR_BGR2GRAY), (96, 124)).astype(np.float32)
    A, B = g(a), g(b); A -= A.mean(); B -= B.mean()
    return float((A * B).sum() / (np.linalg.norm(A) * np.linalg.norm(B) + 1e-6))
def align_up(img, up_to=2000, minin=30):
    h, w = img.shape[:2]; s = up_to / max(h, w)
    if s > 1.05: img = cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_CUBIC)
    old = layout.MIN_INLIERS; layout.MIN_INLIERS = minin
    try: return layout.align(img)
    finally: layout.MIN_INLIERS = old
res = {}
for f in labels:
    ref = cv2.imread(str(ROOT / "data" / "aligned_eval" / (Path(f).stem + ".png")))
    if ref is None: continue
    for name, (side, q) in {"q85": (2000, 85), "q60": (1280, 60), "q50_1000": (1000, 50)}.items():
        data = degrade(ROOT / "data" / "samples" / f, side, q); img = load_image(data)
        for mode, fn in {"current": lambda im: layout.align(im), "upscale": lambda im: align_up(im), "upscale_min20": lambda im: align_up(im, 2000, 20)}.items():
            r = fn(img); ok = r is not None; good = ok and close(r.image, ref) > 0.85
            d = res.setdefault((name, mode), [0, 0, 0]); d[0] += 1; d[1] += ok; d[2] += good
for k, (n, ok, good) in sorted(res.items()):
    print(f"{k[0]:9s} {k[1]:14s} aligned {ok}/{n}  aligned AND matches the original alignment {good}/{n}")
