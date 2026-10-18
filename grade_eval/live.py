"""Live scanning: per-frame framing check used by the phone page (the phone only streams frames, all work is here).

check_frame() answers one of: align (form not found) | back (form found but part of the page is outside the frame) |
blurry | ok. The phone captures a full-resolution frame once a few consecutive frames are ok and the phone is steady.
"""
import time

import cv2
import numpy as np

from grade_eval.vision import layout

CHECK_SIDE = 1800         # frames are checked at this size; cheaper SIFT settings / smaller frames stopped finding the form (tested on synthetic phone frames)
SHARP_SIDE = 960          # sharpness is measured at this size so MIN_SHARP does not depend on CHECK_SIDE
MAX_OUTSIDE = 0.05        # fraction of the aligned page that lies outside the photo before we say "move back"
MIN_SHARP = 200.0         # Laplacian variance of the frame centre below which it is too blurry (sharp samples >= 880, sigma-1 blur <= 90)
MIN_FILL = 0.6            # the sheet must span at least this fraction of the frame's width or height
MAX_EDGE_RATIO = 1.14     # opposite edges of the sheet may differ by this factor before we say the phone is tilted
MAX_ROTATION = 7.0        # degrees

MESSAGES = {
    "align": "Can't match the sheet. Fill the frame with the front page, hold the phone flat and still, avoid glare and shadows",
    "back": "Move the phone back: the whole sheet must be visible",
    "small": "Move closer: the sheet should fill the frame",
    "tilt": "Hold the phone flat, parallel to the sheet",
    "rotated": "Straighten the sheet (it looks rotated)",
    "blurry": "Hold steady: the picture is blurry",
    "ok": "Good. Hold still",
}
CUT = {"left": "left", "right": "right", "top": "up", "bottom": "down"}   # sheet cut off on this side -> move the phone that way


def sheet_quad(res, shape) -> np.ndarray:
    """The template's four corners mapped into the frame, as fractions of the frame (TL, TR, BR, BL)."""
    th, tw = layout._get_template()[0].shape[:2]
    corners = np.float32([[0, 0], [tw, 0], [tw, th], [0, th]]).reshape(-1, 1, 2)
    q = cv2.perspectiveTransform(corners, np.linalg.inv(res.H)).reshape(4, 2)
    return q / np.float32([shape[1], shape[0]])


def advise(q: np.ndarray, outside: float, sharp: float) -> str:
    """Pick the single most useful instruction from the sheet outline `q` (fractions of the frame)."""
    cut = {"left": -q[:, 0].min(), "right": q[:, 0].max() - 1, "top": -q[:, 1].min(), "bottom": q[:, 1].max() - 1}
    cut = [k for k, v in cut.items() if v > 0.02]
    if outside > MAX_OUTSIDE or cut:
        if len(cut) == 1:
            return "back", f"The sheet is cut off at the {cut[0]}: move the phone {CUT[cut[0]]} a little"
        return "back", MESSAGES["back"]
    if max(np.ptp(q[:, 0]), np.ptp(q[:, 1])) < MIN_FILL:
        return "small", MESSAGES["small"]
    top, bot = np.linalg.norm(q[1] - q[0]), np.linalg.norm(q[2] - q[3])
    lef, rig = np.linalg.norm(q[3] - q[0]), np.linalg.norm(q[2] - q[1])
    if max(top / bot, bot / top, lef / rig, rig / lef) > MAX_EDGE_RATIO:
        return "tilt", MESSAGES["tilt"]
    dx, dy = q[1] - q[0]
    if abs(np.degrees(np.arctan2(dy, dx))) > MAX_ROTATION:
        return "rotated", MESSAGES["rotated"]
    if sharp < MIN_SHARP:
        return "blurry", MESSAGES["blurry"]
    return "ok", MESSAGES["ok"]


def measure(img: np.ndarray):
    """-> (outside_fraction of the aligned page, sharpness of the raw frame centre, feature inliers, sheet outline), or None when the form is not found."""
    res = layout.align(img)
    if res is None:
        return None
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    k = SHARP_SIDE / max(g.shape)
    if k < 1:
        g = cv2.resize(g, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
    h, w = g.shape
    outside = float((res.image.max(axis=2) == 0).mean())
    return outside, float(cv2.Laplacian(g[h // 5:4 * h // 5, w // 5:4 * w // 5], cv2.CV_64F).var()), res.inliers, sheet_quad(res, img.shape)


def save_debug_frame(data: bytes) -> None:
    """Keep the latest test-page frame so a "not detected" case can be inspected / replayed."""
    d = CERT_DIR.parent / "live_debug"
    d.mkdir(parents=True, exist_ok=True)
    (d / "last_frame.jpg").write_bytes(data)


def check_frame(data: bytes) -> dict:
    t0 = time.time()
    r = _check_frame(data)
    r["ms"] = int((time.time() - t0) * 1000)
    return r


def _check_frame(data: bytes) -> dict:
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return {"state": "align", "msg": MESSAGES["align"], "quad": None}
    s = CHECK_SIDE / max(img.shape[:2])
    if s < 1:
        img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    m = measure(img)
    if m is None:
        return {"state": "align", "msg": MESSAGES["align"], "quad": None, "outside": None, "sharp": None, "inliers": None}
    state, msg = advise(m[3], m[0], m[1])
    return {"state": state, "msg": msg, "quad": np.round(m[3], 4).tolist(), "outside": round(m[0], 3),
            "sharp": round(m[1], 1), "inliers": m[2]}


# ------------------------------------------------------------------ network helpers (phone link + https)
import datetime
import ipaddress
import socket
from pathlib import Path

CERT_DIR = Path(__file__).resolve().parents[1] / "data" / "certs"
HTTP_PORT, HTTPS_PORT = 8000, 8443


def lan_ip() -> str:
    """The PC's address on the Wi-Fi/LAN (the interface with the default route); no packet is sent."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("10.255.255.255", 1))
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


def cert_files():
    return CERT_DIR / "cert.pem", CERT_DIR / "key.pem"


def ensure_cert() -> tuple[Path, Path]:
    """Self-signed certificate for localhost + this PC's LAN address. Phones only allow the camera on https pages."""
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import NameOID

    cert, key = cert_files()
    ip = lan_ip()
    stamp = CERT_DIR / "ip.txt"
    if cert.exists() and key.exists() and stamp.exists() and stamp.read_text() == ip:
        return cert, key
    CERT_DIR.mkdir(parents=True, exist_ok=True)
    k = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "grade-eval")])
    now = datetime.datetime.now(datetime.timezone.utc)
    san = x509.SubjectAlternativeName([x509.DNSName("localhost"), x509.IPAddress(ipaddress.ip_address("127.0.0.1")),
                                       x509.IPAddress(ipaddress.ip_address(ip))])
    c = (x509.CertificateBuilder().subject_name(name).issuer_name(name).public_key(k.public_key())
         .serial_number(x509.random_serial_number()).not_valid_before(now - datetime.timedelta(days=1))
         .not_valid_after(now + datetime.timedelta(days=825)).add_extension(san, critical=False).sign(k, hashes.SHA256()))
    cert.write_bytes(c.public_bytes(serialization.Encoding.PEM))
    key.write_bytes(k.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.TraditionalOpenSSL,
                                    serialization.NoEncryption()))
    (CERT_DIR / "ip.txt").write_text(ip)
    return cert, key


def page_html(slug: str, test: bool = False) -> str:
    """test=True: framing diagnostics only (no sheet is captured or saved), for trying the camera setup."""
    import html
    return ((Path(__file__).parent / "assets" / "live.html").read_text(encoding="utf-8")
            .replace("__SLUG__", html.escape(slug, quote=True)).replace("__TEST__", "true" if test else "false"))
