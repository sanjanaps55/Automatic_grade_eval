"""Register-number strip extraction and synthetic training data for the header reader."""
import random
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

H, W = 48, 384
REG_CROP = (270, 82, 900, 148)       # x0, y0, x1, y1 on the aligned template (handwriting band, no printed label)
CODES = ["BCE", "BAI", "BEE", "MCS", "MIS", "BRS", "MML", "MAI", "BIT", "BCI", "BCB", "BEC", "BME", "MAM"]
FONTS = ["Inkfree.ttf", "segoepr.ttf", "segoesc.ttf", "BRADHITC.TTF", "comic.ttf", "Candara.ttf",
         "LHANDW.TTF", "Gabriola.ttf", "BRUSHSCI.TTF", "calibri.ttf"]
FONT_DIR = Path("C:/Windows/Fonts")


def ink_strip(aligned: np.ndarray) -> np.ndarray:
    """Aligned sheet -> (H, W) float32 strip, ink = 0, paper = 1."""
    x0, y0, x1, y1 = REG_CROP
    g = cv2.cvtColor(aligned[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
    ink = ((cv2.medianBlur(g, 41).astype(np.int16) - g) > 28).astype(np.float32)
    return 1 - cv2.resize(ink, (W, H), interpolation=cv2.INTER_AREA)


NAME_CROP = (470, 152, 920, 212)     # x0, y0, x1, y1 on the aligned template; past the "Mr/Ms" label, wide for long names (check data/name_preview.png)
NAME_H, NAME_W = 64, 512


def name_strip(aligned: np.ndarray) -> np.ndarray:
    """Aligned sheet -> (NAME_H, NAME_W) float32 strip, ink = 0, paper = 1."""
    x0, y0, x1, y1 = NAME_CROP
    g = cv2.cvtColor(aligned[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
    ink = ((cv2.medianBlur(g, 41).astype(np.int16) - g) > 28).astype(np.float32)
    return 1 - cv2.resize(ink, (NAME_W, NAME_H), interpolation=cv2.INTER_AREA)


def random_reg() -> str:
    code = random.choice(CODES) if random.random() < 0.9 else "".join(random.choices("ABCDEFGHIJKLMNOPQRSTUVWXYZ", k=3))
    yr = random.choice(["21", "22", "23", "23", "24", "24", "25"])
    return yr + code + "".join(random.choices("0123456789", k=4))


def synth_strip(text: str | None = None):
    """Render a register number in a handwriting-like font with wobble, dotted line and stray ink. -> (strip, text)"""
    text = text or random_reg()
    ch, cw = 66, 630
    img = Image.new("L", (cw, ch), 255)
    d = ImageDraw.Draw(img)
    size = random.randint(26, 40)
    font = ImageFont.truetype(str(FONT_DIR / random.choice(FONTS)), size)
    x = random.randint(20, 90); base = random.randint(8, 22)
    for c in text:
        w = d.textlength(c, font=font)
        glyph = Image.new("L", (int(w) + 24, size + 30), 255)
        ImageDraw.Draw(glyph).text((8, 4), c, font=font, fill=random.randint(0, 70))
        glyph = glyph.rotate(random.uniform(-9, 9), resample=Image.BILINEAR, fillcolor=255)
        img.paste(glyph, (int(x), base + random.randint(-4, 4)), Image.eval(glyph, lambda v: 255 - v))
        x += w * random.uniform(0.85, 1.3) + random.randint(-1, 5)
    a = np.array(img)
    for xx in range(0, cw, random.choice([6, 7, 8])):          # dotted guide line
        a[min(ch - 1, random.randint(54, 60)), xx:xx + 2] = random.randint(60, 140)
    if random.random() < 0.5:                                  # name ink bleeding in from below
        for _ in range(random.randint(3, 9)):
            xs = random.randint(0, cw - 30); cv2.line(a, (xs, ch - 1), (xs + random.randint(5, 25), ch - random.randint(3, 14)), 40, 2)
    if random.random() < 0.3:                                  # tick mark / underline
        cv2.line(a, (random.randint(60, 300), random.randint(48, 56)), (random.randint(300, 500), random.randint(48, 56)), 60, 1)
    a = cv2.GaussianBlur(a, (0, 0), random.uniform(0.3, 1.0))
    if random.random() < 0.5:
        a = cv2.erode(a, np.ones((2, 2), np.uint8)) if random.random() < 0.5 else cv2.dilate(a, np.ones((2, 2), np.uint8))
    ink = np.clip((255 - a.astype(np.float32)) / 150, 0, 1)
    return 1 - cv2.resize(ink, (W, H), interpolation=cv2.INTER_AREA), text
