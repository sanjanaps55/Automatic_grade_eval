"""Zero-shot TrOCR on the register-number crop of every aligned labelled sheet; reports exact match.

Usage: python scripts/eval_header.py [model_name]   (default microsoft/trocr-small-handwritten)
"""
import csv
import re
import sys
from pathlib import Path

import cv2
import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grade_eval.validate import repair_reg_no

REG_CROP = (270, 70, 900, 165)   # x0, y0, x1, y1 on the aligned template; skips the printed "Register No :"


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "microsoft/trocr-small-handwritten"
    proc = TrOCRProcessor.from_pretrained(name); model = VisionEncoderDecoderModel.from_pretrained(name).eval()
    rows = list(csv.DictReader(open(ROOT / "data" / "labels_all.csv", encoding="utf-8-sig")))
    n = ok = fmt = 0; bad = []
    for r in rows:
        p = ROOT / "data" / "aligned" / (Path(r["file"]).stem + ".png")
        if not p.exists(): continue
        x0, y0, x1, y1 = REG_CROP
        img = Image.fromarray(cv2.cvtColor(cv2.imread(str(p))[y0:y1, x0:x1], cv2.COLOR_BGR2RGB))
        with torch.no_grad():
            ids = model.generate(proc(images=img, return_tensors="pt").pixel_values, max_new_tokens=16)
        raw = proc.batch_decode(ids, skip_special_tokens=True)[0]
        pred = repair_reg_no(raw); n += 1
        fmt += bool(re.fullmatch(r"\d{2}[A-Z]{3}\d{4}", pred)); ok += pred == r["reg_no"]
        if pred != r["reg_no"]: bad.append((r["file"], r["reg_no"], raw, pred))
    print(f"{name}: exact {ok}/{n} = {ok / n:.3f} | valid format {fmt}/{n}")
    for b in bad[:30]: print(b)


if __name__ == "__main__":
    main()
