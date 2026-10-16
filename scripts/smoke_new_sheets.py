"""Smoke test: align every photo in a folder, read the reg-no (TrOCR-small) and the 10 marks rows (ResNet18+CTC).

Usage: python scripts/smoke_new_sheets.py [folder=data/samples_new] [--out data/smoke_new.csv]
No labels needed; it just shows what the current models say plus the alignment rate. Aligned images are cached in data/aligned_new/.
"""
import csv
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from transformers import VisionEncoderDecoderModel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import finetune_header as fh
from train_marks import MarksNet, decode as ctc_decode
from grade_eval.vision import cells, header, layout
from grade_eval.vision.preprocess import load_image

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    folder = Path(args[0]) if args else ROOT / "data" / "samples_new"
    out = Path(fh.arg("--out", str(ROOT / "data" / "smoke_new.csv")))
    cache = ROOT / "data" / "aligned_new"; cache.mkdir(exist_ok=True)
    fh.DEV = DEV

    tok = fh.load_tokenizer(str(ROOT / "models" / "trocr_reg_small"))
    reg_model = VisionEncoderDecoderModel.from_pretrained(ROOT / "models" / "trocr_reg_small", torch_dtype=torch.float32).to(DEV).eval()
    marks = MarksNet(pretrained=False); marks.load_state_dict(torch.load(ROOT / "models" / "marks.pt", map_location="cpu")); marks.eval()

    rows, failed = [], []
    for p in sorted(folder.glob("*.jp*g")):
        t0 = time.time()
        res = layout.align(load_image(p))
        if res is None:
            failed.append(p.name); rows.append([p.name, 0, "", ""] + [""] * 10); continue
        cv2.imwrite(str(cache / (p.stem + ".png")), res.image)
        a = res.image
        strip = torch.tensor(header.ink_strip(a)).unsqueeze(0).unsqueeze(0)
        reg = fh.decode(reg_model, tok, strip.to(DEV), "reg")[0]
        mask = cells.ink_mask(a)
        X = torch.tensor(np.stack([cells.row_strip(mask, i) for i in range(11)]).astype(np.float32) / 255).unsqueeze(1)
        with torch.no_grad():
            m = ctc_decode(marks(X))
        marks_txt = [f"{s}({c:.2f})" if s else "" for s, c in m]
        rows.append([p.name, res.inliers, reg, f"{time.time() - t0:.1f}s"] + marks_txt)
        print(p.name, res.inliers, reg, marks_txt, flush=True)

    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["file", "inliers", "reg_pred", "time"] + [f"q{i}" for i in range(1, 11)] + ["sub_total"])
        w.writerows([r + [""] * (15 - len(r)) for r in rows])
    print(f"aligned {len(rows) - len(failed)}/{len(rows)}; failed: {failed}")


if __name__ == "__main__":
    main()
