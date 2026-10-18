"""Read one answer-sheet photo with the finalised models: marks per row (marks.pt) + register number (trocr_reg_small).

Usage: python scripts/read_sheet.py path/to/photo.jpeg [--no-reg]
Needs models/marks.pt and models/trocr_reg_small/ (see README "Models"). Rows 1-10 = questions, row 11 = sub-total.
Low-confidence reads are marked REVIEW; these should go to manual review, not Excel.
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from grade_eval.vision import cells, header, layout
from grade_eval.vision.preprocess import load_image
from train_marks import MarksNet, decode

REVIEW_BELOW = 0.9


def main():
    path = Path(sys.argv[1])
    res = layout.align(load_image(path))
    if res is None:
        sys.exit("could not align this photo (curled page / no form found) -> manual review")
    aligned = res.image

    net = MarksNet(pretrained=False); net.load_state_dict(torch.load(ROOT / "models" / "marks.pt", map_location="cpu")); net.eval()
    mask = cells.ink_mask(aligned)
    X = torch.tensor(np.stack([cells.row_strip(mask, i) for i in range(11)])).float()
    if X.max() > 1: X = X / 255
    with torch.no_grad(): reads = decode(net(X.unsqueeze(1) if X.ndim == 3 else X))
    print("marks:")
    for i, (txt, conf) in enumerate(reads):
        simple = cells.is_simple(mask, i)
        flag = "" if simple and conf >= REVIEW_BELOW else "   <- REVIEW" + ("" if simple else " (a|b or sum row)")
        print(f"  {'sub-total' if i == 10 else f'Q{i + 1}':9s} {txt or '-':>5s}  conf {conf:.2f}{flag}")

    if "--no-reg" not in sys.argv:
        import finetune_header as fh
        from transformers import VisionEncoderDecoderModel
        fh.DEV = "cpu"; mp = str(ROOT / "models" / "trocr_reg_small")
        tok = fh.load_tokenizer(mp); model = VisionEncoderDecoderModel.from_pretrained(mp, torch_dtype=torch.float32).eval()
        strip = torch.tensor(header.ink_strip(aligned)).float()[None, None]
        print("register number:", fh.decode(model, tok, strip, "reg")[0])


if __name__ == "__main__":
    main()
