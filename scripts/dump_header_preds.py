"""Run the saved TrOCR reg + name models on val/test sheets (CPU) and save per-sheet predictions.

Usage: python scripts/dump_header_preds.py [--out data/preds_trocr.csv]
Columns: split,reg_true,name_true,reg_pred,name_pred. score_roster.py reads this (and any other engine's file in the same format).
"""
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from transformers import VisionEncoderDecoderModel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import finetune_header as fh
from grade_eval.split import NAMES

fh.DEV = "cpu"


def predict(field, X):
    path = ROOT / "models" / f"trocr_{field}_small"
    tok = fh.load_tokenizer(str(path))
    model = VisionEncoderDecoderModel.from_pretrained(path, torch_dtype=torch.float32).eval()
    return fh.decode(model, tok, X, field)


def main():
    out = Path(fh.arg("--out", str(ROOT / "data" / "preds_trocr.csv")))
    Xr, yr, spr, _, _ = fh.load("reg")
    Xn, yn, spn, _, _ = fh.load("name")
    names_reg = [str(v) for v in np.load(ROOT / "data" / "name_real.npz")["reg_no"]]
    # pair reg strips with name strips of the same sheet: same reg_no, same occurrence order
    name_of = defaultdict(list)
    for i, r in enumerate(names_reg): name_of[r].append(i)
    seen = defaultdict(int); pairs = []
    for i, r in enumerate(yr):
        k = seen[r]; seen[r] += 1
        if spr[i] in (1, 2) and k < len(name_of[r]): pairs.append((i, name_of[r][k]))
    print(f"{len(pairs)} val/test sheets with both strips")
    reg_pred = predict("reg", Xr[[i for i, _ in pairs]])
    name_pred = predict("name", Xn[[j for _, j in pairs]])
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["split", "reg_true", "name_true", "reg_pred", "name_pred"])
        for (i, j), rp, np_ in zip(pairs, reg_pred, name_pred):
            w.writerow([NAMES[int(spr[i])], yr[i], yn[j], rp, np_])
    print("wrote", out)


if __name__ == "__main__":
    main()
