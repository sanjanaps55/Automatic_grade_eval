"""Score the saved TrOCR reader (models/trocr_reg) on val/test strips and on full photos; time per strip."""
import csv
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import VisionEncoderDecoderModel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from grade_eval.validate import repair_reg_no
from grade_eval.vision import header, layout
from grade_eval.vision.preprocess import load_image
from train_reg import load_npz, split_of
from train_trocr import prep

torch.set_num_threads(4)
path = ROOT / "models" / "trocr_reg"
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(path)
model = VisionEncoderDecoderModel.from_pretrained(path, dtype=torch.float32).eval()


def read(x):                      # (B,1,48,384) -> [(text, conf)]
    with torch.no_grad():
        g = model.generate(prep(x), max_new_tokens=14, num_beams=4, output_scores=True, return_dict_in_generate=True)
    t = tok.batch_decode(g.sequences, skip_special_tokens=True)
    return [(re.sub(r"[^0-9A-Za-z]", "", a).upper(), float(np.exp(c))) for a, c in zip(t, g.sequences_scores)]


def report(tag, res, y):
    rep = [repair_reg_no(a) for a, _ in res]
    ex = np.mean([a == b for (a, _), b in zip(res, y)]); fm = np.mean([bool(re.fullmatch(r"\d{2}[A-Z]{3}\d{4}", a)) for a in rep])
    ch = np.mean([sum(c1 == c2 for c1, c2 in zip(a, b)) / 9 for a, b in zip(rep, y)])
    print(f"  [{tag}] exact {ex:.3f} | valid format {fm:.3f} | char acc {ch:.3f} (n={len(y)})", flush=True)
    print("   errors (truth, read, conf):", [(t, r, round(c, 2)) for (g, c), r, t in zip(res, rep, y) if r != t], flush=True)


X, y, _, _, _ = load_npz(); sp = np.array([split_of(r) for r in y])
for k, name in ((1, "VAL"), (2, "TEST")):
    i = np.where(sp == k)[0]; res = []
    t0 = time.time()
    for j in range(0, len(i), 4): res += read(X[i[j:j + 4]])
    report(name, res, [y[j] for j in i])
    print(f"   speed: {(time.time() - t0) / len(i):.2f} s per strip on CPU", flush=True)
rows = list(csv.DictReader(open(ROOT / "data" / "labels_all.csv", encoding="utf-8-sig")))
R = {1: [0, 0], 2: [0, 0]}
for r in rows:
    k = split_of(r["reg_no"])
    if k == 0: continue
    R[k][1] += 1; a = layout.align(load_image(ROOT / "data" / "samples" / r["file"]))
    if a is None: continue
    R[k][0] += read(torch.tensor(header.ink_strip(a.image))[None, None])[0][0] == r["reg_no"]
print("FULL PHOTOS (unaligned counted wrong) val:", R[1], "test:", R[2], flush=True)
