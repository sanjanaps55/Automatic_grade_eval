"""Score a saved TrOCR header model on CPU: val + test CER/exact and seconds per strip.

Usage: python scripts/eval_header.py --field reg|name [--model models/trocr_<field>_small]
"""
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import VisionEncoderDecoderModel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import finetune_header as fh


def main():
    field = fh.arg("--field", "reg"); path = fh.arg("--model", str(ROOT / "models" / f"trocr_{field}_small"))
    fh.DEV = "cpu"
    tok = fh.load_tokenizer(path)
    model = VisionEncoderDecoderModel.from_pretrained(path, torch_dtype=torch.float32).eval()
    X, y, sp, _, _ = fh.load(field)
    for k, tag in ((1, "VAL"), (2, "TEST")):
        idx = np.where(sp == k)[0]; t0 = time.time()
        cer, ex, pred = fh.score(model, tok, X[idx], [y[i] for i in idx], field, tag)
        print(f"    {(time.time() - t0) / len(idx):.2f} s/strip on CPU")
        print("    wrong:", [(y[i], p) for i, p in zip(idx, pred) if p.lower() != y[i].lower()])


if __name__ == "__main__":
    main()
