"""How useful is the weak name reader as a roster *hint*? For val/test sheets: predict the name, find the nearest roster
name, and check it is the right student (by reg no). Also reports how often the margin over 2nd best is enough to trust.

Usage: python scripts/eval_name_hint.py [roster.csv]
"""
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import VisionEncoderDecoderModel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import finetune_header as fh
from grade_eval.roster import load_roster, name_similarity


def main():
    roster = load_roster(sys.argv[1] if len(sys.argv) > 1 else ROOT / "data" / "roster_test.csv")
    d = np.load(ROOT / "data" / "name_real.npz")
    X = fh.to_t(d["X"]); split = [str(s) for s in d["split"]]; reg = [str(r) for r in d["reg_no"]]; names = [str(y) for y in d["y"]]
    idx = [i for i, s in enumerate(split) if s in ("val", "test")]
    fh.DEV = "cuda" if torch.cuda.is_available() else "cpu"
    path = str(ROOT / "models" / "trocr_name_small")
    tok = fh.load_tokenizer(path)
    model = VisionEncoderDecoderModel.from_pretrained(path, torch_dtype=torch.float32).to(fh.DEV).eval()
    pred = fh.decode(model, tok, X[idx], "name")
    ok = trusted = trusted_ok = top3 = 0
    for p, i in zip(pred, idx):
        sc = sorted(((name_similarity(p, r.name), r.reg_no) for r in roster), reverse=True)
        best, second = sc[0], sc[1]
        hit = best[1] == reg[i]; ok += hit; top3 += reg[i] in [s[1] for s in sc[:3]]
        if best[0] >= 0.6 and best[0] - second[0] >= 0.1:
            trusted += 1; trusted_ok += hit
    n = len(idx)
    print(f"sheets {n} | nearest roster name correct {ok}/{n} | in top-3 {top3}/{n} | "
          f"confident (sim>=0.6, margin>=0.1): {trusted}, of which correct {trusted_ok}")


if __name__ == "__main__":
    main()
