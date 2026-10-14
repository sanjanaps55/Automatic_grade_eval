"""Pre-render synthetic register-number strips (needs Windows handwriting fonts) so training can run on Colab.

Usage: python scripts/export_reg_data.py [N]   -> data/reg_synth.npz (X uint8 (N,48,384), y), data/reg_real.npz
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from grade_eval.vision import header
from train_reg import load_real


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 30000
    xs, ys = zip(*(header.synth_strip() for _ in range(n)))
    np.savez_compressed(ROOT / "data" / "reg_synth.npz", X=(np.stack(xs) * 255).astype(np.uint8), y=np.array(ys))
    X, y, val = load_real()
    np.savez_compressed(ROOT / "data" / "reg_real.npz", X=(X[:, 0].numpy() * 255).astype(np.uint8), y=np.array(y), val=val)
    print("synthetic", n, "real", len(y))


if __name__ == "__main__":
    main()
