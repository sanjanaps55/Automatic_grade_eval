"""Held-out register numbers: raw read vs roster rules. Usage: python scripts/eval_roster.py [roster.csv]"""
import csv
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import train_reg as t


def lev(a, b):
    d = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        p, d[0] = d[0], i
        for j, cb in enumerate(b, 1): p, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, p + (ca != cb))
    return d[-1]


def match(read, conf, roster, safe):
    ds = sorted((lev(read, r), r) for r in roster)
    best, second = ds[0], ds[1]
    if best[0] > 2 or second[0] == best[0]: return None                 # nothing close / tie
    if safe and best[0] > 0: return None                                # only exact roster hits auto-accept
    if safe and conf < 0.9 and second[0] <= 1: return None              # a one-digit neighbour exists
    return best[1]


def main():
    ros = [r["reg_no"] for r in csv.DictReader(open(sys.argv[1] if len(sys.argv) > 1 else ROOT / "data" / "roster_test.csv", encoding="utf-8-sig"))]
    net = t.RegNet(); net.load_state_dict(torch.load(ROOT / "models" / "reg.pt", map_location="cpu")); net.eval()
    X, y, val, _, _ = t.load_npz(); vi = np.where(val)[0]
    with torch.no_grad(): res = t.greedy(net(X[vi]))
    truth = [y[i] for i in vi]
    print(f"roster {len(ros)} | held-out sheets {len(truth)}")
    print(f"raw read, no roster: correct {sum(g == tr for (g, _), tr in zip(res, truth))}, wrong {sum(g != tr for (g, _), tr in zip(res, truth))}")
    for name, safe in (("roster nearest-match", False), ("roster safe rules   ", True)):
        ok = wr = rv = 0
        for (g, c), tr in zip(res, truth):
            m = match(g, c, ros, safe)
            if m is None: rv += 1
            elif m == tr: ok += 1
            else: wr += 1
        print(f"{name}: auto-correct {ok}, auto-WRONG {wr}, to review {rv}")


if __name__ == "__main__":
    main()
