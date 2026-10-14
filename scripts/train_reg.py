"""Train the register-number reader: ResNet18 (ImageNet) + CTC, synthetic pretrain -> fine-tune on real strips.

Usage: python scripts/train_reg.py [--syn-epochs N] [--ft-epochs N]
Validation = the same held-out students as the marks model. Weights -> models/reg.pt.
"""
import csv
import random
import re
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grade_eval.split import split_of
from grade_eval.validate import repair_reg_no
from grade_eval.vision import header

VOCAB = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"   # CTC index 0 = blank
random.seed(0); np.random.seed(0); torch.manual_seed(0)
DEV = "cuda" if torch.cuda.is_available() else "cpu"


class RegNet(nn.Module):
    def __init__(self):
        super().__init__()
        try:
            r = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
        except Exception as e:
            print("no pretrained weights:", e); r = torchvision.models.resnet18(weights=None)
        r.layer3[0].conv1.stride = (2, 1); r.layer3[0].downsample[0].stride = (2, 1)
        self.body = nn.Sequential(r.conv1, r.bn1, r.relu, r.maxpool, r.layer1, r.layer2, r.layer3)
        self.rnn = nn.LSTM(256, 128, num_layers=2, bidirectional=True, batch_first=True, dropout=0.1)
        self.head = nn.Linear(256, len(VOCAB) + 1)

    def forward(self, x):                          # (B,1,48,384), ink dark
        f = self.body(((x - 0.5) / 0.5).expand(-1, 3, -1, -1))   # (B,256,3,48)
        seq, _ = self.rnn(f.mean(2).permute(0, 2, 1))             # (B,48,256)
        return self.head(seq)                                     # (B,48,C)


def greedy(logits):
    p = F.softmax(logits, -1); conf, idx = p.max(-1); out = []
    for b in range(idx.shape[0]):
        s, cs, prev = "", [], 0
        for t in range(idx.shape[1]):
            k = int(idx[b, t])
            if k != prev and k != 0: s += VOCAB[k - 1]; cs.append(float(conf[b, t]))
            prev = k
        out.append((s, min(cs) if cs else 0.0))
    return out


def augment(x):
    B = x.shape[0]
    ang = (torch.rand(B) - 0.5) * 0.12; sc = 1 + (torch.rand(B) - 0.5) * 0.2
    theta = torch.zeros(B, 2, 3)
    theta[:, 0, 0] = sc * torch.cos(ang); theta[:, 0, 1] = -sc * torch.sin(ang); theta[:, 0, 2] = (torch.rand(B) - 0.5) * 0.1
    theta[:, 1, 0] = sc * torch.sin(ang); theta[:, 1, 1] = sc * torch.cos(ang); theta[:, 1, 2] = (torch.rand(B) - 0.5) * 0.15
    theta = theta.to(x.device)
    x = 1 - F.grid_sample(1 - x, F.affine_grid(theta, x.shape, align_corners=False), align_corners=False)
    return x.clamp(0, 1)


def is_val(reg): return split_of(reg) == 1


def load_npz():
    """Pre-rendered data from scripts/export_reg_data.py (used on Colab, which lacks the handwriting fonts)."""
    r = np.load(ROOT / "data" / "reg_real.npz"); s = np.load(ROOT / "data" / "reg_synth.npz")
    f = lambda a: torch.tensor(a.astype(np.float32) / 255).unsqueeze(1)
    return f(r["X"]), [str(v) for v in r["y"]], r["val"], f(s["X"]), [str(v) for v in s["y"]]


def load_real():
    X, y, v = [], [], []
    for r in csv.DictReader(open(ROOT / "data" / "labels_all.csv", encoding="utf-8-sig")):
        p = ROOT / "data" / "aligned" / (Path(r["file"]).stem + ".png")
        if p.exists():
            X.append(header.ink_strip(cv2.imread(str(p)))); y.append(r["reg_no"]); v.append(is_val(r["reg_no"]))
    return torch.tensor(np.stack(X)).unsqueeze(1), y, np.array(v)


def synth_batch(n):
    s = [header.synth_strip() for _ in range(n)]
    return torch.tensor(np.stack([a for a, _ in s])).unsqueeze(1), [t for _, t in s]


def step(net, opt, sched, xb, yb, ctc):
    xb = xb.to(DEV)
    out = net(augment(xb)).log_softmax(-1).permute(1, 0, 2)
    tg = [[VOCAB.index(c) + 1 for c in t] for t in yb]
    loss = ctc(out, torch.tensor(sum(tg, [])), torch.full((len(yb),), out.shape[0], dtype=torch.long),
               torch.tensor([len(t) for t in tg], dtype=torch.long))
    opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), 5); opt.step()
    if sched: sched.step()
    return float(loss.detach())


def evaluate(net, X, y, tag):
    net.eval()
    with torch.no_grad(): res = greedy(net(X.to(DEV)).cpu())
    raw = np.mean([a == b for (a, _), b in zip(res, y)])
    rep = [repair_reg_no(a) for a, _ in res]
    fixed = np.mean([a == b for a, b in zip(rep, y)])
    valid = np.mean([bool(re.fullmatch(r"\d{2}[A-Z]{3}\d{4}", a)) for a in rep])
    chars = np.mean([sum(c1 == c2 for c1, c2 in zip(a, b)) / 9 for a, b in zip(rep, y)])
    print(f"  [{tag}] exact {raw:.3f} | after repair {fixed:.3f} | valid format {valid:.3f} | char acc {chars:.3f} (n={len(y)})", flush=True)
    return res, rep


def main():
    a = sys.argv
    syn_ep = int(a[a.index("--syn-epochs") + 1]) if "--syn-epochs" in a else 10
    ft_ep = int(a[a.index("--ft-epochs") + 1]) if "--ft-epochs" in a else 25
    use_npz = "--npz" in a
    if use_npz: X, y, val, SX, SY = load_npz()
    else: X, y, val = load_real()
    sp = np.array([split_of(r) for r in y])
    tr, va, te = np.where(sp == 0)[0], np.where(sp == 1)[0], np.where(sp == 2)[0]
    print(f"real: train {len(tr)} | val {len(va)} | test {len(te)} (test is scored once, at the very end)")
    net = RegNet().to(DEV); ctc = nn.CTCLoss(blank=0, zero_infinity=True)
    Xv, yv = X[va], [y[i] for i in va]

    def syn_pool(n):                       # fresh synthetic strips: sampled from the npz, or rendered on the fly
        if not use_npz: return synth_batch(n)
        sel = np.random.choice(len(SY), n, replace=False)
        return SX[sel], [SY[i] for i in sel]

    opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-4)
    per = 6400 if use_npz else 1600; bs = 64 if use_npz else 32
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=syn_ep * (per // bs) + 2)
    for ep in range(syn_ep):                                        # stage 1: synthetic only
        net.train(); t0 = time.time(); xs, ys = syn_pool(per); tot = 0
        perm = np.random.permutation(per)
        for i in range(0, per - bs + 1, bs):
            idx = perm[i:i + bs]; tot += step(net, opt, sched, xs[idx], [ys[j] for j in idx], ctc)
        print(f"syn ep {ep + 1} loss {tot / (per // bs):.3f} ({time.time() - t0:.0f}s)", flush=True)
        if ep % 3 == 2:
            evaluate(net, Xv, yv, "real val"); evaluate(net, SX[:200], SY[:200], "synthetic")

    opt = torch.optim.AdamW(net.parameters(), 3e-4, weight_decay=1e-4)
    for ep in range(ft_ep):                                         # stage 2: real train + fresh synthetic
        net.train(); xs, ys = syn_pool(len(tr))
        pool_x = torch.cat([X[tr], xs]); pool_y = [y[i] for i in tr] + list(ys)
        perm = np.random.permutation(len(pool_y)); tot = 0
        for i in range(0, len(perm) - bs + 1, bs):
            idx = perm[i:i + bs]; tot += step(net, opt, None, pool_x[idx], [pool_y[j] for j in idx], ctc)
        if ep % 5 == 4: print(f"ft ep {ep + 1} loss {tot / (len(perm) // bs):.3f}", flush=True); evaluate(net, Xv, yv, "real val")
    (ROOT / "models").mkdir(exist_ok=True); torch.save(net.state_dict(), ROOT / "models" / "reg.pt")
    print("FINAL"); evaluate(net, X[tr], [y[i] for i in tr], "train"); res, rep = evaluate(net, Xv, yv, "VAL")
    print("val errors:", [(t, r, round(c, 2)) for (g, c), r, t in zip(res, rep, yv) if r != t])
    print("TEST (unseen, scored once)"); evaluate(net, X[te], [y[i] for i in te], "TEST")


if __name__ == "__main__":
    main()
