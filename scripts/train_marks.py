"""Fine-tune a pretrained ResNet18 + CTC head to read the marks in a row strip.

Usage: python scripts/train_marks.py [--epochs N]
Trains on rows whose mark sits in column a only ("simple"); a|b / written sums are routed to review for now.
Validation = whole students held out in data/cells.npz. Weights -> models/marks.pt.
"""
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

ROOT = Path(__file__).resolve().parents[1]
VOCAB = "0123456789."             # CTC index 0 = blank
torch.manual_seed(0)
np.random.seed(0)


class MarksNet(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        try:
            w = torchvision.models.ResNet18_Weights.DEFAULT if pretrained else None
            r = torchvision.models.resnet18(weights=w)
        except Exception as e:  # offline: fall back to random init
            print("pretrained weights unavailable, training from scratch:", e)
            r = torchvision.models.resnet18(weights=None)
        # stride the height fast, keep width resolution for CTC
        r.layer3[0].conv1.stride = (2, 1); r.layer3[0].downsample[0].stride = (2, 1)
        r.layer4[0].conv1.stride = (2, 1); r.layer4[0].downsample[0].stride = (2, 1)
        self.body = nn.Sequential(r.conv1, r.bn1, r.relu, r.maxpool, r.layer1, r.layer2, r.layer3, r.layer4)
        self.head = nn.Linear(512, len(VOCAB) + 1)

    def forward(self, x):                     # x: (B,1,64,192) in [0,1], ink dark
        x = (x - 0.5) / 0.5
        f = self.body(x.expand(-1, 3, -1, -1))   # (B,512,2,24)
        f = f.mean(2).permute(0, 2, 1)           # (B,T,512)
        return self.head(f)                      # (B,T,C)


def decode(logits):
    """Greedy CTC decode -> (string, confidence = mean max-prob of the emitted frames)."""
    p = F.softmax(logits, -1)
    conf, idx = p.max(-1)
    out = []
    for b in range(idx.shape[0]):
        s, cs, prev = "", [], 0
        for t in range(idx.shape[1]):
            k = int(idx[b, t])
            if k != prev and k != 0:
                s += VOCAB[k - 1]; cs.append(float(conf[b, t]))
            prev = k
        out.append((s, min(cs) if cs else float(conf[b].min())))
    return out


def augment(x):
    """x: (B,1,H,W) tensor; random shift/scale/rotate + thickness change."""
    B = x.shape[0]
    ang = (torch.rand(B) - 0.5) * 0.15
    sc = 1 + (torch.rand(B) - 0.5) * 0.25
    tx = (torch.rand(B) - 0.5) * 0.12
    ty = (torch.rand(B) - 0.5) * 0.2
    theta = torch.zeros(B, 2, 3)
    theta[:, 0, 0] = sc * torch.cos(ang); theta[:, 0, 1] = -sc * torch.sin(ang); theta[:, 0, 2] = tx
    theta[:, 1, 0] = sc * torch.sin(ang); theta[:, 1, 1] = sc * torch.cos(ang); theta[:, 1, 2] = ty
    g = F.affine_grid(theta, x.shape, align_corners=False)
    x = 1 - F.grid_sample(1 - x, g, align_corners=False)
    k = np.random.choice([0, 1, 2])
    if k == 1:   x = 1 - F.max_pool2d(1 - x, 3, 1, 1)      # thicker
    elif k == 2: x = 1 - F.avg_pool2d(1 - x, 3, 1, 1) * 1.6  # thinner/fainter
    return x.clamp(0, 1)


def glyph_bank(X, y, idx):
    """Ink bounding-box crops of single-digit training strips, keyed by digit."""
    bank = {str(k): [] for k in range(10)}
    for i in idx:
        if len(y[i]) == 1 and y[i].isdigit():
            a = X[i, 0].numpy()
            ys, xs = np.where(a < 0.5)
            if len(xs) < 40: continue
            g = a[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
            if g.shape[0] >= 14 and g.shape[1] >= 4: bank[y[i]].append(g)
    return bank


def synth(bank, n):
    """n synthetic strips of 2-digit numbers (10-19 mostly, any pair sometimes) pasted from real single digits."""
    import cv2
    xs, ys = [], []
    for _ in range(n):
        if np.random.rand() < 0.7:
            lab = "1" + str(np.random.randint(0, 10))
        else:
            lab = str(np.random.randint(1, 10)) + str(np.random.randint(0, 10))
        if any(not bank[c] for c in lab): continue
        h = np.random.randint(28, 52); canvas = np.ones((64, 192), np.float32)
        x = np.random.randint(8, 60); top = np.random.randint(4, 64 - h - 2) if 64 - h - 2 > 4 else 4
        for c in lab:
            g = bank[c][np.random.randint(len(bank[c]))]
            w = max(4, int(g.shape[1] * h / g.shape[0] * np.random.uniform(0.85, 1.15)))
            g = cv2.resize(g, (w, h), interpolation=cv2.INTER_AREA)
            jy = np.random.randint(-3, 4); t = min(max(0, top + jy), 64 - h)
            if x + w > 190: break
            canvas[t:t + h, x:x + w] = np.minimum(canvas[t:t + h, x:x + w], g)
            x += w + np.random.randint(-3, 9)
        else:
            xs.append(canvas); ys.append(lab)
    return torch.tensor(np.stack(xs)).unsqueeze(1), ys


def main():
    epochs = int(sys.argv[sys.argv.index("--epochs") + 1]) if "--epochs" in sys.argv else 25
    d = np.load(ROOT / "data" / "cells.npz")
    X = torch.tensor(d["X"], dtype=torch.float32).unsqueeze(1) / 255
    y = [str(v) for v in d["y"]]
    simple, val = d["simple"].astype(bool), d["val"].astype(bool)
    # blank strips with stray strokes are fine; rows with a|b or sums are excluded (no per-cell labels)
    use = simple | np.array([v == "" for v in y])
    tr = np.where(use & ~val)[0]; va = np.where(use & val)[0]
    filled = np.array([y[i] != "" for i in tr])
    # keep all filled rows, subsample blanks to 40% of the training set
    blanks = tr[~filled]; keep = np.random.choice(blanks, min(len(blanks), int(0.67 * filled.sum())), replace=False)
    tr = np.concatenate([tr[filled], keep])
    print(f"train {len(tr)} ({int(filled.sum())} filled) | val {len(va)} ({sum(1 for i in va if y[i])} filled)")

    bank = glyph_bank(X, y, tr)
    print('glyphs:', {k: len(v) for k, v in bank.items()})
    net = MarksNet()
    opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=epochs * ((len(tr) + 300 + 31) // 32) + 5)
    ctc = nn.CTCLoss(blank=0, zero_infinity=True)
    enc = lambda s: [VOCAB.index(c) + 1 for c in s]

    def evaluate():
        net.eval(); res = []
        with torch.no_grad():
            for i in range(0, len(va), 64):
                idx = va[i:i + 64]
                res += [(y[j], *r) for j, r in zip(idx, decode(net(X[idx])))]
        return res

    for ep in range(epochs):
        net.train(); perm = np.random.permutation(tr); t0 = time.time(); tot = 0
        SX, SY = synth(bank, 300)          # fresh synthetic 2-digit strips every epoch
        pool_x = torch.cat([X[perm], SX]); pool_y = [y[j] for j in perm] + SY
        perm = np.random.permutation(len(pool_y))
        for i in range(0, len(perm), 32):
            idx = perm[i:i + 32]
            xb = augment(pool_x[idx]); out = net(xb).log_softmax(-1).permute(1, 0, 2)
            tg = [enc(pool_y[j]) for j in idx]
            loss = ctc(out, torch.tensor(sum(tg, [])), torch.full((len(idx),), out.shape[0], dtype=torch.long),
                       torch.tensor([len(t) for t in tg], dtype=torch.long))
            opt.zero_grad(); loss.backward(); opt.step(); sched.step(); tot += float(loss.detach())
        if ep % 5 == 4 or ep == epochs - 1:
            r = evaluate(); acc = np.mean([a == b for a, b, _ in r])
            print(f"ep {ep + 1} loss {tot / (len(perm) / 32):.3f} val exact {acc:.3f} ({time.time() - t0:.0f}s/ep)", flush=True)
    (ROOT / "models").mkdir(exist_ok=True)
    torch.save(net.state_dict(), ROOT / "models" / "marks.pt")

    r = evaluate()
    print("\nVALIDATION (held-out students, simple rows)")
    filled = [(a, b, c) for a, b, c in r if a]
    print(f"  filled rows exact: {np.mean([a == b for a, b, _ in filled]):.3f} ({len(filled)})")
    blank = [(a, b, c) for a, b, c in r if not a]
    print(f"  blank rows predicted blank: {np.mean([b == '' for _, b, _ in blank]):.3f} ({len(blank)})")
    for th in (0.5, 0.7, 0.9):
        acc_rows = [(a, b, c) for a, b, c in r if c >= th or b == ""]
        conf_wrong = sum(1 for a, b, c in acc_rows if a != b)
        print(f"  conf>={th}: kept {len(acc_rows)}/{len(r)}, confident-wrong {conf_wrong}")
    print("  errors:", [(a, b, round(c, 2)) for a, b, c in r if a != b][:25])


if __name__ == "__main__":
    main()
