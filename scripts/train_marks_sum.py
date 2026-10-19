"""Marks reader that also reads written sums / a|b entries: CTC over "0123456789.+" (a|b is treated as a+b).

Usage: python scripts/train_marks_sum.py [--epochs 40] [--mnist 400] [--out models/marks_sum.pt] [--limit N]
  --mnist N adds N MNIST glyphs per digit to the synthetic-strip glyph bank (downloads MNIST); --limit trains on N rows (smoke test).
Data: data/cells.npz (row strips) + data/labels_all.csv (row totals) + data/label_work/records.txt (original notation).
A non-simple row is used for training only when its notation is fully readable (no '?') and its terms add up to the
labelled row total, so label noise is filtered. Split by student from data/split.csv. Synthetic strips (real single
digits pasted together with a drawn '+') add variety. Prints value-level accuracy and confident-wrong counts on val.
"""
import csv
import re
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from train_marks import MarksNet, augment

VOCAB = "0123456789.+"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
rng = np.random.default_rng(0); torch.manual_seed(0)


def parse_value(s: str):
    """'3+5+1' -> 9.0 ; None if not a clean sum of numbers."""
    if not s or not re.fullmatch(r"\d{1,2}(\.\d)?(\+\d{1,2}(\.\d)?)*", s):
        return None
    return sum(float(t) for t in s.split("+"))


def decode(logits):
    p = F.softmax(logits, -1); conf, idx = p.max(-1); out = []
    for b in range(idx.shape[0]):
        s, cs, prev = "", [], 0
        for t in range(idx.shape[1]):
            k = int(idx[b, t])
            if k != prev and k != 0:
                s += VOCAB[k - 1]; cs.append(float(conf[b, t]))
            prev = k
        out.append((s, min(cs) if cs else float(conf[b].min())))
    return out


def make_net(pretrained_from: Path | None = None):
    net = MarksNet(pretrained=False)
    if pretrained_from and pretrained_from.exists():          # warm start from the single-number reader
        net.load_state_dict(torch.load(pretrained_from, map_location="cpu"))
    net.head = nn.Linear(512, len(VOCAB) + 1)
    return net


def load_records():
    """{(sheet_file, q): normalised notation} for cells with fully readable notation."""
    out = {}
    for line in open(ROOT / "data" / "label_work" / "records.txt", encoding="utf-8"):
        if line.startswith("#") or "|" not in line:
            continue
        parts = [p.strip() for p in line.split("|")]
        # reg/name fields contain no '|', but the marks field uses '|' for a|b -> re-split carefully
        m = re.match(r"^(\d{3})\s*\|\s*([^|]*?)\s*\|\s*([^|]*?)\s*\|\s*(.*?)\s*\|\s*([^|]*?)\s*\|\s*(FLAG.*|[^|]*)$", line.strip())
        if not m:
            continue
        sheet = f"sheet_{m.group(1)}.jpeg"
        for q, cell in enumerate(m.group(4).split(","), 1):
            cell = cell.strip()
            if not cell or "?" in cell or cell == "-":
                continue
            norm = cell.replace("|", "+")
            if re.fullmatch(r"\d{1,2}(\.\d)?(\+\d{1,2}(\.\d)?)*", norm):
                out[(sheet, q)] = norm
    extra = ROOT / "data" / "label_work" / "hard_notation.txt"      # notation for hard rows that had only a total (see PLAN.md)
    if extra.exists():
        for line in open(extra, encoding="utf-8"):
            if line.startswith("#") or "|" not in line:
                continue
            key, norm = line.strip().split("|", 1)
            sh, q = key.split("-")
            norm = norm.replace("|", "+")
            if re.fullmatch(r"\d{1,2}(\.\d)?(\+\d{1,2}(\.\d)?)*", norm):
                out.setdefault((f"sheet_{sh}.jpeg", int(q)), norm)
    return out


def draw_plus(h):
    s = int(h * rng.uniform(0.45, 0.75)); c = np.ones((h, s + 2), np.float32); m = (s + 2) // 2; t = int(rng.integers(1, 3))
    cv2.line(c, (1, h // 2 + int(rng.integers(-2, 3))), (s, h // 2 + int(rng.integers(-2, 3))), 0, t)
    cv2.line(c, (m, h // 2 - s // 2), (m + int(rng.integers(-1, 2)), h // 2 + s // 2), 0, t)
    return c


def synth(bank, n):
    xs, ys = [], []
    for _ in range(n):
        k = int(rng.integers(1, 5)) if rng.random() < 0.8 else 1
        terms = [str(rng.integers(0, 10)) if rng.random() < 0.75 else "1" + str(rng.integers(0, 10)) for _ in range(k)]
        lab = "+".join(terms)
        if any(not bank[c] for t in terms for c in t):
            continue
        h = int(rng.integers(26, 44)); canvas = np.ones((64, 192), np.float32); x = int(rng.integers(4, 30)); top = int(rng.integers(8, 64 - h - 4))
        ok = True
        for ti, t in enumerate(terms):
            if ti:
                g = draw_plus(h)
                if x + g.shape[1] > 188:
                    ok = False; break
                canvas[top:top + h, x:x + g.shape[1]] = np.minimum(canvas[top:top + h, x:x + g.shape[1]], g); x += g.shape[1] + int(rng.integers(0, 4))
            for c in t:
                g = bank[c][int(rng.integers(len(bank[c])))]
                w = max(4, int(g.shape[1] * h / g.shape[0] * rng.uniform(0.85, 1.15)))
                g = cv2.resize(g, (w, h), interpolation=cv2.INTER_AREA)
                if x + w > 188:
                    ok = False; break
                canvas[top:top + h, x:x + w] = np.minimum(canvas[top:top + h, x:x + w], g); x += w + int(rng.integers(-2, 6))
            if not ok:
                break
        if ok:
            xs.append(canvas); ys.append(lab)
    return torch.tensor(np.stack(xs)).unsqueeze(1), ys


def build_dataset():
    d = np.load(ROOT / "data" / "cells.npz")
    X = d["X"].astype(np.float32) / 255
    split = {r["file"]: r["split"] for r in csv.DictReader(open(ROOT / "data" / "split.csv", encoding="utf-8-sig"))}
    rec = load_records()
    rows = []
    for i in range(len(d["y"])):
        if int(d["row"][i]) > 9:
            continue                                       # skip the sub-total row
        sheet, q, total, simple = str(d["sheet"][i]), int(d["row"][i]) + 1, str(d["y"][i]), bool(d["simple"][i])
        sp = split.get(sheet, "train")
        lab = None
        if (sheet, q) in rec:
            n = rec[(sheet, q)]
            v = parse_value(n)
            if v is not None and (total == "" or abs(v - float(total)) < 1e-6):
                lab = n
        if lab is None and simple:
            lab = total                                    # single number or blank
        if lab is None:
            continue
        rows.append((X[i], lab, sp, sheet, q, simple))
    return rows


def arg(name, default, cast=str):
    return cast(sys.argv[sys.argv.index(name) + 1]) if name in sys.argv else default


def mnist_glyphs(bank, per_digit):
    """Add clean handwritten MNIST digits (cropped, dark ink on white) as extra glyphs for the synthetic strips."""
    from torchvision.datasets import MNIST
    d = MNIST(str(ROOT / "data" / "mnist"), train=True, download=True)
    n = {k: 0 for k in bank}
    for img, lab in zip(d.data.numpy(), d.targets.numpy()):
        k = str(int(lab))
        if n[k] >= per_digit:
            continue
        g = 1.0 - img.astype(np.float32) / 255
        ys, xs = np.where(g < 0.5)
        if len(xs) < 20:
            continue
        g = g[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
        if rng.random() < 0.5:                                  # pen strokes are usually thinner/ragged: erode/dilate a little
            g = cv2.erode(g, np.ones((2, 2), np.uint8)) if rng.random() < 0.5 else cv2.dilate(g, np.ones((2, 2), np.uint8))
        bank[k].append(g); n[k] += 1
    return bank


def main():
    epochs = arg("--epochs", 40, int)
    out_path = Path(arg("--out", ROOT / "models" / "marks_sum.pt"))
    rows = build_dataset()
    if "--limit" in sys.argv:
        n = arg("--limit", 64, int)     # smoke test: a few train rows + a few held-out rows
        rows = [r for r in rows if r[2] == "train"][:n] + [r for r in rows if r[2] != "train"][:max(8, n // 2)]
    tr = [r for r in rows if r[2] == "train"]
    filled = [r for r in tr if r[1]]; blanks = [r for r in tr if not r[1]]
    keep = rng.choice(len(blanks), min(len(blanks), int(0.5 * len(filled))), replace=False)
    tr = filled + [blanks[i] for i in keep]       # blanks are the majority; keep a third of the training set
    va = [r for r in rows if r[2] != "train"]
    n_sum = sum("+" in r[1] for r in rows)
    print(f"rows {len(rows)} (train {len(tr)}, val+test {len(va)}), with sums {n_sum}, blanks {sum(r[1] == '' for r in rows)} (train kept {len(tr)})", flush=True)

    # glyph bank from clean single-digit training rows
    bank = {str(k): [] for k in range(10)}
    for x, lab, *_ in tr:
        if len(lab) == 1 and lab.isdigit():
            ys, xs = np.where(x < 0.5)
            if len(xs) >= 40:
                g = x[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
                if g.shape[0] >= 14 and g.shape[1] >= 4:
                    bank[lab].append(g)
    print("glyphs:", {k: len(v) for k, v in bank.items()}, flush=True)
    if "--mnist" in sys.argv:
        bank = mnist_glyphs(bank, arg("--mnist", 400, int))
        print("glyphs with MNIST:", {k: len(v) for k, v in bank.items()}, flush=True)

    net = make_net(ROOT / "models" / "marks.pt").to(DEV)
    opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-4)
    steps = epochs * ((len(tr) + 500 + 31) // 32) + 10
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=steps)
    ctc = nn.CTCLoss(blank=0, zero_infinity=True)
    enc = lambda s: [VOCAB.index(c) + 1 for c in s]
    TX = torch.tensor(np.stack([r[0] for r in tr])).unsqueeze(1); TY = [r[1] for r in tr]
    VX = torch.tensor(np.stack([r[0] for r in va])).unsqueeze(1); VY = [r[1] for r in va]

    def evaluate():
        net.eval(); out = []
        with torch.no_grad():
            for i in range(0, len(VX), 64):
                out += decode(net(VX[i:i + 64].to(DEV)).cpu())
        return out

    for ep in range(epochs):
        net.train(); t0 = time.time(); tot = 0
        SX, SY = synth(bank, 500)
        px = torch.cat([TX, SX]); py = TY + SY
        perm = rng.permutation(len(py))
        for i in range(0, len(perm), 32):
            idx = perm[i:i + 32]
            xb = augment(px[idx]).to(DEV); out = net(xb).log_softmax(-1).permute(1, 0, 2)
            tg = [enc(py[j]) for j in idx]
            loss = ctc(out, torch.tensor(sum(tg, [])), torch.full((len(idx),), out.shape[0], dtype=torch.long),
                       torch.tensor([len(t) for t in tg], dtype=torch.long))
            opt.zero_grad(); loss.backward(); opt.step(); sched.step(); tot += float(loss.detach())
        if ep % 5 == 4 or ep == epochs - 1:
            r = evaluate()
            val = [parse_value(s) if s else 0.0 for s, _ in r]; tru = [parse_value(y) if y else 0.0 for y in VY]
            acc = np.mean([a is not None and abs(a - b) < 1e-6 for a, b in zip(val, tru)])
            print(f"ep {ep + 1} loss {tot / (len(perm) / 32):.3f} val value-exact {acc:.3f} ({time.time() - t0:.0f}s/ep)", flush=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({k: v.cpu() for k, v in net.state_dict().items()}, out_path)
    r = evaluate()
    print("\nVALIDATION+TEST (held-out students)")
    for th in (0.0, 0.7, 0.8, 0.9, 0.95):
        kept = wrong = 0
        for (s, c), y in zip(r, VY):
            v = parse_value(s) if s else 0.0
            t = parse_value(y) if y else 0.0
            if c >= th or s == "":
                kept += 1; wrong += (v is None or abs(v - t) > 1e-6)
        print(f"  conf>={th}: auto {kept}/{len(r)}, wrong among auto {wrong}")
    sums = [(s, c, y) for (s, c), y in zip(r, VY) if "+" in y]
    ok = sum(1 for s, c, y in sums if s == y)
    print(f"  sum rows: exact string {ok}/{len(sums)}; value ok {sum(1 for s, c, y in sums if (parse_value(s) or -1) == parse_value(y))}")


if __name__ == "__main__":
    main()
