"""Train the two total readers (Sub-Total cell, Grand Total box bands) and measure them honestly.

Usage: python scripts/train_total.py --mode eval     train on the train split only -> models/total_sub_eval.pt, total_grand_eval.pt,
                                             writes data/overnight/total_report_eval.json (OOF on train + held-out read-outs)
       python scripts/train_total.py --mode final    train on every trusted sheet with the same recipe -> models/total_sub.pt, total_grand.pt

Anti-overfit: warm start from marks_sum.pt, heavy augmentation (affine, thickness, random circles), synthetic totals pasted
from real train digits, early stopping on an inner validation split, 4-fold out-of-fold read-outs on the TRAIN sheets
(used to pick the confidence threshold), the held-out sheets are only read once at the end.
"""
import json
import math
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
from grade_eval import totals
from train_marks import augment
from train_marks_sum import VOCAB, decode, make_net, parse_value

DEV = "cuda" if torch.cuda.is_available() else "cpu"
rng = np.random.default_rng(0); torch.manual_seed(0)
EPOCHS, K = 45, totals.N_BANDS
N_SYN = int(__import__('os').environ.get('N_SYN', 1200))


def arg(n, d):
    return type(d)(sys.argv[sys.argv.index(n) + 1]) if n in sys.argv else d


def enc(s):
    return [VOCAB.index(c) + 1 for c in s]


def add_circle(a: np.ndarray, p: float):
    """a: (H,W) float in [0,1], ink dark. Draw a pen ellipse around the middle with probability p."""
    if rng.random() > p:
        return a
    a = a.copy(); H, W = a.shape
    c = (int(W * rng.uniform(0.35, 0.65)), int(H * rng.uniform(0.4, 0.6)))
    ax = (int(W * rng.uniform(0.18, 0.4)), int(H * rng.uniform(0.38, 0.56)))
    cv2.ellipse(a, c, ax, float(rng.uniform(-15, 15)), float(rng.uniform(0, 40)), float(rng.uniform(300, 400)), 0.0, int(rng.integers(2, 4)))
    return a


def glyph_bank(S, tsub):
    bank = {str(k): [] for k in range(10)}
    d = np.load(ROOT / "data" / "cells.npz")
    X = d["X"].astype(np.float32) / 255
    import csv
    split = {r["file"]: r["split"] for r in csv.DictReader(open(ROOT / "data" / "split.csv", encoding="utf-8-sig"))}
    for i in range(len(d["y"])):
        y = str(d["y"][i])
        if len(y) == 1 and y.isdigit() and int(d["row"][i]) <= 9 and d["simple"][i] and split.get(str(d["sheet"][i]), "train") == "train":
            a = X[i]; ys, xs = np.where(a < 0.5)
            if len(xs) >= 40:
                g = a[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
                if g.shape[0] >= 14 and g.shape[1] >= 4:
                    bank[y].append(g)
    return bank


def synth_number(bank, wide: bool):
    """A pasted 1-2 digit total (sometimes .5) on a blank strip, circled half of the time."""
    v = int(rng.integers(1, 48))
    if rng.random() < 0.25:
        d = int(rng.integers(1, 5)); v = d * 11                  # 11, 22, 33, 44: repeated digits are what CTC tends to collapse
    lab = str(v)
    if rng.random() < 0.1:
        lab += ".5"
    H, W = (64, 320) if wide else (64, 192)
    for _ in range(3):
        canvas = np.ones((H, W), np.float32)
        h = int(rng.integers(30, 52)); x = int(rng.integers(10, W // 3 if wide else 50)) if not wide else int(rng.integers(20, 180))
        ok = True
        for ch in lab:
            if ch == ".":
                canvas[H // 2 + h // 3:H // 2 + h // 3 + 5, x:x + 5] = 0; x += 9; continue
            if not bank[ch]:
                ok = False; break
            g = bank[ch][int(rng.integers(len(bank[ch])))]
            w = max(4, int(g.shape[1] * h / g.shape[0] * rng.uniform(0.85, 1.15)))
            g = cv2.resize(g, (w, h), interpolation=cv2.INTER_AREA)
            top = (H - h) // 2 + int(rng.integers(-3, 4))
            if x + w >= W - 2:
                ok = False; break
            canvas[top:top + h, x:x + w] = np.minimum(canvas[top:top + h, x:x + w], g); x += w + int(rng.integers(-2, 7))
        if ok:
            return add_circle(canvas, 0.55), lab
    return None


def clutter(a: np.ndarray, n: int):
    """Wavy pen strokes like cursive words / ticks / struck lines, drawn into a strip (ink dark)."""
    a = a.copy(); H, W = a.shape
    for _ in range(n):
        x = int(rng.integers(0, W)); y = int(rng.integers(H // 4, H - 8)); pts = []
        for _k in range(int(rng.integers(4, 14))):
            pts.append((x, y)); x += int(rng.integers(6, 22)); y = int(np.clip(y + rng.integers(-9, 10), 4, H - 4))
        cv2.polylines(a, [np.array(pts, np.int32)], False, 0.0, int(rng.integers(1, 3)))
    return a


def ctc_none(lp, tg, il, tl):
    return F.ctc_loss(lp, tg, il, tl, blank=0, reduction="none", zero_infinity=True)


def items_for(kind, D, idx):
    """-> list of (strips array (n,64,W), target str, is_band_set)."""
    out = []
    for i in idx:
        loc = str(D["loc"][i])
        if not int(D["trust"][i]):
            continue
        if kind == "sub":
            if loc in ("S", "B"):
                out.append((D["S"][i][None], str(D["tsub"][i]), False))
            elif loc in ("G", "N"):
                out.append((D["S"][i][None], "", False))
        else:
            if int(D["grand_skip"][i]):
                continue
            if loc in ("G", "B"):
                out.append((D["G"][i], str(D["tgr"][i]), True))
            elif loc in ("S", "N"):
                out.append((D["G"][i], "", True))
    return out


def to_t(a):
    return torch.tensor(a.astype(np.float32) / 255)


def train_one(kind, items, bank, epochs, val_items=None, seed=0, log=print):
    torch.manual_seed(seed)
    net = make_net(); net.load_state_dict(torch.load(ROOT / "models" / "marks_sum.pt", map_location="cpu")); net = net.to(DEV)
    opt = torch.optim.AdamW(net.parameters(), 5e-4, weight_decay=1e-3)
    bs = 16
    steps = epochs * (math.ceil((len(items) + N_SYN) / bs)) + 5
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 1e-3, total_steps=steps)
    wide = kind == "grand"
    best, best_state, hist = (-1, None), None, []
    for ep in range(epochs):
        net.train(); tot = 0; n = 0
        pool = list(items)
        for _ in range(N_SYN):                                  # fresh synthetic examples every epoch
            r = synth_number(bank, wide)
            if r is None:
                continue
            a, lab = r
            if wide:                                             # one band with the number, the others blank
                blank = np.ones((K, 64, 320), np.float32); j = int(rng.integers(K)); blank[j] = a
                for k2 in range(K):
                    if k2 != j and rng.random() < 0.6:
                        blank[k2] = clutter(blank[k2], int(rng.integers(1, 3)))      # other bands hold words, not the number
                    elif k2 == j and rng.random() < 0.4:
                        blank[k2] = clutter(blank[k2], 1)
                pool.append(((blank * 255).astype(np.uint8), lab, True))
            else:
                if rng.random() < 0.3:
                    a = clutter(a, 1)                                               # grader tick / underline across the cell
                pool.append(((a[None] * 255).astype(np.uint8), lab, False))
        perm = rng.permutation(len(pool))
        for b in range(0, len(perm), bs):
            batch = [pool[j] for j in perm[b:b + bs]]
            xs, loss_terms = [], []
            for strips, lab, is_band in batch:
                arr = strips.astype(np.float32) / 255
                arr = np.stack([add_circle(a, 0.3 if lab else 0.08) for a in arr])
                xs.append(arr)
            x = torch.tensor(np.concatenate(xs)).unsqueeze(1)
            x = augment(x).to(DEV)
            out = net(x).log_softmax(-1).permute(1, 0, 2)                # (T, N, C)
            T = out.shape[0]
            counts = [len(a) for a in xs]
            labs = []
            for (strips, lab, is_band), c in zip(batch, counts):
                labs += [lab] * c
            N = out.shape[1]
            il = torch.full((N,), T, dtype=torch.long)
            tgT = [enc(l) for l in labs]
            LT = ctc_none(out, torch.tensor(sum(tgT, []), dtype=torch.long), il, torch.tensor([len(t) for t in tgT], dtype=torch.long))
            LE = ctc_none(out, torch.tensor([], dtype=torch.long), il, torch.zeros(N, dtype=torch.long))
            lt, le = LT.view(-1), LE.view(-1)
            pos = 0; total = 0.0
            for (strips, lab, is_band), c in zip(batch, counts):
                sl = slice(pos, pos + c); pos += c
                if is_band and lab:
                    total = total + le[sl].sum() + (lt[sl] - le[sl]).min()      # the number sits in whichever band reads it best
                elif is_band:
                    total = total + le[sl].sum()
                else:
                    total = total + lt[sl].sum()
            loss = total / len(batch)
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), 5.0); opt.step(); sched.step()
            tot += float(loss.detach()); n += 1
        if val_items is not None and (ep % 5 == 4 or ep == epochs - 1):
            acc = eval_items(kind, net, val_items)
            hist.append((ep + 1, tot / n, acc))
            log(f"   [{kind}] ep {ep + 1} loss {tot / n:.2f} inner-val sheet-ok {acc:.3f}")
            if acc >= best[0]:
                best = (acc, ep + 1); best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
    if best_state is not None:
        net.load_state_dict(best_state)
    return net, best, hist


@torch.no_grad()
def read_strips(net, strips):
    net.eval()
    return net(to_t(strips).unsqueeze(1).to(DEV)).cpu().numpy()


def sheet_totals(kind, net, strips):
    lg = read_strips(net, strips)
    return [{"src": f"{kind}{k}", "cands": totals.value_candidates(lg[k])} for k in range(len(lg))]


def eval_items(kind, net, items):
    """Fraction of items whose pooled read-out equals the target (blank target: no number read)."""
    ok = 0
    for strips, lab, is_band in items:
        pooled = totals.pool_totals(sheet_totals(kind, net, strips))
        top = max(pooled.items(), key=lambda kv: kv[1]) if pooled else None
        want = parse_value(lab) if lab else None
        got = top[0] if top and top[1] >= 0.5 else None
        ok += (want is None and got is None) or (want is not None and got is not None and abs(want - got) < 1e-6)
    return ok / max(1, len(items))


def main():
    mode = arg("--mode", "eval")
    D = dict(np.load(ROOT / "data" / "overnight" / "total_data.npz"))
    splits = D["split"]
    train_idx = np.where(splits == "train")[0] if mode == "eval" else np.arange(len(splits))
    held_idx = np.where(splits != "train")[0]
    bank = glyph_bank(None, None)
    print("glyph bank", {k: len(v) for k, v in bank.items()}, flush=True)
    report = {"mode": mode, "epochs": EPOCHS}
    models = {}
    for kind in ("sub", "grand"):
        t0 = time.time()
        # inner validation: 20% of the training sheets, never used to fit during early-stopping selection
        if mode == "final":                                    # settings fixed by the eval-mode run (inner validation), no selection on held-out data
            best_ep, best, oof, held, LG = {"sub": 45, "grand": 25}[kind], (None, None), {}, {}, {}
            net, _, _ = train_one(kind, items_for(kind, D, train_idx), bank, best_ep, None, seed=99)
            models[kind] = net
            torch.save({k: v.cpu() for k, v in net.state_dict().items()}, ROOT / "models" / f"total_{kind}.pt")
            print(f"[{kind}] FINAL saved models/total_{kind}.pt trained on {len(items_for(kind, D, train_idx))} items, {best_ep} epochs [{time.time() - t0:.0f}s]", flush=True)
            continue
        perm = rng.permutation(train_idx); nv = max(6, len(perm) // 5)
        inner_val, inner_tr = perm[:nv], perm[nv:]
        net, best, hist = train_one(kind, items_for(kind, D, inner_tr), bank, EPOCHS, items_for(kind, D, inner_val))
        best_ep = best[1] or EPOCHS
        print(f"[{kind}] inner-val best epoch {best_ep} (acc {best[0]:.3f})  [{time.time() - t0:.0f}s]", flush=True)
        # 4-fold out-of-fold read-outs on the training sheets: used to choose the confidence threshold, never the held-out sheets
        folds = np.array_split(rng.permutation(train_idx), 4); oof = {}; LG = {}
        for f in range(4):
            tr = np.concatenate([folds[g] for g in range(4) if g != f])
            m, _, _ = train_one(kind, items_for(kind, D, tr), bank, best_ep, None, seed=f + 1)
            for i in folds[f]:
                strips = D["S"][i][None] if kind == "sub" else D["G"][i]
                oof[int(i)] = totals.pool_totals(sheet_totals(kind, m, strips))
                LG[f"{kind}_oof_{int(i)}"] = read_strips(m, strips)
        # the model used downstream: trained on all training sheets with the epoch count found above
        net, _, _ = train_one(kind, items_for(kind, D, train_idx), bank, best_ep, None, seed=99)
        models[kind] = net
        held = {}
        for i in held_idx if mode == "eval" else []:
            strips = D["S"][i][None] if kind == "sub" else D["G"][i]
            held[int(i)] = totals.pool_totals(sheet_totals(kind, net, strips))
            LG[f"{kind}_held_{int(i)}"] = read_strips(net, strips)
        report[kind] = {"best_epoch": best_ep, "inner_val_acc": best[0], "oof": {str(k): {str(v): p for v, p in d.items()} for k, d in oof.items()},
                        "held": {str(k): {str(v): p for v, p in d.items()} for k, d in held.items()}}
        np.savez_compressed(ROOT / "data" / "overnight" / f"total_logits_{kind}_{mode}.npz", **LG)
        suffix = "_eval" if mode == "eval" else ""
        tag_ = __import__("os").environ.get("TOTAL_TAG", "")
        suffix = suffix + tag_
        torch.save({k: v.cpu() for k, v in net.state_dict().items()}, ROOT / "models" / f"total_{kind}{suffix}.pt")
        print(f"[{kind}] saved models/total_{kind}{suffix}.pt  [{time.time() - t0:.0f}s]", flush=True)
    if mode == "final":
        return
    report["meta"] = {"sheet": [str(s) for s in D["sheet"]], "loc": [str(s) for s in D["loc"]], "split": [str(s) for s in D["split"]],
                      "total": [float(t) for t in D["total"]], "trust": [int(t) for t in D["trust"]]}
    (ROOT / "data" / "overnight" / f"total_report_{mode}.json").write_text(json.dumps(report), encoding="utf-8")


if __name__ == "__main__":
    main()
