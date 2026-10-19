"""Marks reader specialised for HARD rows (a|b entries, written sums, grader tick strokes crossing the column line).

Same architecture and data as train_marks_sum.py (ResNet18 + CTC over "0123456789.+"), but trained for small data:
  * labelled hard rows (data/label_work/records.txt + hard_notation.txt) are oversampled, blanks are subsampled
  * synthetic hard rows built ONLY from digits cut out of the TRAIN sheets (no val/test leakage): a|b pairs placed in
    the a and b columns, written sums, long tick strokes across the dashed column line, scribbled-out first entries
  * optional --coord: two extra input channels tell the net where column a / column b / the spill-over are
  * feature dropout, weight decay, strong augmentation, early stopping on VAL (best checkpoint kept), patience
  * train-vs-val gap printed at every evaluation; the TEST sheets are scored once, at the end, with the best checkpoint
Split = data/split.csv (by student). Usage:
  python scripts/train_marks_hard.py --epochs 80 [--coord] [--init marks_sum|marks|imagenet] [--seed 0] [--out models/marks_hard.pt]
  [--limit N]  (CPU smoke test)
"""
import json
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
import train_marks_sum as tms
from train_marks import augment

DEV = "cuda" if torch.cuda.is_available() else "cpu"
arg = tms.arg
H, W = 64, 192
# strip x positions (strip = template x 206..450 scaled to 192 px): column a | column b | spill into the next group
BOUND_AB, BOUND_B = int((290 - 206) * W / 244), int((373 - 206) * W / 244)


# ------------------------------------------------------------------ model wrapper (dropout, optional coordinate channels)
def coord_channels(B):
    xs = torch.arange(W, dtype=torch.float32)
    ramp = (xs / (W - 1) * 2 - 1).view(1, 1, 1, W).expand(B, 1, H, W)
    bound = torch.zeros(1, 1, 1, W)
    for b in (BOUND_AB, BOUND_B):
        bound += torch.exp(-((xs - b) ** 2) / (2 * 3.0 ** 2)).view(1, 1, 1, W)
    return ramp, bound.clamp(0, 1).expand(B, 1, H, W) * 2 - 1


def forward(net, x, coord, drop=0.0):
    x = (x - 0.5) / 0.5
    if coord:
        ramp, bound = coord_channels(x.shape[0])
        x = torch.cat([x, ramp.to(x.device), bound.to(x.device)], 1)
    else:
        x = x.expand(-1, 3, -1, -1)
    f = net.body(x).mean(2).permute(0, 2, 1)
    f = F.dropout(f, drop, training=net.training)
    return net.head(f)


def make_model(init):
    net = tms.make_net(ROOT / "models" / "marks.pt" if init == "marks" else None)
    if init == "marks_sum":
        net.load_state_dict(torch.load(ROOT / "models" / "marks_sum.pt", map_location="cpu"))
    elif init == "imagenet":
        net = tms.MarksNet(pretrained=True); net.head = nn.Linear(512, len(tms.VOCAB) + 1)
    return net


# ------------------------------------------------------------------ realistic synthetic hard rows
def paste(canvas, g, x, top, h):
    w = max(4, int(g.shape[1] * h / g.shape[0] * tms.rng.uniform(0.85, 1.15)))
    if x + w > W - 2:
        return None
    g = cv2.resize(g, (w, h), interpolation=cv2.INTER_AREA)
    canvas[top:top + h, x:x + w] = np.minimum(canvas[top:top + h, x:x + w], g)
    return x + w + int(tms.rng.integers(-1, 4))


def number(canvas, bank, s, x, top, h):
    for c in s:
        x = paste(canvas, bank[c][int(tms.rng.integers(len(bank[c])))], x, top, h)
        if x is None:
            return None
    return x


def draw_tick(canvas):
    """the grader's check-mark swoosh: a long stroke rising to the right, usually crossing the column line"""
    r = tms.rng
    x0, y0 = int(r.integers(8, 70)), int(r.integers(40, 62))
    x1, y1 = min(W - 3, x0 + int(r.integers(45, 120))), max(2, y0 - int(r.integers(15, 55)))
    mx, my = (x0 + x1) // 2 + int(r.integers(-8, 9)), (y0 + y1) // 2 + int(r.integers(-6, 7))
    pts = np.array([[x0, y0], [mx, my], [x1, y1]], np.int32)
    cv2.polylines(canvas, [pts], False, 0.0, int(r.integers(1, 3)))
    if r.random() < 0.35:                                        # short down-stroke of a tick
        cv2.line(canvas, (x0, y0), (x0 - int(r.integers(4, 12)), y0 - int(r.integers(6, 16))), 0.0, 1)


def draw_scribble(canvas, x0, x1):
    r = tms.rng
    for _ in range(int(r.integers(5, 11))):
        cv2.line(canvas, (int(r.integers(x0, x1)), int(r.integers(6, 58))), (int(r.integers(x0, x1)), int(r.integers(6, 58))), 0.0, int(r.integers(2, 4)))
    cv2.ellipse(canvas, ((x0 + x1) // 2, 32), (int((x1 - x0) / 2), int(r.integers(10, 22))), 0, 0, 360, 0.0, 2)


def synth_hard(bank, n):
    r = tms.rng
    xs, ys = [], []
    tries = 0
    while len(xs) < n and tries < n * 4:
        tries += 1
        kind = r.choice(["single_tick", "ab", "ab_tick", "single_scrib"], p=[0.3, 0.35, 0.2, 0.15])
        digit = lambda: str(r.integers(0, 10)) if r.random() < 0.8 else "1" + str(r.integers(0, 3))
        a, b = digit(), digit()
        if any(not bank[c] for t in (a, b) for c in t):
            continue
        h = int(r.integers(28, 46)); top = int(r.integers(6, max(7, 64 - h - 4)))
        canvas = np.ones((H, W), np.float32)
        if kind == "single_tick":
            x = number(canvas, bank, a, int(r.integers(6, 36)), top, h); lab = a
            draw_tick(canvas)
        elif kind == "single_scrib":
            x = number(canvas, bank, b, int(r.integers(BOUND_AB + 6, BOUND_AB + 30)), top, h); lab = b
            draw_scribble(canvas, 4, BOUND_AB - 4)
            if r.random() < 0.5:
                draw_tick(canvas)
        else:
            x = number(canvas, bank, a, int(r.integers(6, 36)), top, h)
            if x is not None:
                x = number(canvas, bank, b, int(r.integers(BOUND_AB + 8, BOUND_AB + 34)), int(np.clip(top + r.integers(-5, 6), 2, 64 - h - 2)), h)
            lab = f"{a}+{b}"
            if kind == "ab_tick":
                draw_tick(canvas)
        if x is None:
            continue
        if r.random() < 0.5:                                      # speckle / stray marks
            for _ in range(int(r.integers(1, 6))):
                cv2.circle(canvas, (int(r.integers(0, W)), int(r.integers(0, H))), 1, 0.0, -1)
        xs.append(canvas); ys.append(lab)
    return torch.tensor(np.stack(xs)).unsqueeze(1), ys


def _glyph(x):
    """largest ink component (+ small pieces inside its box) of a single-digit strip, or None"""
    ink = (x < 0.5).astype(np.uint8)
    n, cc, st, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    if n < 2:
        return None
    big = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    bx, by, bw, bh = st[big, :4]
    keep = np.zeros_like(ink)
    for k in range(1, n):
        kx, ky, kw, kh, ka = st[k]
        if k == big or (ka >= 6 and kx >= bx - 4 and kx + kw <= bx + bw + 4 and ky >= by - 6 and ky + kh <= by + bh + 6):
            keep[cc == k] = 1
    ys, xs = np.where(keep)
    if len(xs) < 40:
        return None
    g = np.where(keep[ys.min():ys.max() + 1, xs.min():xs.max() + 1] > 0, x[ys.min():ys.max() + 1, xs.min():xs.max() + 1], 1.0).astype(np.float32)
    h, w = g.shape
    return g if h >= 16 and 0.1 <= w / h <= 1.5 else None


def build_bank(tr, min_per_digit=15):
    """Clean single-digit glyphs from TRAIN rows only. First pass: rows with no stray ink in column b; digits with too few
    glyphs are topped up from the other single-digit rows (largest component only, so ticks and neighbours are dropped)."""
    bank = {str(k): [] for k in range(10)}
    for first in (True, False):
        for x, lab, sp, sheet, q, simple in tr:
            if len(lab) != 1 or not lab.isdigit() or (first and not simple) or (not first and len(bank[lab]) >= min_per_digit):
                continue
            g = _glyph(x)
            if g is not None:
                bank[lab].append(g)
    return bank


# ------------------------------------------------------------------ evaluation helpers
def value(s):
    return tms.parse_value(s) if s else 0.0


def is_ok(s, y):
    v = value(s)
    return v is not None and abs(v - value(y)) < 1e-6


def predict(net, X, coord):
    net.eval(); out = []
    with torch.no_grad():
        for i in range(0, len(X), 64):
            out += tms.decode(forward(net, X[i:i + 64].to(DEV), coord).cpu())
    return out


def report(name, rows, preds):
    y = [r[1] for r in rows]; simple = [r[5] for r in rows]
    res = {}
    for lab, sel in (("all", lambda i: True), ("simple", lambda i: simple[i]), ("hard", lambda i: not simple[i]), ("sum/ab notation", lambda i: "+" in y[i])):
        idx = [i for i in range(len(rows)) if sel(i)]
        ok = sum(is_ok(preds[i][0], y[i]) for i in idx)
        res[lab] = (ok, len(idx))
    line = "  ".join(f"{k}: {a}/{b}" + (f" ({100 * a / b:.0f}%)" if b else "") for k, (a, b) in res.items())
    print(f"  {name:6s} {line}", flush=True)
    return res


def main():
    epochs = arg("--epochs", 80, int); seed = arg("--seed", 0, int); init = arg("--init", "marks_sum")
    coord = "--coord" in sys.argv; patience = arg("--patience", 15, int); hard_rep = arg("--hard-rep", 3, int)
    n_syn_hard = arg("--syn-hard", 1500, int); n_syn_sum = arg("--syn-sum", 500, int)
    out_path = Path(arg("--out", ROOT / "models" / "marks_hard.pt"))
    tms.rng = np.random.default_rng(seed); torch.manual_seed(seed); np.random.seed(seed)

    rows = tms.build_dataset()
    if "--limit" in sys.argv:
        n = arg("--limit", 64, int)
        rows = [r for r in rows if r[2] == "train"][:n] + [r for r in rows if r[2] == "val"][:n // 2] + [r for r in rows if r[2] == "test"][:n // 2]
    tr = [r for r in rows if r[2] == "train"]; va = [r for r in rows if r[2] == "val"]; te = [r for r in rows if r[2] == "test"]
    hard = [r for r in tr if not r[5] and r[1]]; simple_filled = [r for r in tr if r[5] and r[1]]; blanks = [r for r in tr if not r[1]]
    keep = tms.rng.choice(len(blanks), min(len(blanks), int(0.4 * (len(simple_filled) + len(hard)))), replace=False) if blanks else []
    base = simple_filled + hard * hard_rep + [blanks[i] for i in keep]
    print(f"rows: train {len(tr)} (hard labelled {len(hard)} x{hard_rep}, simple filled {len(simple_filled)}, blanks kept {len(keep)}), val {len(va)}, test {len(te)}"
          f" | coord={coord} init={init} seed={seed}", flush=True)

    bank = build_bank(tr)                                                     # glyphs from TRAIN single-digit rows only
    print("glyphs:", {k: len(v) for k, v in bank.items()}, flush=True)

    net = make_model(init).to(DEV)
    opt = torch.optim.AdamW(net.parameters(), 5e-4 if init != "imagenet" else 1e-3, weight_decay=1e-3)
    spe = (len(base) + n_syn_hard + n_syn_sum + 31) // 32
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 1e-3 if init != "imagenet" else 2e-3, total_steps=epochs * spe + 10)
    ctc = nn.CTCLoss(blank=0, zero_infinity=True)
    enc = lambda s: [tms.VOCAB.index(c) + 1 for c in s]
    TX = torch.tensor(np.stack([r[0] for r in base])).unsqueeze(1); TY = [r[1] for r in base]
    VX = torch.tensor(np.stack([r[0] for r in va])).unsqueeze(1); VY = [r[1] for r in va]
    EX = torch.tensor(np.stack([r[0] for r in te])).unsqueeze(1)
    real_tr = [r for r in tr if r[1]][:400]                                  # fixed slice of real train rows for the train-vs-val gap
    RX = torch.tensor(np.stack([r[0] for r in real_tr])).unsqueeze(1)

    best, best_ep, best_state, bad = -1.0, 0, None, 0
    for ep in range(epochs):
        net.train(); t0 = time.time(); tot = 0.0
        SX, SY = synth_hard(bank, n_syn_hard); S2X, S2Y = tms.synth(bank, n_syn_sum)
        px = torch.cat([TX, SX, S2X]); py = TY + SY + S2Y
        perm = tms.rng.permutation(len(py))
        for i in range(0, len(perm), 32):
            idx = perm[i:i + 32]
            xb = augment(px[idx]).to(DEV)
            out = forward(net, xb, coord, drop=0.2).log_softmax(-1).permute(1, 0, 2)
            tg = [enc(py[j]) for j in idx]
            loss = ctc(out, torch.tensor(sum(tg, [])), torch.full((len(idx),), out.shape[0], dtype=torch.long), torch.tensor([len(t) for t in tg], dtype=torch.long))
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), 5.0); opt.step(); sched.step(); tot += float(loss.detach())
        pv = predict(net, VX, coord)
        ok_all = np.mean([is_ok(p[0], y) for p, y in zip(pv, VY)])
        hv = [i for i, r in enumerate(va) if not r[5]]
        ok_hard = np.mean([is_ok(pv[i][0], VY[i]) for i in hv]) if hv else 0.0
        score = ok_all + ok_hard                                            # model selection on VAL only
        if ep % 5 == 4 or ep == epochs - 1 or score > best:
            ptr = predict(net, RX, coord); gap = np.mean([is_ok(p[0], r[1]) for p, r in zip(ptr, real_tr)])
            print(f"ep {ep + 1:3d} loss {tot / (len(perm) / 32):.3f} | val all {ok_all:.3f} hard {ok_hard:.3f} | real-train {gap:.3f} (gap {gap - ok_all:+.3f}) | {time.time() - t0:.0f}s/ep" + ("  *best*" if score > best else ""), flush=True)
        if score > best:
            best, best_ep, bad = score, ep + 1, 0
            best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                print(f"early stop at epoch {ep + 1} (best epoch {best_ep}, val score {best:.3f})", flush=True); break

    net.load_state_dict(best_state)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state": best_state, "coord": coord}, out_path)
    print(f"\nbest checkpoint: epoch {best_ep}; saved {out_path}")
    print("VALIDATION (used for selection)"); res_v = report("val", va, predict(net, VX, coord))
    print("TEST (scored once)"); pt = predict(net, EX, coord); res_t = report("test", te, pt)
    for th in (0.9, 0.97):
        kept = [(p, r) for p, r in zip(pt, te) if p[1] >= th or p[0] == ""]
        print(f"  test conf>={th}: auto {len(kept)}/{len(te)}, wrong among auto {sum(not is_ok(p[0], r[1]) for p, r in kept)}")
    Path(str(out_path) + ".json").write_text(json.dumps({"best_epoch": best_ep, "coord": coord, "init": init, "seed": seed, "val": res_v, "test": res_t}))


if __name__ == "__main__":
    main()
