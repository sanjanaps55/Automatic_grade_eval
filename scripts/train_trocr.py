"""Fine-tune TrOCR (handwriting-pretrained) on register-number strips, to compare with the ResNet+BiLSTM+CTC reader.

Usage (Kaggle/Colab GPU): python -u scripts/train_trocr.py --npz [--epochs 6] [--syn 6000] [--save]
Same data and the same 25 held-out students as train_reg.py (data/reg_real.npz, data/reg_synth.npz).
"""
import random
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from grade_eval.validate import repair_reg_no
from train_reg import augment, load_npz, split_of

NAME = "microsoft/trocr-base-handwritten"      # --model microsoft/trocr-small-handwritten for the ~5x smaller one
DEV = "cuda" if torch.cuda.is_available() else "cpu"
random.seed(0); np.random.seed(0); torch.manual_seed(0)


def prep(x):                       # (B,1,48,384) ink-dark strips -> (B,3,384,384) like the TrOCR image processor
    x = F.interpolate(x, size=(384, 384), mode="bilinear", align_corners=False)
    return ((x - 0.5) / 0.5).expand(-1, 3, -1, -1)


def arg(name, default):
    a = sys.argv
    return type(default)(a[a.index(name) + 1]) if name in a else default


def decode(model, tok, X, bs=32):
    out = []
    model.eval()
    for i in range(0, len(X), bs):
        with torch.no_grad():
            g = model.generate(prep(X[i:i + bs].to(DEV)), max_new_tokens=14, num_beams=4,
                               output_scores=True, return_dict_in_generate=True)
        txt = tok.batch_decode(g.sequences, skip_special_tokens=True)
        conf = torch.exp(g.sequences_scores).cpu().tolist()
        out += [(re.sub(r"[^0-9A-Za-z]", "", t).upper(), c) for t, c in zip(txt, conf)]
    return out


def evaluate(model, tok, X, y, tag):
    res = decode(model, tok, X)
    rep = [repair_reg_no(a) for a, _ in res]
    ex = np.mean([a == b for (a, _), b in zip(res, y)]); fx = np.mean([a == b for a, b in zip(rep, y)])
    ok = np.mean([bool(re.fullmatch(r"\d{2}[A-Z]{3}\d{4}", a)) for a in rep])
    ch = np.mean([sum(c1 == c2 for c1, c2 in zip(a, b)) / 9 for a, b in zip(rep, y)])
    print(f"  [{tag}] exact {ex:.3f} | after repair {fx:.3f} | valid format {ok:.3f} | char acc {ch:.3f} (n={len(y)})", flush=True)
    return res, rep


def main():
    smoke = "--smoke" in sys.argv
    name = arg("--model", NAME); tag = "small" if "small" in name else "base"
    epochs, n_syn, bs, lr = arg("--epochs", 6), arg("--syn", 6000), arg("--bs", 16), arg("--lr", 4e-5)
    if smoke: epochs, n_syn, bs = 1, 32, 4
    X, y, val, SX, SY = load_npz()
    sp = np.array([split_of(r) for r in y])
    tr, va, te = np.where(sp == 0)[0], np.where(sp == 1)[0], np.where(sp == 2)[0]
    Xv, yv = X[va], [y[i] for i in va]
    proc = TrOCRProcessor.from_pretrained(name); tok = proc.tokenizer
    model = VisionEncoderDecoderModel.from_pretrained(name).to(DEV)
    model.config.decoder_start_token_id = tok.cls_token_id; model.config.pad_token_id = tok.pad_token_id
    model.generation_config.decoder_start_token_id = tok.cls_token_id; model.generation_config.pad_token_id = tok.pad_token_id
    model.generation_config.eos_token_id = tok.sep_token_id
    print("params", sum(p.numel() for p in model.parameters()) // 10**6, "M | device", DEV, "| real train", len(tr), "val", len(va), "test", len(te), flush=True)
    if not smoke: evaluate(model, tok, Xv, yv, "zero-shot val")

    opt = torch.optim.AdamW(model.parameters(), lr, weight_decay=1e-2)
    per_epoch = n_syn + len(tr) * 4                      # real strips seen 4x per epoch next to fresh synthetic ones
    steps = epochs * (per_epoch // bs) + 1
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=steps, pct_start=0.1)
    scaler = torch.amp.GradScaler(enabled=DEV == "cuda")
    for ep in range(epochs):
        model.train(); t0 = time.time()
        sel = np.random.choice(len(SY), n_syn, replace=False)
        pool_x = torch.cat([SX[sel]] + [X[tr]] * 4); pool_y = [SY[i] for i in sel] + [y[i] for i in tr] * 4
        perm = np.random.permutation(len(pool_y)); tot = n = 0
        for i in range(0, len(perm) - bs + 1, bs):
            idx = perm[i:i + bs]
            px = prep(augment(pool_x[idx])).to(DEV)
            lab = tok([pool_y[j] for j in idx], padding="longest", return_tensors="pt").input_ids
            lab[lab == tok.pad_token_id] = -100
            with torch.autocast(DEV, dtype=torch.float16, enabled=DEV == "cuda"):
                loss = model(pixel_values=px, labels=lab.to(DEV)).loss
            opt.zero_grad(); scaler.scale(loss).backward(); scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); scaler.step(opt); scaler.update(); sched.step()
            tot += float(loss.detach()); n += 1
        print(f"ep {ep + 1} loss {tot / max(n, 1):.3f} ({time.time() - t0:.0f}s)", flush=True)
        if smoke or ep % 2 == 1 or ep == epochs - 1: evaluate(model, tok, Xv, yv, "real val")
    print("FINAL")
    res, rep = evaluate(model, tok, Xv, yv, "VAL")
    print("val errors:", [(t, r, round(c, 2)) for (g, c), r, t in zip(res, rep, yv) if r != t])
    print("TEST (unseen, scored once)"); evaluate(model, tok, X[te], [y[i] for i in te], "TEST")
    if "--save" in sys.argv:
        out = ROOT / "models" / (f"trocr_reg_{tag}" if tag == "small" else "trocr_reg"); model.half().save_pretrained(out); tok.save_pretrained(out); print("saved", out)


if __name__ == "__main__":
    main()
