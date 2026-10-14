"""Fine-tune TrOCR on header strips (register number or name) with overfitting guards.

Usage (Kaggle/Colab GPU, or CPU for --smoke):
  python -u scripts/finetune_header.py --field name --model microsoft/trocr-small-handwritten --save
  python -u scripts/finetune_header.py --field reg  --model microsoft/trocr-small-handwritten --save

Data split: by student (grade_eval/split.py, same split as every other model), so nobody is in two sets.
  train -> gradient updates | val -> picks the best epoch + early stopping | test -> scored ONCE at the end.
Overfitting guards: strong augmentation, weight decay, low LR + warm-up/decay, frozen encoder for the first
epochs, early stopping on val CER, best-on-val weights restored, train-vs-val gap printed every eval.
"""
import copy
import random
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, VisionEncoderDecoderModel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from grade_eval.split import NAMES, split_of
from grade_eval.validate import repair_reg_no
from train_reg import augment

DEV = "cuda" if torch.cuda.is_available() else "cpu"
random.seed(0); np.random.seed(0); torch.manual_seed(0)


def arg(name, default):
    a = sys.argv
    return type(default)(a[a.index(name) + 1]) if name in a else default


def prep(x):                       # (B,1,H,W) ink-dark strips -> (B,3,384,384) normalised like the TrOCR processor
    x = F.interpolate(x, size=(384, 384), mode="bilinear", align_corners=False)
    return ((x - 0.5) / 0.5).expand(-1, 3, -1, -1)


def load_tokenizer(name):
    """trocr-small uses a sentencepiece (XLM-R) vocab; the fast converter fails on some transformers versions, so try explicit classes."""
    from transformers import XLMRobertaTokenizer, XLMRobertaTokenizerFast
    errs = []
    for make in (lambda: AutoTokenizer.from_pretrained(name, use_fast=False),
                 lambda: XLMRobertaTokenizer.from_pretrained(name),
                 lambda: XLMRobertaTokenizerFast.from_pretrained(name),
                 lambda: AutoTokenizer.from_pretrained(name)):
        try:
            return make()
        except Exception as e:
            errs.append(repr(e)[:150])
    raise RuntimeError("no tokenizer loaded (pip install sentencepiece protobuf, then restart the kernel):\n" + "\n".join(errs))


def to_t(a): return torch.tensor(a.astype(np.float32) / 255).unsqueeze(1)


def load(field):
    """-> X (N,1,H,W), y list[str], split array of 0/1/2 (train/val/test), plus synthetic (SX, SY) for reg."""
    if field == "name":
        r = np.load(ROOT / "data" / "name_real.npz")
        inv = {v: k for k, v in NAMES.items()}
        return to_t(r["X"]), [str(v) for v in r["y"]], np.array([inv[str(s)] for s in r["split"]]), None, None
    r = np.load(ROOT / "data" / "reg_real.npz"); s = np.load(ROOT / "data" / "reg_synth.npz")
    y = [str(v) for v in r["y"]]
    return to_t(r["X"]), y, np.array([split_of(v) for v in y]), to_t(s["X"]), [str(v) for v in s["y"]]


def edit_distance(a, b):
    d = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        prev, d[0] = d[0], i
        for j, cb in enumerate(b, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (ca != cb))
    return d[-1]


def clean(field, t):
    if field == "reg":
        return repair_reg_no(re.sub(r"[^0-9A-Za-z]", "", t).upper())
    return re.sub(r"\s+", " ", t).strip()


@torch.no_grad()
def decode(model, tok, X, field, bs=32):
    model.eval(); out = []
    for i in range(0, len(X), bs):
        g = model.generate(prep(X[i:i + bs].to(DEV)), max_new_tokens=16 if field == "reg" else 40, num_beams=4)
        out += [clean(field, t) for t in tok.batch_decode(g, skip_special_tokens=True)]
    return out


def score(model, tok, X, y, field, tag, verbose=True):
    """-> (CER, exact). CER = char edit distance / reference length; name compared case-insensitively."""
    pred = decode(model, tok, X, field)
    norm = (lambda s: s) if field == "reg" else (lambda s: s.lower())
    cer = sum(edit_distance(norm(p), norm(t)) for p, t in zip(pred, y)) / max(1, sum(len(t) for t in y))
    ex = float(np.mean([norm(p) == norm(t) for p, t in zip(pred, y)]))
    if verbose: print(f"  [{tag}] CER {cer:.3f} | exact {ex:.3f} (n={len(y)})", flush=True)
    return cer, ex, pred


def main():
    field = arg("--field", "name"); smoke = "--smoke" in sys.argv
    name = arg("--model", "microsoft/trocr-small-handwritten"); tag = "small" if "small" in name else "base"
    epochs, bs, lr, patience = arg("--epochs", 30), arg("--bs", 8), arg("--lr", 3e-5), arg("--patience", 5)
    n_syn, freeze_ep = arg("--syn", 3000), arg("--freeze-epochs", 2)
    if field == "name": n_syn = 0                              # no synthetic cursive names
    if smoke: epochs, n_syn, bs = 1, (16 if field == "reg" else 0), 4

    X, y, sp, SX, SY = load(field)
    tr, va, te = (np.where(sp == k)[0] for k in (0, 1, 2))
    assert not set(tr) & set(va) | set(tr) & set(te) | set(va) & set(te)
    print(f"field={field} model={name} device={DEV} | train {len(tr)} val {len(va)} test {len(te)}", flush=True)
    tok = load_tokenizer(name)
    model = VisionEncoderDecoderModel.from_pretrained(name).to(DEV)
    model.config.decoder_start_token_id = tok.cls_token_id; model.config.pad_token_id = tok.pad_token_id
    gc = model.generation_config
    gc.decoder_start_token_id = tok.cls_token_id; gc.pad_token_id = tok.pad_token_id; gc.eos_token_id = tok.sep_token_id

    Xv, yv = X[va], [y[i] for i in va]
    Xt0, yt0 = X[tr[:min(len(tr), 40)]], [y[i] for i in tr[:min(len(tr), 40)]]   # fixed train subset for the gap report
    if not smoke: score(model, tok, Xv, yv, field, "zero-shot val")

    real_rep = 4 if n_syn else 1                              # reg: real strips seen 4x per epoch next to synthetic
    per_epoch = n_syn + len(tr) * real_rep
    steps = epochs * (per_epoch // bs) + 1
    opt = torch.optim.AdamW(model.parameters(), lr, weight_decay=5e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=steps, pct_start=0.1)
    scaler = torch.amp.GradScaler(enabled=DEV == "cuda")
    best, best_state, bad = 9e9, None, 0

    for ep in range(epochs):
        for p in model.encoder.parameters(): p.requires_grad = ep >= freeze_ep      # decoder adapts first
        model.train(); t0 = time.time(); tot = n = 0
        px_pool, py_pool = [X[tr]] * real_rep, [y[i] for i in tr] * real_rep
        if n_syn:
            sel = np.random.choice(len(SY), n_syn, replace=False)
            px_pool.append(SX[sel]); py_pool += [SY[i] for i in sel]
        px_pool = torch.cat(px_pool); perm = np.random.permutation(len(py_pool))
        for i in range(0, len(perm) - bs + 1, bs):
            idx = perm[i:i + bs]
            px = prep(augment(px_pool[idx])).to(DEV)
            lab = tok([py_pool[j] for j in idx], padding="longest", return_tensors="pt").input_ids
            lab[lab == tok.pad_token_id] = -100
            with torch.autocast(DEV, dtype=torch.float16, enabled=DEV == "cuda"):
                loss = model(pixel_values=px, labels=lab.to(DEV)).loss
            opt.zero_grad(); scaler.scale(loss).backward(); scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); scaler.step(opt); scaler.update(); sched.step()
            tot += float(loss.detach()); n += 1
        print(f"ep {ep + 1} loss {tot / max(n, 1):.3f} ({time.time() - t0:.0f}s)", flush=True)

        tr_cer, _, _ = score(model, tok, Xt0, yt0, field, "train subset")
        v_cer, _, _ = score(model, tok, Xv, yv, field, "val")
        print(f"  gap (val CER - train CER) = {v_cer - tr_cer:+.3f}  {'<- overfitting?' if v_cer - tr_cer > 0.15 else ''}", flush=True)
        if v_cer < best - 1e-4:
            best, bad = v_cer, 0; best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
        else:
            bad += 1
            if bad >= patience: print(f"early stop at epoch {ep + 1} (best val CER {best:.3f})"); break

    if best_state: model.load_state_dict(best_state)
    print(f"FINAL (best-on-val weights, val CER {best:.3f})")
    _, _, pv = score(model, tok, Xv, yv, field, "VAL")
    print("val errors:", [(t, p) for p, t in zip(pv, yv) if p.lower() != t.lower()])
    if len(te):
        print("TEST (unseen students, scored once - do not tune on this)")
        score(model, tok, X[te], [y[i] for i in te], field, "TEST")
    if "--save" in sys.argv:
        out = ROOT / "models" / f"trocr_{field}_{tag}"; model.half().save_pretrained(out); tok.save_pretrained(out); print("saved", out)


if __name__ == "__main__":
    main()
