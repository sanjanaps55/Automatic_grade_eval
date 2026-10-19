"""Marks reader built on TrOCR-small-handwritten (pretrained on handwriting) instead of ResNet18-CTC.

Same data, split and metric as train_marks_sum.py so the two are directly comparable: row strip -> "3+4+8" / "12" / "".
Usage (GPU, Kaggle): python scripts/train_marks_trocr.py --epochs 25 [--model microsoft/trocr-small-handwritten]
                     [--out models/marks_trocr] [--limit N]   (--limit = CPU smoke test)
Prints value-exact on the held-out students (single-number rows and sum rows separately) and confident-wrong counts.
"""
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import train_marks_sum as tms
from train_marks import augment

DEV = "cuda" if torch.cuda.is_available() else "cpu"
arg = tms.arg
rng = np.random.default_rng(0); torch.manual_seed(0)


def load_tokenizer(name):
    """trocr-small uses a sentencepiece (XLM-R) vocab; the default fast converter fails on newer transformers, so try explicit classes."""
    from transformers import AutoTokenizer, XLMRobertaTokenizer, XLMRobertaTokenizerFast
    errs = []
    for make in (lambda: AutoTokenizer.from_pretrained(name, use_fast=False),
                 lambda: XLMRobertaTokenizer.from_pretrained(name),
                 lambda: XLMRobertaTokenizerFast.from_pretrained(name),
                 lambda: AutoTokenizer.from_pretrained(name)):
        try:
            return make()
        except Exception as e:
            errs.append(repr(e)[:150])
    raise RuntimeError("no tokenizer loaded (pip install sentencepiece protobuf, then restart the kernel): " + " | ".join(errs))


def to_pixels(proc, X):
    """(B,1,64,192) float in [0,1], white paper / dark ink -> TrOCR pixel_values (strip padded to a squarer canvas)."""
    imgs = []
    for x in X.squeeze(1).numpy():
        a = (x * 255).astype(np.uint8)
        canvas = np.full((96, 192), 255, np.uint8); canvas[16:80] = a            # headroom so the squash to 384x384 is milder
        imgs.append(Image.fromarray(canvas).convert("RGB"))
    return proc(images=imgs, return_tensors="pt").pixel_values


def main():
    from transformers import AutoImageProcessor, VisionEncoderDecoderModel
    epochs = arg("--epochs", 25, int)
    name = arg("--model", "microsoft/trocr-small-handwritten")
    out_dir = Path(arg("--out", ROOT / "models" / "marks_trocr"))
    rows = tms.build_dataset()
    if "--limit" in sys.argv:
        n = arg("--limit", 64, int)     # smoke test: a few train rows + a few held-out rows
        rows = [r for r in rows if r[2] == "train"][:n] + [r for r in rows if r[2] != "train"][:max(8, n // 2)]
    tr = [r for r in rows if r[2] == "train"]
    filled = [r for r in tr if r[1]]; blanks = [r for r in tr if not r[1]]
    keep = rng.choice(len(blanks), min(len(blanks), int(0.5 * len(filled))), replace=False) if blanks else []
    tr = filled + [blanks[i] for i in keep]
    va = [r for r in rows if r[2] != "train"]
    print(f"rows {len(rows)} (train {len(tr)}, val+test {len(va)})", flush=True)

    bank = {str(k): [] for k in range(10)}
    for x, lab, *_ in tr:
        if len(lab) == 1 and lab.isdigit():
            ys, xs = np.where(x < 0.5)
            if len(xs) >= 40:
                g = x[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
                if g.shape[0] >= 14 and g.shape[1] >= 4:
                    bank[lab].append(g)
    if "--mnist" in sys.argv:
        bank = tms.mnist_glyphs(bank, arg("--mnist", 400, int))

    tok = load_tokenizer(name)
    proc = AutoImageProcessor.from_pretrained(name)
    model = VisionEncoderDecoderModel.from_pretrained(name).to(DEV)
    model.config.decoder_start_token_id = tok.cls_token_id or tok.bos_token_id
    model.config.pad_token_id = tok.pad_token_id
    model.config.eos_token_id = tok.sep_token_id or tok.eos_token_id
    model.generation_config.max_new_tokens = 24
    model.generation_config.num_beams = 1

    TX = torch.tensor(np.stack([r[0] for r in tr])).unsqueeze(1); TY = [r[1] for r in tr]
    VX = torch.tensor(np.stack([r[0] for r in va])).unsqueeze(1); VY = [r[1] for r in va]
    opt = torch.optim.AdamW(model.parameters(), 4e-5, weight_decay=1e-4)
    bs = 16

    def labels_of(texts):
        ids = tok(texts, padding=True, return_tensors="pt").input_ids
        ids[ids == tok.pad_token_id] = -100
        return ids

    @torch.no_grad()
    def predict(X):
        model.eval(); out = []
        for i in range(0, len(X), 32):
            pv = to_pixels(proc, X[i:i + 32]).to(DEV)
            g = model.generate(pv, return_dict_in_generate=True, output_scores=True)
            sc = model.compute_transition_scores(g.sequences, g.scores, normalize_logits=True).cpu()
            for seq, s in zip(g.sequences.cpu(), sc):
                txt = tok.decode(seq, skip_special_tokens=True).replace(" ", "")
                n = max(1, int((seq[1:] != tok.pad_token_id).sum()))
                out.append((txt, float(torch.exp(s[:n]).min())))
        return out

    def value(s):
        return tms.parse_value(s) if s else 0.0

    for ep in range(epochs):
        model.train(); t0 = time.time(); tot = 0.0
        SX, SY = tms.synth(bank, 500) if not "--limit" in sys.argv else tms.synth(bank, 16)
        px = torch.cat([TX, SX]); py = TY + SY
        perm = rng.permutation(len(py))
        for i in range(0, len(perm), bs):
            idx = perm[i:i + bs]
            pv = to_pixels(proc, augment(px[idx])).to(DEV)
            lab = labels_of([py[j] for j in idx]).to(DEV)
            loss = model(pixel_values=pv, labels=lab).loss
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); tot += float(loss.detach())
        if ep % 3 == 2 or ep == epochs - 1:
            r = predict(VX)
            acc = np.mean([(value(s) is not None) and abs(value(s) - value(y)) < 1e-6 for (s, _), y in zip(r, VY)])
            print(f"ep {ep + 1} loss {tot / (len(perm) / bs):.3f} val value-exact {acc:.3f} ({time.time() - t0:.0f}s/ep)", flush=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir); proc.save_pretrained(out_dir); tok.save_pretrained(out_dir)
    r = predict(VX)
    print("\nVALIDATION+TEST (held-out students)")
    for lab, sel in (("all rows", lambda y: True), ("single-number / blank rows", lambda y: "+" not in y), ("sum rows (a|b, written sums)", lambda y: "+" in y)):
        idx = [i for i, y in enumerate(VY) if sel(y)]
        ok = sum((value(r[i][0]) is not None) and abs(value(r[i][0]) - value(VY[i])) < 1e-6 for i in idx)
        print(f"  {lab}: value-exact {ok}/{len(idx)}")
    for th in (0.0, 0.7, 0.8, 0.9, 0.95):
        kept = wrong = 0
        for (s, c), y in zip(r, VY):
            if c >= th or s == "":
                kept += 1; v = value(s); wrong += (v is None or abs(v - value(y)) > 1e-6)
        print(f"  conf>={th}: auto {kept}/{len(r)}, wrong among auto {wrong}")


if __name__ == "__main__":
    main()
