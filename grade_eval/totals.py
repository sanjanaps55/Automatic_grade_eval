"""Read the written total of a sheet: the Sub-Total cell and the Grand Total box.

The total is written in the Sub-Total cell on about half the sheets and in the Grand Total box (often circled, sometimes as
words) on the other half; some sheets have it in both places. Two small CTC readers (same ResNet18 body as the marks reader):
  * `sub`  reads the ink-only strip of the first Sub-Total cell,
  * `grand` reads K overlapping horizontal bands of the Grand Total box (the handwriting height varies per sheet; a band that
    holds only words or half a number must read as blank).
`read_totals` returns candidate values with probabilities from every source; `grade_eval.pipeline` uses them to cross-check
the question marks.
"""
import math
from pathlib import Path

import cv2
import numpy as np
import torch

from grade_eval import marks_solver
from grade_eval.vision import cells

ROOT = Path(__file__).resolve().parents[1]
GRAND_X = (1140, 1540)                     # Grand Total box in template coordinates (the aligned sheet is 1540 px wide)
BAND_Y0, BAND_H, BAND_STRIDE, N_BANDS = 1170, 90, 50, 5
BAND_W, BAND_HP = 320, 64                  # band strips are resized to 64 x 320
MAX_TOTAL = 99.0


def sub_strip(mask: np.ndarray) -> np.ndarray:
    """First Sub-Total cell (a + b columns + spill) -> (64, 192) uint8, ink dark."""
    return cells.row_strip(mask, 10)


def grand_bands(mask: np.ndarray) -> np.ndarray:
    """-> (N_BANDS, 64, 320) uint8 strips of the Grand Total box."""
    h, w = mask.shape
    out = []
    for k in range(N_BANDS):
        y0 = BAND_Y0 + k * BAND_STRIDE
        m = mask[y0:y0 + BAND_H, GRAND_X[0]:min(w, GRAND_X[1])].astype(np.uint8) * 255
        out.append(255 - cv2.resize(m, (BAND_W, BAND_HP), interpolation=cv2.INTER_AREA))
    return np.stack(out)


def value_candidates(logits: np.ndarray, keep: int = 4, spread: float = 8.0) -> dict:
    """One strip's CTC logits (T, C) -> {value: logprob}; the blank reading is the key None. Values above MAX_TOTAL are dropped."""
    logp = logits - np.logaddexp.reduce(logits, axis=-1, keepdims=True)
    out = {}
    for s, lp in marks_solver.beam_search(logp):
        v = None if s == "" else marks_solver.parse_sum(s)
        if s != "" and (v is None or v > MAX_TOTAL):
            continue
        out[v] = np.logaddexp(out.get(v, -math.inf), lp)
    items = sorted(out.items(), key=lambda kv: -kv[1])[: keep + 1]
    return {v: lp for v, lp in items if lp > items[0][1] - spread}


def pool_totals(sources: list) -> dict:
    """Pool the per-source candidate dicts into {value: probability} (a value's probability = its best single source)."""
    totals = {}
    for s in sources:
        for v, lp in s["cands"].items():
            if v is not None:
                totals[v] = max(totals.get(v, 0.0), float(math.exp(lp)))
    return totals


class TotalReader:
    def __init__(self, sub_path: Path, grand_path: Path, device: str = "cpu"):
        import train_marks_sum as tms
        self.dev = device
        self.sub = tms.make_net(); self.sub.load_state_dict(torch.load(sub_path, map_location="cpu")); self.sub.to(device).eval()
        self.grand = tms.make_net(); self.grand.load_state_dict(torch.load(grand_path, map_location="cpu")); self.grand.to(device).eval()

    @torch.no_grad()
    def logits(self, mask: np.ndarray):
        s = torch.tensor(sub_strip(mask).astype(np.float32) / 255).view(1, 1, 64, 192).to(self.dev)
        g = torch.tensor(grand_bands(mask).astype(np.float32) / 255).unsqueeze(1).to(self.dev)
        return self.sub(s)[0].cpu().numpy(), self.grand(g).cpu().numpy()

    def read_totals(self, mask: np.ndarray) -> dict:
        """-> {"sources": [{"src": "sub"|"band k", "cands": {value|None: logprob}}], "totals": {value: prob}}.

        `totals` pools every source that read a number: a value's probability is the best single source's probability of
        that value (a number is normally present in only one place, or the same number in both).
        """
        sl, gl = self.logits(mask)
        sources = [{"src": "sub", "cands": value_candidates(sl)}]
        sources += [{"src": f"band{k}", "cands": value_candidates(gl[k])} for k in range(gl.shape[0])]
        return {"sources": sources, "totals": pool_totals(sources)}


# --------------------------------------------------------------------------------------------------------------------
# Likelihood-ratio evidence: how well do the total strips match a *given* number, compared with "nothing written"?
# Decoding the total (argmax) collapses repeated digits and breaks on circles; scoring a candidate string does not.
# --------------------------------------------------------------------------------------------------------------------
def fmt_total(v: float) -> str:
    return str(int(round(v))) if abs(v - round(v)) < 1e-9 else f"{v:g}"


def string_loglik(logits: np.ndarray, strings: list) -> np.ndarray:
    """log P(string | CTC logits) for each string; the empty string is allowed. -> (len(strings),)"""
    from train_marks_sum import VOCAB
    import torch.nn.functional as F
    lp = torch.log_softmax(torch.tensor(logits, dtype=torch.float32), -1)          # (T, C)
    n = len(strings); T = lp.shape[0]
    enc = [[VOCAB.index(c) + 1 for c in s] for s in strings]
    out = F.ctc_loss(lp.unsqueeze(1).expand(T, n, -1), torch.tensor(sum(enc, []), dtype=torch.long),
                     torch.full((n,), T, dtype=torch.long), torch.tensor([len(e) for e in enc], dtype=torch.long),
                     blank=0, reduction="none", zero_infinity=False)
    ll = -out.numpy()
    return np.where(np.isfinite(ll), ll, -1e4)


def evidence(sub_logits: np.ndarray, band_logits: np.ndarray, values: list) -> np.ndarray:
    """LR(v) = best over the strips of log P(strip reads fmt(v)) - log P(strip reads nothing), for each candidate total v."""
    strings = [""] + [fmt_total(v) for v in values]
    best = np.full(len(values), -1e4)
    for lg in [sub_logits] + list(band_logits):
        ll = string_loglik(lg, strings)
        best = np.maximum(best, ll[1:] - ll[0])
    return best
