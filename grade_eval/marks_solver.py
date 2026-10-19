"""Choose the per-question readings that add up to the sub-total written on the sheet.

Each marks row is read by a CTC model. Instead of keeping only its single best string, `candidates` returns several
readings per row (with probabilities) and `solve` picks the combination whose values add up to one of the sub-total
readings. A wrong reading rarely adds up to the written total by accident, so a unique match is strong evidence.
"""
import math
import re

import numpy as np

VOCAB = "0123456789.+"
_SUM_RE = re.compile(r"\d{1,2}(\.\d)?(\+\d{1,2}(\.\d)?)*")


def _lse(a, b):
    if a == -math.inf:
        return b
    if b == -math.inf:
        return a
    m = max(a, b)
    return m + math.log(math.exp(a - m) + math.exp(b - m))


def beam_search(logp: np.ndarray, width: int = 16, prune: float = -12.0):
    """CTC prefix beam search. logp: (T, len(VOCAB)+1) log-probabilities, blank = 0. -> [(string, logprob)] best first."""
    beams = {"": (0.0, -math.inf)}                       # prefix -> (log p ending in blank, log p ending in a symbol)
    for t in range(logp.shape[0]):
        row = logp[t]
        nxt = {}
        def add(pre, pb, pnb):
            ob, onb = nxt.get(pre, (-math.inf, -math.inf))
            nxt[pre] = (_lse(ob, pb), _lse(onb, pnb))
        keep = [k for k in range(len(row)) if row[k] > prune]
        for pre, (pb, pnb) in beams.items():
            tot = _lse(pb, pnb)
            for k in keep:
                p = float(row[k])
                if k == 0:
                    add(pre, tot + p, -math.inf)
                    continue
                ch = VOCAB[k - 1]
                if pre and pre[-1] == ch:
                    add(pre + ch, -math.inf, pb + p)       # repeat only counts after a blank
                    add(pre, -math.inf, pnb + p)           # same char, no blank: collapses into the same prefix
                else:
                    add(pre + ch, -math.inf, tot + p)
        beams = dict(sorted(nxt.items(), key=lambda kv: -_lse(*kv[1]))[:width])
    return sorted(((s, _lse(*v)) for s, v in beams.items()), key=lambda x: -x[1])


def parse_sum(s: str):
    if not _SUM_RE.fullmatch(s or ""):
        return None
    return sum(float(t) for t in s.split("+"))


def candidates(logits: np.ndarray, keep: int = 6, spread: float = 9.0):
    """One row -> {value: logprob}. value is None-free: a blank row is the value 0.0 (also what the string '' means)."""
    logp = logits - np.logaddexp.reduce(logits, axis=-1, keepdims=True)
    out = {}
    for s, lp in beam_search(logp):
        v = 0.0 if s == "" else parse_sum(s)
        if v is None or v > 30:
            continue
        out[v] = _lse(out.get(v, -math.inf), lp)
    items = sorted(out.items(), key=lambda kv: -kv[1])
    if not items:
        return {}
    return {v: lp for v, lp in items[:keep] if lp > items[0][1] - spread}


def solve(rows: list[dict], subs: dict, top: int = 2):
    """rows: per-question {value: logprob}; subs: {sub_total: logprob}.

    -> (best_values, margin, n_solutions): best_values is the list of per-row values of the best combination that
    adds up to some sub-total candidate (None when none does); margin is the log-probability gap to the runner-up
    combination with different values (inf when it is the only one).
    """
    T = lambda v: int(round(v * 10))
    dp = {0: [(0.0, ())]}
    for r in rows:
        nd = {}
        for s, lst in dp.items():
            for v, lp in r.items():
                for sc, asg in lst:
                    nd.setdefault(s + T(v), []).append((sc + lp, asg + (v,)))
        dp = {s: sorted(l, key=lambda x: -x[0])[:top * 4] for s, l in nd.items()}
    sols = []
    for sub, slp in subs.items():
        for sc, asg in dp.get(T(sub), []):
            sols.append((sc + slp, asg))
    if not sols:
        return None, 0.0, 0
    sols.sort(key=lambda x: -x[0])
    best = sols[0]
    other = next((s for s in sols[1:] if s[1] != best[1]), None)
    margin = math.inf if other is None else best[0] - other[0]
    return list(best[1]), margin, len(sols)
