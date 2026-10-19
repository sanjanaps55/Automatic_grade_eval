"""Does the written sub-total help? Held-out evaluation of the sub-total search (grade_eval/marks_solver.py).

Rows are read by the routed models (marks_sum.pt on simple rows, hard_b_s0.pt on hard rows). For sheets whose
sub-total is labelled, the solver picks the row readings that add up to the sub-total reading.
    python scripts/eval_subtotal.py
"""
import csv
import math
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import numpy as np
import torch

import train_marks_hard as th
import train_marks_sum as tms
from grade_eval import marks_solver as ms
from grade_eval.vision import cells, layout
from grade_eval.vision.preprocess import load_image

CACHE = ROOT / "data" / "subtotal_eval.pkl"


def collect():
    d = torch.load(ROOT / "models" / "hard_b_s0.pt", map_location="cpu")
    hard = tms.make_net(); hard.load_state_dict(d["state"]); hard.eval()
    cur = tms.make_net(); cur.load_state_dict(torch.load(ROOT / "models" / "marks_sum.pt", map_location="cpu")); cur.eval()
    labels = {r["file"]: r for r in csv.DictReader(open(ROOT / "data" / "labels_all.csv", encoding="utf-8-sig"))}
    split = {r["file"]: r["split"] for r in csv.DictReader(open(ROOT / "data" / "split.csv", encoding="utf-8-sig"))}
    out = []
    for f, L in labels.items():
        if split.get(f) not in ("train", "val", "test"):
            continue
        res = layout.align(load_image(open(ROOT / "data" / "samples" / f, "rb").read()))
        if res is None:
            continue
        mask = cells.ink_mask(res.image)
        X = torch.tensor(np.stack([cells.row_strip(mask, i) for i in range(11)]).astype(np.float32) / 255).unsqueeze(1)
        with torch.no_grad():
            lc = cur(X).numpy(); lh = th.forward(hard, X, d["coord"]).numpy()
        simple = [cells.is_simple(mask, i) for i in range(11)]
        lab = [float(L[f"q{i + 1}"]) if L[f"q{i + 1}"] != "" else 0.0 for i in range(10)]
        sub = float(L["sub_total"]) if L["sub_total"] != "" else None
        out.append(dict(file=f, split=split[f], lc=lc, lh=lh, simple=simple, label=lab, sub=sub))
    pickle.dump(out, open(CACHE, "wb"))
    return out


def main():
    data = pickle.load(open(CACHE, "rb")) if CACHE.exists() else collect()
    for sp in (("val", "test"), ("train",)):
        S = [r for r in data if r["split"] in sp]
        print(f"\n== {'/'.join(sp)}: {len(S)} sheets, {sum(r['sub'] is not None for r in S)} with a labelled sub-total"
              + ("  (train = optimistic, models saw these)" if sp == ("train",) else ""))
        base_wrong = base_rows = 0
        res = []
        for r in S:
            if r["sub"] is None:
                continue
            rows = [ms.candidates(r["lc"][i] if r["simple"][i] else r["lh"][i]) for i in range(10)]
            subs = ms.candidates(r["lc"][10])
            argmax = [max(c, key=c.get) if c else None for c in rows]
            base_ok = [a is not None and abs(a - l) < 1e-6 for a, l in zip(argmax, r["label"])]
            sol, margin, n = ms.solve(rows, subs)
            sol_ok = None if sol is None else [abs(a - l) < 1e-6 for a, l in zip(sol, r["label"])]
            res.append(dict(file=r["file"], base_ok=base_ok, sol=sol, sol_ok=sol_ok, margin=margin,
                            sub_ok=any(abs(v - r["sub"]) < 1e-6 for v in subs), sub_top=max(subs, key=subs.get) if subs else None, sub=r["sub"]))
        if not res:
            continue
        print(f"  sub-total reading: top candidate right {sum(x['sub_top'] == x['sub'] for x in res)}/{len(res)}, "
              f"right one among candidates {sum(x['sub_ok'] for x in res)}/{len(res)}")
        print(f"  rows right, argmax only: {sum(sum(x['base_ok']) for x in res)}/{10 * len(res)}; "
              f"sheets fully right: {sum(all(x['base_ok']) for x in res)}/{len(res)}")
        for M in (0, 1, 2, 3, 5):
            acc = [x for x in res if x["sol"] is not None and x["margin"] >= M]
            right = sum(all(x["sol_ok"]) for x in acc)
            print(f"  solver margin>={M}: accepts {len(acc)}/{len(res)} sheets, fully right {right}, wrong sheets {len(acc) - right} "
                  f"(argmax alone on the same sheets: {sum(all(x['base_ok']) for x in acc)} right)")
        for x in res:
            tag = "ok" if x["sol_ok"] and all(x["sol_ok"]) else ("no-solution" if x["sol"] is None else "WRONG")
            print(f"    {x['file']} sub={x['sub']:g} read={x['sub_top']} argmax_ok={all(x['base_ok'])} solver={tag} margin={x['margin']:.1f}")


if __name__ == "__main__":
    main()
