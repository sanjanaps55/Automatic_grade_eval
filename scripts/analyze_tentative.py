"""How good is the best-guess reading of rows that the pipeline holds back as unclear? (held-out sheets, labelled rows)"""
import csv, os, sys
from pathlib import Path
import numpy as np
os.environ["GRADE_NO_TOTAL"] = "1"
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import cv2
from grade_eval.pipeline import GradePipeline, MARK_CONF
labels = {r["file"]: r for r in csv.DictReader(open(ROOT / "data" / "labels_all.csv", encoding="utf-8-sig"))}
split = {r["file"]: r["split"] for r in csv.DictReader(open(ROOT / "data" / "split.csv", encoding="utf-8-sig"))}
want = sys.argv[1] if len(sys.argv) > 1 else "heldout"
pipe = GradePipeline(db_path=":memory:")
bins = {}
tot_unclear = ok_unclear = 0
for f, r in labels.items():
    sp = split.get(f, "train")
    if (want == "heldout" and sp == "train") or (want == "train" and sp != "train"):
        continue
    a = cv2.imread(str(ROOT / "data" / "aligned_eval" / (Path(f).stem + ".png")))
    if a is None: continue
    q, sub, confs, reasons, cross = pipe.read_marks(a)
    flagged = {int(x.split(":")[0][1:]) for x in reasons if x.startswith("Q") and "unclear" in x}
    for (g, i, box), c in zip(pipe.last_keys, pipe.last_cands):
        if i == 10 or i + 1 > 10: continue
        qn = i + 1
        truth = float(r[f"q{qn}"] or 0)
        if qn in flagged and c:
            v, lp = max(c.items(), key=lambda kv: kv[1]); p = float(np.exp(lp))
            tot_unclear += 1; ok_unclear += abs(v - truth) < 1e-6
            b = "p>=.6" if p >= .6 else "p.3-.6" if p >= .3 else "p<.3"
            n_ok = bins.setdefault(b, [0, 0]); n_ok[0] += 1; n_ok[1] += abs(v - truth) < 1e-6
print(want, "unclear rows:", tot_unclear, "best guess right:", ok_unclear, {k: f"{v[1]}/{v[0]}" for k, v in bins.items()})
