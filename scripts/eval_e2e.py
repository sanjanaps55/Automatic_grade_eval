"""End-to-end scoring: run the real GradePipeline (with a class list) on labelled sheets and count, per tier,
how many sheets are fully correct vs wrong. The number that must not rise: confirmed-WRONG.

Usage: python scripts/eval_e2e.py [--split heldout|val|test|train|all] [--variant orig|q85|q60|all3] [--roster data/roster_test.csv]
                                  [--tag baseline] [--out data/overnight/e2e_<tag>.json]
heldout = val + test (the 39 sheets no model trained on). Variants re-encode the photo like a phone gallery / WhatsApp would.
"""
import csv
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from grade_eval.pipeline import CONFIRM, GradePipeline
from grade_eval.roster import load_roster

VARIANTS = {"orig": None, "q85": (2000, 85), "q60": (1280, 60)}


def arg(n, d):
    return sys.argv[sys.argv.index(n) + 1] if n in sys.argv else d


def encode(path: Path, variant: str) -> bytes:
    if VARIANTS[variant] is None:
        return path.read_bytes()
    side, q = VARIANTS[variant]
    im = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)
    h, w = im.shape[:2]; s = min(1.0, side / max(h, w))
    if s < 1:
        im = cv2.resize(im, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
    return cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, q])[1].tobytes()


def num(s):
    s = (s or "").strip()
    return float(s) if s else 0.0


def tier(status, reasons):
    if status == "verified":
        return "confirmed"
    return "check" if reasons and all(r.startswith(CONFIRM) for r in reasons) else "fix"


def run(pipe_factory, split="heldout", variants=("orig",), tag="run", out=None, verbose=False):
    labels = list(csv.DictReader(open(ROOT / "data" / "labels_all.csv", encoding="utf-8-sig")))
    sp = {r["file"]: r["split"] for r in csv.DictReader(open(ROOT / "data" / "split.csv", encoding="utf-8-sig"))}
    want = {"heldout": ("val", "test"), "val": ("val",), "test": ("test",), "train": ("train",), "all": ("train", "val", "test")}[split]
    res = {}
    for var in variants:
        pipe = pipe_factory()
        c = dict(sheets=0, aligned_fail=0, confirmed_ok=0, confirmed_wrong=0, check_ok=0, check_wrong=0, fix_ok=0, fix_wrong=0)
        detail = []
        t0 = time.time()
        for r in labels:
            if sp.get(r["file"], "train") not in want:
                continue
            c["sheets"] += 1
            data = encode(ROOT / "data" / "samples" / r["file"], var)
            pipe.conn.execute("DELETE FROM sheets"); pipe.conn.commit()
            o = pipe.process(data, r["file"])
            if "could not align" in " ".join(o["reasons"]):
                c["aligned_fail"] += 1; detail.append((r["file"], "unaligned")); continue
            ok = o["reg_no"] == r["reg_no"] and all(num(r[f"q{q}"]) == (o["questions"].get(q, {}).get("a") or 0.0) for q in range(1, 11))
            t = tier(o["status"], o["reasons"])
            c[f"{t}_{'ok' if ok else 'wrong'}"] += 1
            detail.append((r["file"][6:9], t, "ok" if ok else "WRONG", [x[:60] for x in o["reasons"]][:3]))
        c["seconds"] = round(time.time() - t0)
        res[var] = {"counts": c, "detail": detail}
        n = c["sheets"] - c["aligned_fail"]
        print(f"[{tag}/{var}] sheets {c['sheets']} (unaligned {c['aligned_fail']}) | CONFIRMED ok {c['confirmed_ok']} WRONG {c['confirmed_wrong']} | "
              f"to-check ok {c['check_ok']} wrong {c['check_wrong']} | fix ok {c['fix_ok']} wrong {c['fix_wrong']} | {c['seconds']}s", flush=True)
        if verbose:
            for d in detail:
                print("   ", d)
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(json.dumps(res, indent=1), encoding="utf-8")
    return res


def main():
    variants = ("orig", "q85", "q60") if arg("--variant", "orig") == "all3" else (arg("--variant", "orig"),)
    roster = load_roster(arg("--roster", str(ROOT / "data" / "roster_test.csv")))
    tag = arg("--tag", "run")
    run(lambda: GradePipeline(db_path=":memory:", roster=roster), arg("--split", "heldout"), variants, tag,
        arg("--out", str(ROOT / "data" / "overnight" / f"e2e_{tag}.json")), verbose="--verbose" in sys.argv)


if __name__ == "__main__":
    main()
