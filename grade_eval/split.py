"""One train / validation / test split for every model, by student (reg_no) so a person never straddles two sets."""
import csv
import hashlib
from collections import Counter
from pathlib import Path

TRAIN, VAL, TEST = 0, 1, 2
NAMES = {TRAIN: "train", VAL: "val", TEST: "test"}


def split_of(reg_no: str) -> int:
    """~65% train, 20% validation (same students as the old `% 5 == 0` rule), ~15% test (scored once, at the end)."""
    b = int(hashlib.md5(reg_no.encode()).hexdigest(), 16) % 20
    return VAL if b % 5 == 0 else TEST if b in (1, 6, 11) else TRAIN


def write_split_csv(labels_csv: Path, out_csv: Path) -> Counter:
    """data/labels_all.csv -> data/split.csv (file, reg_no, split). Returns sheets per split."""
    rows = list(csv.DictReader(open(labels_csv, encoding="utf-8-sig")))
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["file", "reg_no", "split"])
        for r in rows:
            w.writerow([r["file"], r["reg_no"], NAMES[split_of(r["reg_no"])]])
    return Counter(NAMES[split_of(r["reg_no"])] for r in rows)
