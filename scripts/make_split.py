"""Write data/split.csv (file, reg_no, split) from data/labels_all.csv and check no student is in two sets."""
import csv
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grade_eval.split import write_split_csv

counts = write_split_csv(ROOT / "data" / "labels_all.csv", ROOT / "data" / "split.csv")
where = defaultdict(set)
for r in csv.DictReader(open(ROOT / "data" / "split.csv")):
    where[r["reg_no"]].add(r["split"])
assert all(len(s) == 1 for s in where.values()), "a student appears in two splits"
print("sheets:", dict(counts), "| students:", len(where))
