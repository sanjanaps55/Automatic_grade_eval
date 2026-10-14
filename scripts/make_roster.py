"""Build a TEST roster (data/roster_test.csv): every labelled student + invented neighbours (reg_no +-1..3).

Dense on purpose: a real class has consecutive numbers, so one wrong digit often lands on another valid student.
Columns: reg_no,name,invented. Replace with the real class roster when available.
"""
import csv
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
random.seed(7)
FIRST = "Aarav Ananya Arjun Diya Ishaan Kavya Meera Nikhil Priya Rohan Sneha Tanvi Varun Zoya Harsh Pooja Karthik Lakshmi".split()
LAST = "Sharma Iyer Reddy Nair Patel Gupta Menon Rao Kumar Singh Das Bose Pillai Verma Joshi".split()
real = {}
for r in csv.DictReader(open(ROOT / "data" / "labels_all.csv", encoding="utf-8-sig")):
    real.setdefault(r["reg_no"], r["name"])
out = {k: (v, 0) for k, v in real.items()}
for k in real:
    for d in (-3, -2, -1, 1, 2, 3):
        n = int(k[-4:]) + d
        if 0 < n < 10000:
            nk = k[:-4] + f"{n:04d}"
            out.setdefault(nk, (f"{random.choice(FIRST)} {random.choice(LAST)}", 1))
with open(ROOT / "data" / "roster_test.csv", "w", newline="", encoding="utf-8-sig") as f:
    w = csv.writer(f); w.writerow(["reg_no", "name", "invented"])
    for k in sorted(out): w.writerow([k, *out[k]])
print(len(real), "real +", sum(v[1] for v in out.values()), "invented =", len(out))
