"""Class roster (register no. + name) and joint matching of what the readers extracted.

Both fields vote for one roster row. A one-digit slip in the register number is fixed when the name agrees, and a
confident-looking wrong register number is caught when the name points elsewhere. Anything unclear goes to review,
so a misread never creates (or overwrites) the wrong student.
"""
import csv
import re
from dataclasses import dataclass, field
from pathlib import Path

REG_PTS = {0: 2.0, 1: 0.6, 2: 0.0}      # register no. edit distance -> evidence
NAME_W = 2.0                            # weight of name similarity
MIN_SCORE = 2.0                         # best row must reach this ...
MIN_MARGIN = 1.0                        # ... and lead the runner-up by this
STRONG_NAME = 0.75                      # similarity at which the name "agrees"
MIN_NAME_LEN = 3                        # shorter reads carry no evidence
_TITLES = {"dr", "mr", "ms", "mrs", "prof"}


@dataclass
class RosterRow:
    reg_no: str
    name: str


@dataclass
class Match:
    status: str                         # "auto" (reg + name agree) | "auto_reg_only" | "review"
    row: RosterRow | None = None
    score: float = 0.0
    margin: float = 0.0
    reasons: list[str] = field(default_factory=list)


def lev(a: str, b: str) -> int:
    d = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        prev, d[0] = d[0], i
        for j, cb in enumerate(b, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (ca != cb))
    return d[-1]


def norm_name(s: str) -> str:
    """Lower-case, drop punctuation and titles, sort the words: 'Dr. Madura Meenakshi' ~ 'Meenakshi Madura'."""
    words = [w for w in re.sub(r"[^a-z\s]", " ", s.lower()).split() if w not in _TITLES]
    return " ".join(sorted(words))


def name_similarity(read: str, true: str) -> float:
    a, b = norm_name(read), norm_name(true)
    if not a or not b:
        return 0.0
    return 1 - lev(a, b) / max(len(a), len(b))


def load_roster(path: str | Path) -> list[RosterRow]:
    """Read the class list from .xlsx or .csv. Needs a register-no. column and a name column (any header casing)."""
    path = Path(path)
    if path.suffix.lower() in (".xlsx", ".xlsm"):
        from openpyxl import load_workbook
        rows = [list(r) for r in load_workbook(path, read_only=True, data_only=True).active.iter_rows(values_only=True)]
    else:
        rows = list(csv.reader(open(path, encoding="utf-8-sig")))
    rows = [[("" if c is None else str(c)).strip() for c in r] for r in rows if any(c not in (None, "") for c in r)]
    head = [h.lower().replace(" ", "").replace(".", "").replace("_", "") for h in rows[0]]
    reg_i = next((i for i, h in enumerate(head) if h in ("regno", "registerno", "registernumber", "regnumber", "register")), None)
    name_i = next((i for i, h in enumerate(head) if "name" in h), None)
    if reg_i is None or name_i is None:
        raise ValueError(f"roster needs register-number and name columns, found headers {rows[0]}")
    out, seen = [], set()
    for r in rows[1:]:
        reg = re.sub(r"\s+", "", r[reg_i]).upper()
        if reg and reg not in seen:
            seen.add(reg); out.append(RosterRow(reg, r[name_i]))
    return out


def match_sheet(reg_read: str, name_read: str, roster: list[RosterRow], reg_conf: float | None = None) -> Match:
    """Pick the roster row both readings agree on, or send the sheet to review."""
    reg_read = re.sub(r"\s+", "", reg_read).upper()
    has_name = len(norm_name(name_read).replace(" ", "")) >= MIN_NAME_LEN
    scored = []
    for r in roster:
        d = lev(reg_read, r.reg_no)
        if d > 2:
            continue
        sim = name_similarity(name_read, r.name) if has_name else 0.0
        pts = REG_PTS[d] - (0.4 if d == 0 and reg_conf is not None and reg_conf < 0.9 else 0.0)
        scored.append((pts + NAME_W * sim, d, sim, r))
    if not scored:
        return Match("review", reasons=["no roster register number within 2 edits"])
    scored.sort(key=lambda t: -t[0])
    best = scored[0]
    margin = best[0] - (scored[1][0] if len(scored) > 1 else 0.0)
    score, d, sim, row = best
    why = [f"reg distance {d}", f"name similarity {sim:.2f}" if has_name else "name unreadable"]
    if score < MIN_SCORE:
        return Match("review", row, score, margin, why + ["weak evidence"])
    if margin < MIN_MARGIN:
        return Match("review", row, score, margin, why + [f"too close to {scored[1][3].reg_no}"])
    return Match("auto" if has_name and sim >= STRONG_NAME else "auto_reg_only", row, score, margin, why)
