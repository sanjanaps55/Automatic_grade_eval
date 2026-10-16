import hashlib
import json
import sqlite3
from pathlib import Path

DEFAULT_DB = Path(__file__).resolve().parents[2] / "data" / "grades.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS students (
    reg_no TEXT PRIMARY KEY,
    name TEXT
);
CREATE TABLE IF NOT EXISTS marks (
    reg_no TEXT NOT NULL REFERENCES students(reg_no),
    q_no INTEGER NOT NULL,
    a REAL,
    b REAL,
    total REAL NOT NULL,
    PRIMARY KEY (reg_no, q_no)
);
CREATE TABLE IF NOT EXISTS sheets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    image_hash TEXT NOT NULL UNIQUE,
    reg_no TEXT,
    course_code TEXT,
    status TEXT NOT NULL,           -- verified | needs_review
    reasons TEXT NOT NULL DEFAULT '[]',
    extracted TEXT NOT NULL         -- JSON of the raw extraction, used by the review UI
);
"""


def connect(path: Path | str = DEFAULT_DB) -> sqlite3.Connection:
    if str(path) != ":memory:":
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, check_same_thread=False)  # the web server calls from worker threads (guarded by a lock there)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    return conn


def hash_image(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sheet_exists(conn, image_hash: str) -> bool:
    return conn.execute("SELECT 1 FROM sheets WHERE image_hash = ?", (image_hash,)).fetchone() is not None


def save_sheet(conn, image_hash, reg_no, name, course_code, questions, reasons, extracted) -> str:
    """Record one scanned sheet. Returns 'duplicate', 'verified' or 'needs_review'.

    The same image twice is ignored (image_hash). A re-scan of the same student
    upserts that student's rows, so there is never more than one record per reg_no.
    Only clean sheets write to students/marks; flagged ones wait in `sheets` for review.
    """
    if sheet_exists(conn, image_hash):
        return "duplicate"
    status = "needs_review" if reasons else "verified"
    with conn:
        conn.execute(
            "INSERT INTO sheets (image_hash, reg_no, course_code, status, reasons, extracted)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (image_hash, reg_no, course_code, status, json.dumps(reasons), json.dumps(extracted)),
        )
        if status == "verified":
            write_marks(conn, reg_no, name, questions)
    return status


def write_marks(conn, reg_no, name, questions: dict[int, dict]):
    conn.execute(
        "INSERT INTO students (reg_no, name) VALUES (?, ?)"
        " ON CONFLICT(reg_no) DO UPDATE SET name = excluded.name",
        (reg_no, name),
    )
    conn.execute("DELETE FROM marks WHERE reg_no = ?", (reg_no,))
    for q_no, parts in questions.items():
        a, b = parts.get("a"), parts.get("b")
        conn.executemany(
            "INSERT INTO marks (reg_no, q_no, a, b, total) VALUES (?, ?, ?, ?, ?)",
            [(reg_no, q_no, a, b, (a or 0) + (b or 0))],
        )


def approve_sheet(conn, sheet_id: int, reg_no, name, questions: dict[int, dict]):
    """Review-UI action: store the human-corrected values and mark the sheet verified."""
    with conn:
        write_marks(conn, reg_no, name, questions)
        conn.execute(
            "UPDATE sheets SET status = 'verified', reg_no = ?, reasons = '[]' WHERE id = ?",
            (reg_no, sheet_id),
        )
