import pytest
from openpyxl import load_workbook

from grade_eval.exam_config import parse_config
from grade_eval.store import db
from grade_eval.store.excel import export_excel
from grade_eval.validate import repair_reg_no, validate_sheet

CONFIG = parse_config({
    "course_code": "TEST101", "total_max": 50,
    "questions": {1: {"max": 10}, 2: {"a_max": 5, "b_max": 5}},
})


@pytest.fixture
def conn():
    return db.connect(":memory:")


def test_repair_reg_no():
    assert repair_reg_no("23BAI1337") == "23BAI1337"
    assert repair_reg_no("2 3 BA1 l337") == "23BAI1337"
    assert repair_reg_no("24MC51074") == "24MCS1074"
    assert repair_reg_no("24MCSIO74") == "24MCS1074"


def test_a_plus_b_is_summed_and_checked():
    ok = validate_sheet("23BAI1337", {1: {"a": 4}, 2: {"a": 3, "b": 2}}, 9, CONFIG)
    assert ok == []
    bad = validate_sheet("23BAI1337", {2: {"a": 3, "b": 8}}, 11, CONFIG)
    assert any("part b exceeds" in r for r in bad)
    assert any("exceed max" in r for r in bad)


def test_flags_over_max_subtotal_and_missing_config():
    reasons = validate_sheet("23BAI1337", {1: {"a": 12}}, 10, CONFIG)
    assert any("exceed max 10" in r for r in reasons)
    assert any("sub-total" in r for r in reasons)
    assert "no exam config" in validate_sheet("23BAI1337", {1: {"a": 1}}, None, None)
    assert any("not in exam config" in r for r in validate_sheet("23BAI1337", {7: {"a": 1}}, None, CONFIG))


def test_same_image_twice_is_one_record(conn):
    h = db.hash_image(b"img")
    assert db.save_sheet(conn, h, "23BAI1337", "Yash", "TEST101", {1: {"a": 4}}, [], {}) == "verified"
    assert db.save_sheet(conn, h, "23BAI1337", "Yash", "TEST101", {1: {"a": 4}}, [], {}) == "duplicate"
    assert conn.execute("SELECT COUNT(*) FROM students").fetchone()[0] == 1
    assert conn.execute("SELECT COUNT(*) FROM sheets").fetchone()[0] == 1


def test_rescan_of_same_student_upserts(conn):
    db.save_sheet(conn, "h1", "23BAI1337", "Yash", "TEST101", {1: {"a": 4}, 2: {"a": 3}}, [], {})
    db.save_sheet(conn, "h2", "23BAI1337", "Yash M", "TEST101", {1: {"a": 5}}, [], {})
    assert conn.execute("SELECT COUNT(*) FROM students").fetchone()[0] == 1
    rows = conn.execute("SELECT q_no, total FROM marks").fetchall()
    assert [(r["q_no"], r["total"]) for r in rows] == [(1, 5)]


def test_flagged_sheet_not_written_until_approved(conn):
    status = db.save_sheet(conn, "h1", "23BAI1337", "Yash", "TEST101", {1: {"a": 12}}, ["Q1 exceeds"], {})
    assert status == "needs_review"
    assert conn.execute("SELECT COUNT(*) FROM students").fetchone()[0] == 0
    sheet_id = conn.execute("SELECT id FROM sheets").fetchone()[0]
    db.approve_sheet(conn, sheet_id, "23BAI1337", "Yash", {1: {"a": 2}})
    assert conn.execute("SELECT total FROM marks").fetchone()[0] == 2
    assert conn.execute("SELECT status FROM sheets").fetchone()[0] == "verified"


def test_excel_export(conn, tmp_path):
    db.save_sheet(conn, "h1", "23BAI1337", "Yash", "TEST101", {1: {"a": 4}, 2: {"a": 3, "b": 2}}, [], {})
    db.save_sheet(conn, "h2", "23BAI1642", "Vikash", "TEST101", {1: {"a": 5}}, [], {})
    ws = load_workbook(export_excel(conn, tmp_path / "m.xlsx")).active
    rows = list(ws.values)
    assert rows[0] == ("Register No", "Name", "Q1", "Q2", "Total")
    assert rows[1] == ("23BAI1337", "Yash", 4, 5, 9)
    assert rows[2] == ("23BAI1642", "Vikash", 5, None, 5)
