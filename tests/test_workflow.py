"""Pure-logic tests for the sum parser, tiering and the roster-ordered Excel (no model weights needed)."""
import json

from openpyxl import load_workbook

from grade_eval.pipeline import CONFIRM, parse_sum
from grade_eval.roster import RosterRow
from grade_eval.server import tier_of
from grade_eval.store import db
from grade_eval.store.excel import export_roster_excel


def test_parse_sum():
    assert parse_sum("7") == 7.0
    assert parse_sum("3+5+1") == 9.0
    assert parse_sum("2.5+1") == 3.5
    assert parse_sum("") is None
    assert parse_sum("3+") is None
    assert parse_sum("++") is None
    assert parse_sum("123") is None          # three digits is never a mark


def test_tiers():
    assert tier_of("verified", []) == "confirmed"
    assert tier_of("needs_review", [CONFIRM + "no sub-total"]) == "check"
    assert tier_of("needs_review", [CONFIRM + "x", "Q3: unclear read '4'"]) == "fix"
    assert tier_of("needs_review", ["could not align"]) == "fix"


def test_roster_excel_order_absent_and_extras(tmp_path):
    conn = db.connect(":memory:")
    db.save_sheet(conn, "h1", "23BCE0002", "Bee", "T", {1: {"a": 4.0, "b": None}, 2: {"a": 6.0, "b": None}}, [], {"file": "a.jpg"})
    db.save_sheet(conn, "h2", "99XXX9999", "Stranger", "T", {1: {"a": 1.0, "b": None}}, [], {"file": "b.jpg"})
    db.save_sheet(conn, "h3", "23BCE0003", "", "T", {}, ["Q1: unclear read '4'"], {"file": "c.jpg", "reg_raw": "23BCE0003"})
    roster = [RosterRow("23BCE0001", "Ann"), RosterRow("23BCE0002", "Bee"), RosterRow("23BCE0003", "Cat")]
    out = export_roster_excel(conn, roster, 3, tmp_path / "m.xlsx")
    wb = load_workbook(out)
    rows = list(wb["Marks"].iter_rows(values_only=True))
    assert [r[0] for r in rows[1:]] == ["23BCE0001", "23BCE0002", "23BCE0003", "99XXX9999"]   # her order, extras last
    assert rows[2][2:5] == (4, 6, None) and rows[2][5] == 10                                  # Bee: Q1, Q2, Q3, total
    assert rows[1][-1].startswith("absent") and rows[4][-1] == "NOT ON CLASS LIST"
    issues = list(wb["Needs attention"].iter_rows(values_only=True))
    assert issues[1][0] == "c.jpg" and "unclear" in issues[1][2]
