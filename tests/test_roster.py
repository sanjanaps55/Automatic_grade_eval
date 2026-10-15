import csv

from grade_eval.roster import RosterRow, load_roster, match_sheet, name_similarity

ROSTER = [
    RosterRow("24MCS1073", "Dr Madura Meenakshi"),
    RosterRow("24MCS1077", "Yash Malhotra"),
    RosterRow("23BAI1337", "Diya Choudhuri"),
    RosterRow("23BAI1338", "Rohan Verma"),
]


def test_exact_reg_and_name_agree():
    m = match_sheet("23BAI1337", "Diya Choudhuri", ROSTER)
    assert m.status == "auto" and m.row.reg_no == "23BAI1337"


def test_exact_reg_unreadable_name_is_reg_only():
    m = match_sheet("23BAI1337", "", ROSTER)
    assert m.status == "auto_reg_only" and m.row.reg_no == "23BAI1337"


def test_name_contradicting_exact_hit_goes_to_review():
    # read 1077 (a valid student, Yash) but the name is Madura's (1073): conflict -> review, never auto-accept either
    for name in ("Madura Meenakshi", "Dr Madura Meenakshi"):
        m = match_sheet("24MCS1077", name, ROSTER)
        assert m.status == "review" and "too close" in " ".join(m.reasons)


def test_one_digit_slip_with_clean_name_is_fixed():
    roster = ROSTER + [RosterRow("24MCS1999", "Far Away")]
    m = match_sheet("24MCS1074", "Dr Madura Meenakshi", roster)   # 1074 is no roster row; 1073 is 1 edit and the name matches
    assert m.status == "auto" and m.row.reg_no == "24MCS1073"


def test_slip_without_name_goes_to_review():
    m = match_sheet("23BAI1339", "", ROSTER)          # 1 edit from two rows -> ambiguous, no name to break the tie
    assert m.status == "review"


def test_nothing_close_goes_to_review():
    assert match_sheet("21XXX0000", "Someone Else", ROSTER).status == "review"


def test_name_similarity_ignores_titles_and_order():
    assert name_similarity("Dr. Madura Meenakshi", "Meenakshi Madura") == 1.0
    assert name_similarity("", "Anyone") == 0.0


def test_load_roster_csv_and_xlsx(tmp_path):
    p = tmp_path / "r.csv"
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["Register No", "Student Name"]); w.writerow(["24MCS1073", "Madura"]); w.writerow(["24mcs1073", "dup"])
    assert [(r.reg_no, r.name) for r in load_roster(p)] == [("24MCS1073", "Madura")]
    from openpyxl import Workbook
    wb = Workbook(); ws = wb.active; ws.append(["Name", "Reg. No."]); ws.append(["Yash", " 24MCS1077 "])
    x = tmp_path / "r.xlsx"; wb.save(x)
    assert [(r.reg_no, r.name) for r in load_roster(x)] == [("24MCS1077", "Yash")]
