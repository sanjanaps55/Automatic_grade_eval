import json
from pathlib import Path

from openpyxl import Workbook

DEFAULT_XLSX = Path(__file__).resolve().parents[2] / "marks.xlsx"


def export_excel(conn, path: Path | str = DEFAULT_XLSX) -> Path:
    """Regenerate the workbook from verified rows: one row per student, one column per question."""
    q_nos = [r[0] for r in conn.execute("SELECT DISTINCT q_no FROM marks ORDER BY q_no")]
    wb = Workbook()
    ws = wb.active
    ws.title = "Marks"
    ws.append(["Register No", "Name", *[f"Q{q}" for q in q_nos], "Total"])

    for s in conn.execute("SELECT reg_no, name FROM students ORDER BY reg_no"):
        by_q = {r["q_no"]: r["total"] for r in conn.execute(
            "SELECT q_no, total FROM marks WHERE reg_no = ?", (s["reg_no"],))}
        ws.append([s["reg_no"], s["name"], *[by_q.get(q) for q in q_nos], sum(by_q.values())])

    wb.save(path)
    return Path(path)


def export_roster_excel(conn, roster, n_questions: int, path: Path | str) -> Path:
    """Teacher's class list, in her order, with each scanned student's marks in that row.

    `roster`: list of objects with .reg_no and .name (grade_eval.roster.RosterRow). Absent students stay blank.
    Verified students who are not on the list are appended and marked. A second sheet lists what still needs attention.
    """
    from openpyxl.styles import Font, PatternFill

    marks = {}
    for r in conn.execute("SELECT reg_no, q_no, total FROM marks"):
        marks.setdefault(r["reg_no"], {})[r["q_no"]] = r["total"]
    names = {r["reg_no"]: r["name"] for r in conn.execute("SELECT reg_no, name FROM students")}

    wb = Workbook()
    ws = wb.active
    ws.title = "Marks"
    ws.append(["Register No", "Name", *[f"Q{q}" for q in range(1, n_questions + 1)], "Total", "Status"])
    for c in ws[1]:
        c.font = Font(bold=True)
    on_list = set()
    for row in roster:
        on_list.add(row.reg_no)
        m = marks.get(row.reg_no)
        ws.append([row.reg_no, row.name, *[(m or {}).get(q) for q in range(1, n_questions + 1)],
                   sum(m.values()) if m else None, "scanned" if m else "absent / not scanned"])
    warn = PatternFill("solid", fgColor="FFF2CC")
    for reg, m in sorted(marks.items()):
        if reg not in on_list:
            ws.append([reg, names.get(reg, ""), *[m.get(q) for q in range(1, n_questions + 1)], sum(m.values()), "NOT ON CLASS LIST"])
            for c in ws[ws.max_row]:
                c.fill = warn

    iss = wb.create_sheet("Needs attention")
    iss.append(["Photo", "Register No (read)", "Why"])
    for c in iss[1]:
        c.font = Font(bold=True)
    for s in conn.execute("SELECT reg_no, reasons, extracted FROM sheets WHERE status = 'needs_review' ORDER BY id"):
        ex = json.loads(s["extracted"])
        iss.append([ex.get("file", ""), s["reg_no"] or ex.get("reg_raw", ""), "; ".join(json.loads(s["reasons"]))])
    wb.save(path)
    return Path(path)
