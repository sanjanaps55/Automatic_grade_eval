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
