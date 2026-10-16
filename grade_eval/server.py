"""Phone upload + review page.   Run: python -m uvicorn grade_eval.server:app --host 0.0.0.0 --port 8000

Open http://<pc-ip>:8000 on the phone (same Wi-Fi). Env: ROSTER=path/to/class.xlsx|csv to snap register numbers.
"""
import html
import json
import os
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse

from grade_eval.pipeline import GradePipeline
from grade_eval.roster import load_roster
from grade_eval.store import db
from grade_eval.store.excel import export_excel

app = FastAPI()
_pipe: GradePipeline | None = None


def pipe() -> GradePipeline:
    global _pipe
    if _pipe is None:
        r = os.environ.get("ROSTER")
        _pipe = GradePipeline(roster=load_roster(r) if r else None)
    return _pipe


PAGE = """<!doctype html><meta name=viewport content="width=device-width,initial-scale=1"><title>Grade eval</title>
<style>body{{font-family:system-ui;margin:16px;max-width:760px}}input,button{{font-size:16px;margin:4px 0}}
.r{{border:1px solid #ccc;border-radius:8px;padding:8px;margin:8px 0}}.ok{{color:#0a0}}.bad{{color:#c40}}</style>{body}"""


@app.get("/", response_class=HTMLResponse)
def home():
    c = pipe().conn
    n = {s: c.execute("SELECT COUNT(*) FROM sheets WHERE status=?", (s,)).fetchone()[0] for s in ("verified", "needs_review")}
    return PAGE.format(body=f"""<h2>Upload answer sheet</h2>
<form action=/upload method=post enctype=multipart/form-data>
<input type=file name=files accept="image/*" capture=environment multiple><br><button>Upload</button></form>
<p><span class=ok>{n['verified']} verified</span> · <a href=/review><span class=bad>{n['needs_review']} need review</span></a>
 · <a href=/marks.xlsx>Download marks.xlsx</a></p>""")


@app.post("/upload", response_class=HTMLResponse)
async def upload(files: list[UploadFile] = File(...)):
    rows = []
    for f in files:
        r = pipe().process(await f.read(), f.filename)
        why = "<br>".join(html.escape(x) for x in r["reasons"])
        cls = "ok" if r["status"] == "verified" else "bad"
        rows.append(f"<div class=r><b>{html.escape(f.filename or '')}</b> <span class={cls}>{r['status']}</span> "
                    f"{html.escape(r['reg_no'] or r.get('reg_raw', ''))}<br><small>{why}</small></div>")
    return PAGE.format(body="<h2>Result</h2>" + "".join(rows) + "<p><a href=/>Back</a> · <a href=/review>Review</a></p>")


@app.get("/review", response_class=HTMLResponse)
def review():
    out = []
    for s in pipe().conn.execute("SELECT * FROM sheets WHERE status='needs_review' ORDER BY id"):
        ex = json.loads(s["extracted"]); q = ex.get("questions", {})
        fields = "".join(f"Q{i} <input name=q{i} size=4 value='{(q.get(str(i)) or {}).get('a', '') or ''}'> " for i in range(1, 11))
        why = "<br>".join(html.escape(x) for x in json.loads(s["reasons"]))
        out.append(f"""<div class=r><b>#{s['id']} {html.escape(ex.get('file', ''))}</b><br><small class=bad>{why}</small>
<form action=/approve method=post><input type=hidden name=sheet_id value={s['id']}>
Reg <input name=reg_no value='{html.escape(s['reg_no'] or ex.get('reg_raw', ''))}'> Name <input name=name value='{html.escape(ex.get('name', ''))}'><br>
{fields}<br><button>Approve</button></form></div>""")
    return PAGE.format(body="<h2>Needs review</h2><p><a href=/>Back</a></p>" + ("".join(out) or "<p>Nothing to review.</p>"))


@app.post("/approve")
async def approve(sheet_id: int = Form(...), reg_no: str = Form(...), name: str = Form(""),
                  q1: str = Form(""), q2: str = Form(""), q3: str = Form(""), q4: str = Form(""), q5: str = Form(""),
                  q6: str = Form(""), q7: str = Form(""), q8: str = Form(""), q9: str = Form(""), q10: str = Form("")):
    vals = [q1, q2, q3, q4, q5, q6, q7, q8, q9, q10]
    questions = {i + 1: {"a": float(v), "b": None} for i, v in enumerate(vals) if v.strip()}
    db.approve_sheet(pipe().conn, sheet_id, reg_no.strip().upper(), name.strip(), questions)
    return RedirectResponse("/review", status_code=303)


@app.get("/marks.xlsx")
def marks_xlsx():
    path = Path(tempfile.gettempdir()) / "marks.xlsx"
    export_excel(pipe().conn, path)
    return FileResponse(path, filename="marks.xlsx")
