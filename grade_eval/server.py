"""Phone upload + review page.   Run: python -m uvicorn grade_eval.server:app --host 0.0.0.0 --port 8000

Open http://<pc-ip>:8000 on the phone (same Wi-Fi). Env: ROSTER=path/to/class.xlsx|csv to snap register numbers.
"""
import html
import json
import os
import tempfile
import threading
from pathlib import Path

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse, Response

from grade_eval.pipeline import GradePipeline
from grade_eval.roster import load_roster
from grade_eval.store import db
from grade_eval.store.excel import export_excel

ROOT = Path(__file__).resolve().parents[1]
ALIGNED = ROOT / "data" / "aligned_web"
app = FastAPI()
_pipe: GradePipeline | None = None
_lock = threading.RLock()          # one sheet at a time: models and the sqlite connection are shared


def pipe() -> GradePipeline:
    global _pipe
    with _lock:
        if _pipe is None:
            r = os.environ.get("ROSTER")
            _pipe = GradePipeline(roster=load_roster(r) if r else None, save_aligned=ALIGNED)
        return _pipe


CSS = """
:root{--bg:#f6f7f9;--card:#fff;--tx:#1b1f24;--mut:#667;--ok:#0a7d33;--bad:#c2410c;--ln:#dde1e6;--ac:#2563eb}
@media(prefers-color-scheme:dark){:root{--bg:#111418;--card:#1b2027;--tx:#e8eaed;--mut:#9aa4b2;--ok:#4ade80;--bad:#fb923c;--ln:#2d343d;--ac:#60a5fa}}
*{box-sizing:border-box}body{font-family:system-ui,sans-serif;background:var(--bg);color:var(--tx);margin:0;padding:16px;max-width:820px;margin:auto}
h1{font-size:20px}a{color:var(--ac)}.card{background:var(--card);border:1px solid var(--ln);border-radius:12px;padding:14px;margin:12px 0}
.ok{color:var(--ok)}.bad{color:var(--bad)}.mut{color:var(--mut);font-size:13px}
button,.btn{background:var(--ac);color:#fff;border:0;border-radius:8px;padding:10px 16px;font-size:16px;cursor:pointer}
input[type=file]{font-size:16px;max-width:100%}input[type=text]{font-size:16px;padding:6px;border:1px solid var(--ln);border-radius:6px;background:var(--bg);color:var(--tx)}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(88px,1fr));gap:6px}.grid label{font-size:13px;color:var(--mut)}
.grid input{width:100%}img.prev{width:100%;border-radius:8px;border:1px solid var(--ln)}
.badge{display:inline-block;padding:2px 10px;border-radius:99px;font-size:13px;border:1px solid currentColor}
#busy{display:none;position:fixed;inset:0;background:#000a;color:#fff;align-items:center;justify-content:center;font-size:18px;z-index:9}
nav a{margin-right:14px}table{border-collapse:collapse}td,th{padding:4px 10px;border-bottom:1px solid var(--ln);text-align:left}
"""


def page(body: str) -> str:
    return (f"<!doctype html><meta charset=utf-8><meta name=viewport content='width=device-width,initial-scale=1'>"
            f"<title>Grade eval</title><style>{CSS}</style>"
            f"<div id=busy><div>Reading sheet(s)… this takes a few seconds each</div></div>"
            f"<nav><a href=/>Upload</a><a href=/review>Review</a><a href=/marks.xlsx>Download Excel</a></nav>{body}"
            f"<script>document.querySelectorAll('form[data-busy]').forEach(f=>f.addEventListener('submit',()=>"
            f"document.getElementById('busy').style.display='flex'))</script>")


def counts() -> dict:
    c = db.connect(db.DEFAULT_DB)
    try:
        return {s: c.execute("SELECT COUNT(*) FROM sheets WHERE status=?", (s,)).fetchone()[0] for s in ("verified", "needs_review")}
    finally:
        c.close()


@app.get("/", response_class=HTMLResponse)
def home():
    n = counts()
    return page(f"""<h1>Upload answer sheets</h1>
<div class=card><form action=/upload method=post enctype=multipart/form-data data-busy>
<p><input type=file name=files accept="image/*" multiple required></p>
<button>Upload and read</button></form>
<p class=mut>On a phone this opens the camera or gallery. Pick one or many photos.</p></div>
<div class=card><span class=ok>{n['verified']} verified</span> &middot; <a href=/review class=bad>{n['needs_review']} need review</a></div>""")


def result_card(r: dict) -> str:
    st = r["status"]
    cls = "ok" if st == "verified" else "bad"
    label = {"verified": "verified", "needs_review": "needs review", "duplicate": "already processed"}[st]
    if st == "duplicate":
        label += f" (was {r.get('previous_status', '?').replace('_', ' ')})"
    q = r.get("questions", {})
    rows = "".join(f"<tr><td>Q{i}</td><td>{q[i]['a']:g}</td></tr>" for i in sorted(q))
    img = f"<img class=prev src='/img/{r['image']}'>" if r.get("image") else ""
    why = "".join(f"<li>{html.escape(x)}</li>" for x in r["reasons"])
    return f"""<div class=card><b>{html.escape(r['file'] or '')}</b> <span class='badge {cls}'>{label}</span>
<p>Register no: <b>{html.escape(r['reg_no'] or r.get('reg_raw', '') or '?')}</b>
{'&middot; ' + html.escape(r['name']) if r.get('name') else ''}
{'&middot; sub-total ' + format(r['sub_total'], 'g') if r.get('sub_total') is not None else ''}</p>
{('<table>' + rows + '</table>') if rows else '<p class=mut>No marks read automatically.</p>'}
{('<p class=bad>Held for review:</p><ul>' + why + '</ul>') if why else ''}{img}</div>"""


@app.post("/upload", response_class=HTMLResponse)
def upload(files: list[UploadFile] = File(...)):
    p = pipe()
    cards = []
    for f in files:
        data = f.file.read()
        try:
            with _lock:
                r = p.process(data, f.filename or "")
            cards.append(result_card(r))
        except Exception as e:                                 # a bad photo must not kill the batch
            cards.append(f"<div class=card><b>{html.escape(f.filename or '')}</b> <span class='badge bad'>error</span>"
                         f"<p class=bad>{html.escape(repr(e))}</p></div>")
    return page("<h1>Results</h1>" + "".join(cards) + "<p><a class=btn href=/>Upload more</a> <a href=/review>Go to review</a></p>")


@app.get("/img/{name}")
def img(name: str):
    path = ALIGNED / (Path(name).stem + ".png")
    return FileResponse(path) if path.exists() else Response(status_code=404)


@app.get("/review", response_class=HTMLResponse)
def review():
    p = pipe()
    out = []
    with _lock:
        sheets = p.conn.execute("SELECT * FROM sheets WHERE status='needs_review' ORDER BY id").fetchall()
    for s in sheets:
        ex = json.loads(s["extracted"]); q = ex.get("questions", {})
        fields = "".join(f"<div><label>Q{i}</label><input type=text inputmode=decimal name=q{i} "
                         f"value='{(q.get(str(i)) or {}).get('a', '') if (q.get(str(i)) or {}).get('a') is not None else ''}'></div>"
                         for i in range(1, 11))
        why = "".join(f"<li>{html.escape(x)}</li>" for x in json.loads(s["reasons"]))
        im = f"<img class=prev src='/img/{s['image_hash']}'>" if (ALIGNED / (s["image_hash"] + ".png")).exists() else ""
        out.append(f"""<div class=card><b>#{s['id']} {html.escape(ex.get('file', ''))}</b>
<ul class=bad>{why}</ul>{im}
<form action=/approve method=post><input type=hidden name=sheet_id value={s['id']}>
<p>Reg no <input type=text name=reg_no value='{html.escape(s['reg_no'] or ex.get('reg_raw', ''))}'>
Name <input type=text name=name value='{html.escape(ex.get('name', ''))}'></p>
<div class=grid>{fields}</div><p><button>Approve with these values</button></p></form></div>""")
    return page("<h1>Needs review</h1>" + ("".join(out) or "<p>Nothing to review.</p>"))


@app.post("/approve")
def approve(sheet_id: int = Form(...), reg_no: str = Form(...), name: str = Form(""),
            q1: str = Form(""), q2: str = Form(""), q3: str = Form(""), q4: str = Form(""), q5: str = Form(""),
            q6: str = Form(""), q7: str = Form(""), q8: str = Form(""), q9: str = Form(""), q10: str = Form("")):
    vals = [q1, q2, q3, q4, q5, q6, q7, q8, q9, q10]
    questions = {i + 1: {"a": float(v), "b": None} for i, v in enumerate(vals) if v.strip()}
    with _lock:
        db.approve_sheet(pipe().conn, sheet_id, reg_no.strip().upper(), name.strip(), questions)
    return RedirectResponse("/review", status_code=303)


@app.get("/marks.xlsx")
def marks_xlsx():
    path = Path(tempfile.gettempdir()) / "marks.xlsx"
    with _lock:
        export_excel(pipe().conn, path)
    return FileResponse(path, filename="marks.xlsx")
