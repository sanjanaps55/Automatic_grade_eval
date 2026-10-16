"""Web app: sets of papers -> upload photos -> see extracted data -> fix anything held back -> download Excel.

Run: python -m uvicorn grade_eval.server:app --host 0.0.0.0 --port 8000     (open http://<pc-ip>:8000 on the phone)
Env: ROSTER=path/to/class.xlsx|csv  snaps register numbers to the class list.

A "set" is one exam / one pile of papers. Each set has its own database + Excel under data/sets/<name>/, so the same
student can appear in different sets, and starting a new set never touches an old one.
"""
import html
import json
import os
import re
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
SETS = ROOT / "data" / "sets"
ALIGNED = ROOT / "data" / "aligned_web"
app = FastAPI()
_pipe: GradePipeline | None = None
_conns: dict = {}
_lock = threading.RLock()          # one sheet at a time: models and sqlite connections are shared


def slugify(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", name.strip()).strip("_")[:60]


def conn_for(slug: str):
    with _lock:
        if slug not in _conns:
            _conns[slug] = db.connect(SETS / slug / "grades.db")
        return _conns[slug]


def pipe_for(slug: str) -> GradePipeline:
    """Shared models, with the connection pointed at this set's database."""
    global _pipe
    with _lock:
        if _pipe is None:
            r = os.environ.get("ROSTER")
            _pipe = GradePipeline(db_path=":memory:", roster=load_roster(r) if r else None, save_aligned=ALIGNED)
        _pipe.conn = conn_for(slug)
        return _pipe


def list_sets():
    out = []
    for d in sorted(SETS.glob("*/grades.db"), key=lambda p: p.stat().st_mtime, reverse=True):
        c = conn_for(d.parent.name)
        n = {s: c.execute("SELECT COUNT(*) FROM sheets WHERE status=?", (s,)).fetchone()[0] for s in ("verified", "needs_review")}
        out.append((d.parent.name, n))
    return out


CSS = """
:root{--bg:#f6f7f9;--card:#fff;--tx:#1b1f24;--mut:#667;--ok:#0a7d33;--bad:#c2410c;--ln:#dde1e6;--ac:#2563eb}
@media(prefers-color-scheme:dark){:root{--bg:#111418;--card:#1b2027;--tx:#e8eaed;--mut:#9aa4b2;--ok:#4ade80;--bad:#fb923c;--ln:#2d343d;--ac:#60a5fa}}
*{box-sizing:border-box}body{font-family:system-ui,sans-serif;background:var(--bg);color:var(--tx);margin:auto;padding:16px;max-width:1100px}
h1{font-size:22px;margin:8px 0}h2{font-size:17px;margin:0 0 8px}a{color:var(--ac)}
.card{background:var(--card);border:1px solid var(--ln);border-radius:12px;padding:14px;margin:12px 0}
.ok{color:var(--ok)}.bad{color:var(--bad)}.mut{color:var(--mut);font-size:13px}
button,.btn{background:var(--ac);color:#fff;border:0;border-radius:8px;padding:10px 16px;font-size:16px;cursor:pointer;text-decoration:none;display:inline-block}
.btn.alt{background:transparent;color:var(--ac);border:1px solid var(--ac)}
input[type=file]{font-size:16px;max-width:100%}
input[type=text]{font-size:16px;padding:6px;border:1px solid var(--ln);border-radius:6px;background:var(--bg);color:var(--tx)}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(80px,1fr));gap:6px}.grid label{font-size:13px;color:var(--mut)}.grid input{width:100%}
img.prev{width:100%;border-radius:8px;border:1px solid var(--ln)}
.badge{display:inline-block;padding:1px 9px;border-radius:99px;font-size:12px;border:1px solid currentColor;white-space:nowrap}
.tw{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:14px}
td,th{padding:6px 8px;border-bottom:1px solid var(--ln);text-align:left;white-space:nowrap}th{color:var(--mut);font-weight:600}
td.n{text-align:right}.flash{background:var(--card);border:1px solid var(--ok);border-radius:8px;padding:8px 12px;margin:8px 0}
#busy{display:none;position:fixed;inset:0;background:#000b;color:#fff;align-items:center;justify-content:center;font-size:18px;z-index:9}
.row{display:flex;gap:12px;flex-wrap:wrap;align-items:center}
"""


def page(body: str, crumbs: str = "") -> str:
    return (f"<!doctype html><meta charset=utf-8><meta name=viewport content='width=device-width,initial-scale=1'>"
            f"<title>Grade eval</title><style>{CSS}</style>"
            f"<div id=busy><div>Reading sheet(s)… a few seconds each, please wait</div></div>"
            f"<div class=mut><a href=/>All sets</a>{crumbs}</div>{body}"
            f"<script>document.querySelectorAll('form[data-busy]').forEach(f=>f.addEventListener('submit',()=>"
            f"document.getElementById('busy').style.display='flex'))</script>")


def fmt(v):
    return "" if v is None else format(v, "g")


# ------------------------------------------------------------------ home: list of sets + create
@app.get("/", response_class=HTMLResponse)
def home():
    sets = list_sets()
    items = "".join(
        f"<tr><td><a href='/s/{s}'><b>{html.escape(s)}</b></a></td><td class=ok>{n['verified']} verified</td>"
        f"<td class=bad>{n['needs_review']} need review</td><td><a class='btn alt' href='/s/{s}'>Open</a></td></tr>"
        for s, n in sets) or "<tr><td class=mut>No sets yet. Create one below.</td></tr>"
    return page(f"""<h1>Grade evaluation</h1>
<div class=card><h2>Start a new set of papers</h2>
<form action=/sets method=post class=row>
<input type=text name=name placeholder="e.g. BCSE209L CAT-1 slot B2" required size=32>
<button>Create set</button></form>
<p class=mut>One set = one exam / one pile of answer sheets. Each set has its own results and its own Excel file.</p></div>
<div class=card><h2>Your sets</h2><div class=tw><table>{items}</table></div></div>""")


@app.post("/sets")
def create_set(name: str = Form(...)):
    slug = slugify(name) or "set"
    conn_for(slug)                                   # creates data/sets/<slug>/grades.db
    return RedirectResponse(f"/s/{slug}", status_code=303)


# ------------------------------------------------------------------ one set: upload + results table
def sheet_rows(slug):
    c = conn_for(slug)
    rows = []
    for s in c.execute("SELECT * FROM sheets ORDER BY id DESC").fetchall():
        ex = json.loads(s["extracted"])
        if s["status"] == "verified":                # canonical (possibly human-corrected) values live in marks
            qs = {r["q_no"]: r["total"] for r in c.execute("SELECT q_no, total FROM marks WHERE reg_no=?", (s["reg_no"],))}
            st = c.execute("SELECT name FROM students WHERE reg_no=?", (s["reg_no"],)).fetchone()
            name = st["name"] if st else ex.get("name", "")
        else:
            qs = {int(q): v["a"] for q, v in ex.get("questions", {}).items()}
            name = ex.get("name", "")
        rows.append(dict(id=s["id"], hash=s["image_hash"], status=s["status"], file=ex.get("file", ""),
                         reg=s["reg_no"] or ex.get("reg_raw", ""), name=name, qs=qs, sub=ex.get("sub_total"),
                         reasons=json.loads(s["reasons"])))
    return rows


@app.get("/s/{slug}", response_class=HTMLResponse)
def set_page(slug: str, added: int = 0, dup: int = 0, err: int = 0, filter: str = ""):
    if not (SETS / slug / "grades.db").exists():
        return RedirectResponse("/")
    rows = sheet_rows(slug)
    nv = sum(r["status"] == "verified" for r in rows); nr = len(rows) - nv
    flash = ""
    if added or dup or err:
        flash = f"<div class=flash>Processed: {added} new, {dup} already in this set, {err} failed.</div>"
    shown = [r for r in rows if not filter or r["status"] == filter]
    head = "".join(f"<th>Q{i}</th>" for i in range(1, 11))
    body = ""
    for r in shown:
        ok = r["status"] == "verified"
        badge = f"<span class='badge {'ok' if ok else 'bad'}'>{'verified' if ok else 'needs review'}</span>"
        cells = "".join(f"<td class=n>{fmt(r['qs'].get(i))}</td>" for i in range(1, 11))
        total = sum(r["qs"].values()) if (ok and r["qs"]) else None
        why = "; ".join(r["reasons"])[:140]
        body += (f"<tr><td>{badge}</td><td><a href='/s/{slug}/sheet/{r['id']}'>{html.escape(r['reg'] or '?')}</a></td>"
                 f"<td>{html.escape(r['name'])}</td>{cells}<td class=n><b>{fmt(total)}</b></td>"
                 f"<td class=n>{fmt(r['sub'])}</td><td class=mut title='{html.escape('; '.join(r['reasons']), quote=True)}'>"
                 f"{html.escape(why) if not ok else ''}</td>"
                 f"<td><a href='/s/{slug}/sheet/{r['id']}'>{'view' if ok else 'fix'}</a></td></tr>")
    table = (f"<div class=tw><table><tr><th>Status</th><th>Reg no</th><th>Name</th>{head}<th>Sum</th><th>Written sub</th>"
             f"<th>Why held</th><th></th></tr>{body}</table></div>") if shown else "<p class=mut>No sheets yet. Upload photos above.</p>"
    return page(f"""<h1>{html.escape(slug)}</h1>{flash}
<div class=card><h2>Add answer sheets to this set</h2>
<form action='/s/{slug}/upload' method=post enctype=multipart/form-data data-busy>
<p><input type=file name=files accept="image/*" multiple required></p>
<button>Upload and read</button></form>
<p class=mut>Pick one or many photos (on a phone: camera or gallery). Each is aligned, read, checked and added to the table below.</p></div>
<div class=card><div class=row><span class=ok>{nv} verified</span><span class=bad>{nr} need review</span>
<a class=btn href='/s/{slug}/marks.xlsx'>Download Excel</a>
<a href='/s/{slug}'>all</a><a href='/s/{slug}?filter=needs_review'>only review</a><a href='/s/{slug}?filter=verified'>only verified</a></div>
<p class=mut>Excel contains verified sheets only. Fix a held-back sheet (click its reg no) to include it.</p>{table}</div>""",
                f" &rsaquo; {html.escape(slug)}")


@app.post("/s/{slug}/upload")
def upload(slug: str, files: list[UploadFile] = File(...)):
    p = pipe_for(slug)
    added = dup = err = 0
    for f in files:
        data = f.file.read()
        try:
            with _lock:
                p.conn = conn_for(slug)
                r = p.process(data, f.filename or "")
            if r["status"] == "duplicate":
                dup += 1
            else:
                added += 1
        except Exception:
            err += 1
    return RedirectResponse(f"/s/{slug}?added={added}&dup={dup}&err={err}", status_code=303)


# ------------------------------------------------------------------ one sheet: image + editable values
@app.get("/s/{slug}/sheet/{sheet_id}", response_class=HTMLResponse)
def sheet_page(slug: str, sheet_id: int):
    r = next((x for x in sheet_rows(slug) if x["id"] == sheet_id), None)
    if r is None:
        return RedirectResponse(f"/s/{slug}")
    fields = "".join(f"<div><label>Q{i}</label><input type=text inputmode=decimal name=q{i} value='{fmt(r['qs'].get(i))}'></div>"
                     for i in range(1, 11))
    why = "".join(f"<li>{html.escape(x)}</li>" for x in r["reasons"])
    im = f"<img class=prev src='/img/{r['hash']}'>" if (ALIGNED / (r["hash"] + ".png")).exists() else "<p class=mut>No preview.</p>"
    ok = r["status"] == "verified"
    return page(f"""<h1>{html.escape(r['reg'] or 'Unknown')} <span class='badge {'ok' if ok else 'bad'}'>{'verified' if ok else 'needs review'}</span></h1>
<p class=mut>{html.escape(r['file'])}</p>
{('<div class=card><b class=bad>Why this was held back</b><ul>' + why + '</ul></div>') if why else ''}
<div class=card><h2>Correct values, then save</h2>
<form action='/s/{slug}/approve' method=post><input type=hidden name=sheet_id value={sheet_id}>
<p>Reg no <input type=text name=reg_no value='{html.escape(r['reg'])}'>
Name <input type=text name=name value='{html.escape(r['name'])}'></p>
<div class=grid>{fields}</div><p><button>Save as verified</button></p></form>
<p class=mut>Leave a box empty if that question has no marks.</p></div>
<div class=card><h2>Sheet (aligned)</h2>{im}</div>""", f" &rsaquo; <a href='/s/{slug}'>{html.escape(slug)}</a>")


@app.post("/s/{slug}/approve")
def approve(slug: str, sheet_id: int = Form(...), reg_no: str = Form(...), name: str = Form(""),
            q1: str = Form(""), q2: str = Form(""), q3: str = Form(""), q4: str = Form(""), q5: str = Form(""),
            q6: str = Form(""), q7: str = Form(""), q8: str = Form(""), q9: str = Form(""), q10: str = Form("")):
    vals = [q1, q2, q3, q4, q5, q6, q7, q8, q9, q10]
    questions = {i + 1: {"a": float(v), "b": None} for i, v in enumerate(vals) if v.strip()}
    with _lock:
        db.approve_sheet(conn_for(slug), sheet_id, reg_no.strip().upper(), name.strip(), questions)
    return RedirectResponse(f"/s/{slug}", status_code=303)


@app.get("/img/{name}")
def img(name: str):
    path = ALIGNED / (Path(name).stem + ".png")
    return FileResponse(path) if path.exists() else Response(status_code=404)


@app.get("/s/{slug}/marks.xlsx")
def marks_xlsx(slug: str):
    path = Path(tempfile.gettempdir()) / f"marks_{slug}.xlsx"
    with _lock:
        export_excel(conn_for(slug), path)
    return FileResponse(path, filename=f"{slug}.xlsx")
