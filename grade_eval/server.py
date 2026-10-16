"""Web app for a teacher: new set (class list + number of questions) -> upload photos -> check -> Excel in class-list order.

Run: python -m uvicorn grade_eval.server:app --host 0.0.0.0 --port 8000     (open http://<pc-ip>:8000 on the phone)

A "set" is one exam / one pile of answer sheets. Each set lives in data/sets/<name>/ (grades.db, roster file, meta.json),
so the same student can be in different sets and starting a new set never touches an old one.

Sheet tiers: confirmed (cross-checked, written automatically) | to check (read fine, glance at it, one click) |
needs fix (something unclear, correct it by hand). Only confirmed and approved sheets reach the Excel.
"""
import html
import json
import re
import tempfile
import threading
from pathlib import Path

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse, Response

from grade_eval.exam_config import ExamConfig, QuestionLimit
from grade_eval.pipeline import CONFIRM, GradePipeline
from grade_eval.roster import load_roster
from grade_eval.store import db
from grade_eval.store.excel import export_roster_excel

ROOT = Path(__file__).resolve().parents[1]
SETS = ROOT / "data" / "sets"
ALIGNED = ROOT / "data" / "aligned_web"
app = FastAPI()
_pipe: GradePipeline | None = None
_conns: dict = {}
_rosters: dict = {}
_lock = threading.RLock()          # one sheet at a time: models and sqlite connections are shared


def slugify(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", name.strip()).strip("_")[:60]


def conn_for(slug: str):
    with _lock:
        if slug not in _conns:
            _conns[slug] = db.connect(SETS / slug / "grades.db")
        return _conns[slug]


def meta_of(slug: str) -> dict:
    p = SETS / slug / "meta.json"
    m = {"n_questions": 10, "max_mark": 100, "roster_file": ""}
    if p.exists():
        m.update(json.loads(p.read_text(encoding="utf-8")))
    return m


def save_meta(slug: str, meta: dict):
    (SETS / slug).mkdir(parents=True, exist_ok=True)
    (SETS / slug / "meta.json").write_text(json.dumps(meta), encoding="utf-8")


def roster_of(slug: str) -> list:
    with _lock:
        if slug not in _rosters:
            f = meta_of(slug).get("roster_file")
            _rosters[slug] = load_roster(SETS / slug / f) if f and (SETS / slug / f).exists() else []
        return _rosters[slug]


def pipe_for(slug: str) -> GradePipeline:
    """Shared models, pointed at this set's database, class list and question count."""
    global _pipe
    with _lock:
        meta = meta_of(slug)
        if _pipe is None:
            _pipe = GradePipeline(db_path=":memory:", save_aligned=ALIGNED)
        _pipe.conn = conn_for(slug)
        _pipe.roster = roster_of(slug)
        _pipe.n_questions = meta["n_questions"]
        _pipe.config = ExamConfig(slug, None, {q: QuestionLimit(max=meta["max_mark"]) for q in range(1, meta["n_questions"] + 1)})
        return _pipe


def list_sets():
    out = []
    for d in sorted(SETS.glob("*/grades.db"), key=lambda p: p.stat().st_mtime, reverse=True):
        rows = sheet_rows(d.parent.name)
        out.append((d.parent.name, sum(r["tier"] == "confirmed" for r in rows), sum(r["tier"] != "confirmed" for r in rows)))
    return out


CSS = """
:root{--bg:#f6f7f9;--card:#fff;--tx:#1b1f24;--mut:#667;--ok:#0a7d33;--bad:#c2410c;--chk:#a16207;--ln:#dde1e6;--ac:#2563eb}
@media(prefers-color-scheme:dark){:root{--bg:#111418;--card:#1b2027;--tx:#e8eaed;--mut:#9aa4b2;--ok:#4ade80;--bad:#fb923c;--chk:#facc15;--ln:#2d343d;--ac:#60a5fa}}
*{box-sizing:border-box}body{font-family:system-ui,sans-serif;background:var(--bg);color:var(--tx);margin:auto;padding:16px;max-width:1100px}
h1{font-size:22px;margin:8px 0}h2{font-size:17px;margin:0 0 8px}a{color:var(--ac)}
.card{background:var(--card);border:1px solid var(--ln);border-radius:12px;padding:14px;margin:12px 0}
.ok{color:var(--ok)}.bad{color:var(--bad)}.chk{color:var(--chk)}.mut{color:var(--mut);font-size:13px}
button,.btn{background:var(--ac);color:#fff;border:0;border-radius:8px;padding:10px 16px;font-size:16px;cursor:pointer;text-decoration:none;display:inline-block}
.btn.alt{background:transparent;color:var(--ac);border:1px solid var(--ac)}
input[type=file]{font-size:16px;max-width:100%}
input[type=text],input[type=number]{font-size:16px;padding:6px;border:1px solid var(--ln);border-radius:6px;background:var(--bg);color:var(--tx)}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(80px,1fr));gap:6px}.grid label{font-size:13px;color:var(--mut)}.grid input{width:100%}
img.prev{width:100%;border-radius:8px;border:1px solid var(--ln)}
.badge{display:inline-block;padding:1px 9px;border-radius:99px;font-size:12px;border:1px solid currentColor;white-space:nowrap}
.tw{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:14px}
td,th{padding:6px 8px;border-bottom:1px solid var(--ln);text-align:left;white-space:nowrap}th{color:var(--mut);font-weight:600}
td.n{text-align:right}.flash{background:var(--card);border:1px solid var(--ok);border-radius:8px;padding:8px 12px;margin:8px 0}
#busy{display:none;position:fixed;inset:0;background:#000b;color:#fff;align-items:center;justify-content:center;font-size:18px;z-index:9}
.row{display:flex;gap:12px;flex-wrap:wrap;align-items:center}.bar{height:10px;background:var(--ln);border-radius:6px;overflow:hidden}
.bar i{display:block;height:100%;background:var(--ok)}
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
    items = "".join(
        f"<tr><td><a href='/s/{s}'><b>{html.escape(s)}</b></a></td><td class=ok>{a} confirmed</td>"
        f"<td class=chk>{b} to check / fix</td><td><a class='btn alt' href='/s/{s}'>Open</a></td></tr>"
        for s, a, b in list_sets()) or "<tr><td class=mut>No sets yet. Create one above.</td></tr>"
    return page(f"""<h1>Grade evaluation</h1>
<div class=card><h2>Start a new set of papers</h2>
<form action=/sets method=post enctype=multipart/form-data>
<p>Name of this set<br><input type=text name=name placeholder="e.g. BCSE209L CAT-1 slot B2" required size=36></p>
<p>Class list (Excel or CSV with register number and name columns)<br><input type=file name=roster accept=".xlsx,.csv"></p>
<p>Number of questions in the paper<br><input type=number name=n_questions value=10 min=1 max=40 style="width:90px"></p>
<p>Maximum marks for any one question (optional safety check)<br><input type=number name=max_mark value=100 min=1 style="width:90px"></p>
<button>Create set</button></form>
<p class=mut>The class list lets the app match each sheet to a student and fill the Excel in your order.</p></div>
<div class=card><h2>Your sets</h2><div class=tw><table>{items}</table></div></div>""")


def store_roster(slug: str, upload: UploadFile | None) -> str:
    """Save an uploaded class list into the set folder, validate it, return a message ('' if none)."""
    if upload is None or not upload.filename:
        return ""
    ext = Path(upload.filename).suffix.lower()
    if ext not in (".xlsx", ".csv"):
        return "Class list must be .xlsx or .csv"
    (SETS / slug).mkdir(parents=True, exist_ok=True)
    fname = "roster" + ext
    (SETS / slug / fname).write_bytes(upload.file.read())
    try:
        r = load_roster(SETS / slug / fname)
    except Exception as e:
        (SETS / slug / fname).unlink(missing_ok=True)
        return f"Could not read class list: {e}"
    with _lock:
        _rosters[slug] = r
        m = meta_of(slug); m["roster_file"] = fname; save_meta(slug, m)
    return f"Class list loaded: {len(r)} students."


@app.post("/sets")
def create_set(name: str = Form(...), n_questions: int = Form(10), max_mark: float = Form(100),
               roster: UploadFile | None = File(None)):
    slug = slugify(name) or "set"
    save_meta(slug, {"n_questions": max(1, min(40, n_questions)), "max_mark": max_mark, "roster_file": ""})
    conn_for(slug)
    msg = store_roster(slug, roster)
    return RedirectResponse(f"/s/{slug}?msg={html.escape(msg)}", status_code=303)


@app.post("/s/{slug}/roster")
def replace_roster(slug: str, roster: UploadFile = File(...)):
    msg = store_roster(slug, roster)
    return RedirectResponse(f"/s/{slug}?msg={html.escape(msg)}", status_code=303)


# ------------------------------------------------------------------ one set: upload + results table
def tier_of(status: str, reasons: list[str]) -> str:
    if status == "verified":
        return "confirmed"
    return "check" if reasons and all(r.startswith(CONFIRM) for r in reasons) else "fix"


def sheet_rows(slug):
    c = conn_for(slug)
    rows = []
    for s in c.execute("SELECT * FROM sheets ORDER BY id DESC").fetchall():
        ex = json.loads(s["extracted"])
        reasons = json.loads(s["reasons"])
        if s["status"] == "verified":                # canonical (possibly human-corrected) values live in marks
            qs = {r["q_no"]: r["total"] for r in c.execute("SELECT q_no, total FROM marks WHERE reg_no=?", (s["reg_no"],))}
            st = c.execute("SELECT name FROM students WHERE reg_no=?", (s["reg_no"],)).fetchone()
            name = st["name"] if st else ex.get("name", "")
        else:
            qs = {int(q): v["a"] for q, v in ex.get("questions", {}).items()}
            name = ex.get("name", "")
        rows.append(dict(id=s["id"], hash=s["image_hash"], status=s["status"], tier=tier_of(s["status"], reasons),
                         file=ex.get("file", ""), reg=s["reg_no"] or ex.get("reg_raw", ""), name=name, qs=qs,
                         sub=ex.get("sub_total"), reasons=reasons, raw={int(q): v.get("raw", "") for q, v in ex.get("questions", {}).items()}))
    return rows


TIER = {"confirmed": ("ok", "confirmed"), "check": ("chk", "to check"), "fix": ("bad", "needs fix")}


@app.get("/s/{slug}", response_class=HTMLResponse)
def set_page(slug: str, added: int = 0, dup: int = 0, err: int = 0, filter: str = "", msg: str = ""):
    if not (SETS / slug / "grades.db").exists():
        return RedirectResponse("/")
    meta, roster = meta_of(slug), roster_of(slug)
    rows = sheet_rows(slug)
    n = meta["n_questions"]
    cnt = {t: sum(r["tier"] == t for r in rows) for t in TIER}
    on_list = {x.reg_no for x in roster}
    done = {r["reg"] for r in rows if r["tier"] == "confirmed"} & on_list
    flash = ""
    if msg:
        flash += f"<div class=flash>{html.escape(msg)}</div>"
    if added or dup or err:
        flash += f"<div class=flash>Processed: {added} new, {dup} already in this set, {err} failed.</div>"
    progress = ""
    if roster:
        pct = 100 * len(done) // len(roster)
        progress = (f"<p><b>{len(done)} of {len(roster)}</b> students on the class list are confirmed"
                    f"<div class=bar><i style='width:{pct}%'></i></div></p>")
    shown = [r for r in rows if not filter or r["tier"] == filter]
    head = "".join(f"<th>Q{i}</th>" for i in range(1, n + 1))
    body = ""
    for r in shown:
        cls, label = TIER[r["tier"]]
        cells = "".join(f"<td class=n title='{html.escape(r['raw'].get(i, ''), quote=True)}'>{fmt(r['qs'].get(i))}</td>"
                        for i in range(1, n + 1))
        total = sum(r["qs"].values()) if r["qs"] else None
        why = "; ".join(x.replace(CONFIRM, "") for x in r["reasons"])
        body += (f"<tr><td><span class='badge {cls}'>{label}</span></td>"
                 f"<td><a href='/s/{slug}/sheet/{r['id']}'>{html.escape(r['reg'] or '?')}</a></td>"
                 f"<td>{html.escape(r['name'])}</td>{cells}<td class=n><b>{fmt(total)}</b></td>"
                 f"<td class=mut title='{html.escape(why, quote=True)}'>{html.escape(why[:90])}</td>"
                 f"<td><a href='/s/{slug}/sheet/{r['id']}'>{'view' if r['tier'] == 'confirmed' else 'open'}</a></td></tr>")
    table = (f"<div class=tw><table><tr><th>Status</th><th>Reg no</th><th>Name</th>{head}<th>Total</th>"
             f"<th>Note</th><th></th></tr>{body}</table></div>") if shown else "<p class=mut>No sheets yet. Upload photos above.</p>"
    roster_line = (f"{len(roster)} students" if roster else "none (register numbers will not be checked)")
    bulk = (f"<form action='/s/{slug}/confirm_all' method=post onsubmit=\"return confirm('Confirm all {cnt['check']} sheets in \\'to check\\' "
            f"as correct without opening them?')\"><button class=btn style='background:var(--chk)'>Confirm all {cnt['check']} “to check”</button></form>"
            ) if cnt["check"] else ""
    return page(f"""<h1>{html.escape(slug)}</h1>{flash}
<div class=card><div class=mut>{n} questions &middot; class list: {roster_line}</div>{progress}
<h2>Add answer sheets</h2>
<form action='/s/{slug}/upload' method=post enctype=multipart/form-data data-busy>
<p><input type=file name=files accept="image/*" multiple required></p>
<button>Upload and read</button></form>
<p class=mut>Pick one or many photos (phone: camera or gallery). Each is aligned, read, matched to the class list and added below.</p></div>
<div class=card><div class=row><span class=ok>{cnt['confirmed']} confirmed</span><span class=chk>{cnt['check']} to check</span>
<span class=bad>{cnt['fix']} need fix</span><a class=btn href='/s/{slug}/marks.xlsx'>Download Excel</a>{bulk}</div>
<div class=row mut><a href='/s/{slug}'>all</a><a href='/s/{slug}?filter=check'>to check</a><a href='/s/{slug}?filter=fix'>need fix</a>
<a href='/s/{slug}?filter=confirmed'>confirmed</a></div>
<p class=mut><b>confirmed</b>: cross-checked, in the Excel. <b>to check</b>: read fine, open it, glance, press save (or confirm all).
<b>need fix</b>: something unclear, correct it by hand. Hover a mark to see the raw read (e.g. 3+5+1).</p>{table}</div>
<div class=card><h2>Class list</h2><form action='/s/{slug}/roster' method=post enctype=multipart/form-data>
<input type=file name=roster accept=".xlsx,.csv" required> <button class=alt style='background:transparent;color:var(--ac);border:1px solid var(--ac)'>Upload / replace</button></form></div>""",
                f" &rsaquo; {html.escape(slug)}")


@app.post("/s/{slug}/upload")
def upload(slug: str, files: list[UploadFile] = File(...)):
    added = dup = err = 0
    for f in files:
        data = f.file.read()
        try:
            with _lock:
                p = pipe_for(slug)
                r = p.process(data, f.filename or "")
            dup += r["status"] == "duplicate"
            added += r["status"] != "duplicate"
        except Exception:
            err += 1
    return RedirectResponse(f"/s/{slug}?added={added}&dup={dup}&err={err}", status_code=303)


@app.post("/s/{slug}/confirm_all")
def confirm_all(slug: str):
    with _lock:
        c = conn_for(slug)
        for r in sheet_rows(slug):
            if r["tier"] == "check" and r["reg"]:
                db.approve_sheet(c, r["id"], r["reg"], r["name"], {q: {"a": v, "b": None} for q, v in r["qs"].items()})
    return RedirectResponse(f"/s/{slug}", status_code=303)


# ------------------------------------------------------------------ one sheet: image + editable values
@app.get("/s/{slug}/sheet/{sheet_id}", response_class=HTMLResponse)
def sheet_page(slug: str, sheet_id: int):
    r = next((x for x in sheet_rows(slug) if x["id"] == sheet_id), None)
    if r is None:
        return RedirectResponse(f"/s/{slug}")
    n = meta_of(slug)["n_questions"]
    roster = roster_of(slug)
    fields = "".join(f"<div><label>Q{i}</label><input type=text inputmode=decimal name=q{i} value='{fmt(r['qs'].get(i))}'>"
                     f"<span class=mut>{html.escape(r['raw'].get(i, '')) if '+' in r['raw'].get(i, '') else ''}</span></div>"
                     for i in range(1, n + 1))
    why = "".join(f"<li>{html.escape(x)}</li>" for x in r["reasons"])
    im = f"<img class=prev src='/img/{r['hash']}'>" if (ALIGNED / (r["hash"] + ".png")).exists() else "<p class=mut>No preview.</p>"
    cls, label = TIER[r["tier"]]
    opts = "".join(f"<option value='{html.escape(x.reg_no)}'>{html.escape(x.name)}</option>" for x in roster)
    return page(f"""<h1>{html.escape(r['reg'] or 'Unknown')} <span class='badge {cls}'>{label}</span></h1>
<p class=mut>{html.escape(r['file'])}</p>
{('<div class=card><b>Notes</b><ul>' + why + '</ul></div>') if why else ''}
<div class=card><h2>Check the values against the sheet, then save</h2>
<form action='/s/{slug}/approve' method=post><input type=hidden name=sheet_id value={sheet_id}>
<p>Reg no <input type=text name=reg_no list=regs value='{html.escape(r['reg'])}'><datalist id=regs>{opts}</datalist>
Name <input type=text name=name value='{html.escape(r['name'])}'></p>
<div class=grid>{fields}</div><p><button>Save as confirmed</button></p></form>
<p class=mut>Leave a box empty if that question has no marks. If the reg no is on the class list, the name is filled from it.</p></div>
<div class=card><h2>Sheet (aligned)</h2>{im}</div>""", f" &rsaquo; <a href='/s/{slug}'>{html.escape(slug)}</a>")


@app.post("/s/{slug}/approve")
async def approve(slug: str, request: Request):
    form = await request.form()
    sheet_id = int(form["sheet_id"])
    reg = str(form.get("reg_no", "")).strip().upper()
    name = str(form.get("name", "")).strip()
    for x in roster_of(slug):
        if x.reg_no == reg and not name:
            name = x.name
    questions = {}
    for k, v in form.items():
        m = re.fullmatch(r"q(\d+)", k)
        if m and str(v).strip():
            try:
                questions[int(m.group(1))] = {"a": float(str(v).strip()), "b": None}
            except ValueError:
                pass
    with _lock:
        db.approve_sheet(conn_for(slug), sheet_id, reg, name, questions)
    return RedirectResponse(f"/s/{slug}", status_code=303)


@app.get("/img/{name}")
def img(name: str):
    path = ALIGNED / (Path(name).stem + ".png")
    return FileResponse(path) if path.exists() else Response(status_code=404)


@app.get("/s/{slug}/marks.xlsx")
def marks_xlsx(slug: str):
    path = Path(tempfile.gettempdir()) / f"marks_{slug}.xlsx"
    with _lock:
        export_roster_excel(conn_for(slug), roster_of(slug), meta_of(slug)["n_questions"], path)
    return FileResponse(path, filename=f"{slug}.xlsx")
