"""Web app for a teacher: new set (class list + number of questions) -> upload photos -> check -> Excel in class-list order.

Run: python -m uvicorn grade_eval.server:app --host 0.0.0.0 --port 8000     (open http://<pc-ip>:8000 on the phone)

A "set" is one exam / one pile of answer sheets. Each set lives in data/sets/<name>/ (grades.db, roster file, meta.json),
so the same student can be in different sets and starting a new set never touches an old one.

Sheet tiers: confirmed (cross-checked, written automatically) | to check (read fine, glance at it, one click) |
needs fix (something unclear, correct it by hand). Only confirmed and approved sheets reach the Excel.
"""
import csv
import html
import json
import re
import tempfile
import time
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse, Response

from starlette.concurrency import run_in_threadpool

from grade_eval import phone
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


TRUTH_CSV = ROOT / "data" / "labels_all.csv"
_truth: dict | None = None


def truth() -> dict:
    """Hand labels by photo file name (sheet_NNN.jpeg) so a test run can be scored; {} when the file is absent."""
    global _truth
    if _truth is None:
        _truth = {}
        if TRUTH_CSV.exists():
            for r in csv.DictReader(open(TRUTH_CSV, encoding="utf-8-sig")):
                _truth[r["file"].strip().lower()] = {
                    "reg": r["reg_no"].strip().upper(),
                    "qs": {q: float(r[f"q{q}"]) for q in range(1, 11) if (r.get(f"q{q}") or "").strip()}}
    return _truth


def score(r: dict, n: int):
    """Compare one result row with its hand label -> None (no label) or dict(reg_ok, bad_q {q: truth}, ok)."""
    t = truth().get(Path(r["file"]).name.lower())
    if t is None:
        return None
    bad = {q: t["qs"].get(q) for q in range(1, min(n, 10) + 1) if t["qs"].get(q) != r["qs"].get(q)}
    reg_ok = r["reg"].upper() == t["reg"]
    return dict(reg_ok=reg_ok, bad_q=bad, ok=reg_ok and not bad, reg_true=t["reg"])


def slugify(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", name.strip()).strip("_")[:60]


def conn_for(slug: str):
    with _lock:
        if slug not in _conns:
            _conns[slug] = db.connect(SETS / slug / "grades.db")
        return _conns[slug]


def meta_of(slug: str) -> dict:
    p = SETS / slug / "meta.json"
    m = {"n_questions": 10, "max_mark": 100, "roster_file": "", "test_mode": False}
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
        _pipe.test_mode = bool(meta["test_mode"])
        _pipe.config = ExamConfig(slug, None, {q: QuestionLimit(max=meta["max_mark"]) for q in range(1, meta["n_questions"] + 1)})
        return _pipe


def list_sets():
    out = []
    for d in sorted(SETS.glob("*/grades.db"), key=lambda p: p.stat().st_mtime, reverse=True):
        rows = sheet_rows(d.parent.name)
        out.append((d.parent.name, sum(r["tier"] == "confirmed" for r in rows), sum(r["tier"] != "confirmed" for r in rows)))
    return out


UI_CSS = ROOT / "grade_eval" / "assets" / "ui.css"
FONTS = ("<link rel=preconnect href='https://fonts.googleapis.com'><link rel=preconnect href='https://fonts.gstatic.com' crossorigin>"
         "<link href='https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&display=swap' rel=stylesheet>")
LOGO = "<i><svg viewBox='0 0 24 24' fill=none stroke=#fff stroke-width=2.4 stroke-linecap=round stroke-linejoin=round><path d='M5 12.5l4.5 4.5L19 7'/></svg></i><span>Grade<b>Eval</b></span>"
ICON_HOME = "<svg viewBox='0 0 24 24'><rect x=3 y=3 width=7.5 height=7.5 rx=2 /><rect x=13.5 y=3 width=7.5 height=7.5 rx=2 /><rect x=3 y=13.5 width=7.5 height=7.5 rx=2 /><rect x=13.5 y=13.5 width=7.5 height=7.5 rx=2 /></svg>"
ICON_PLUS = "<svg viewBox='0 0 24 24'><circle cx=12 cy=12 r=9 /><path d='M12 8v8M8 12h8'/></svg>"


@app.get("/ui.css")
def ui_css():
    return FileResponse(UI_CSS, media_type="text/css")


def page(body: str, crumbs: str = "") -> str:
    """crumbs: html shown above the page title (a link back to the set, etc.)."""
    return (f"<!doctype html><meta charset=utf-8><meta name=viewport content='width=device-width,initial-scale=1'>"
            f"<title>GradeEval</title>{FONTS}<style>{UI_CSS.read_text(encoding='utf-8')}</style>"
            f"<div id=busy><div>Reading sheets…<br><span style='font-weight:400;color:var(--mut)'>a few seconds each, please wait</span></div></div>"
            f"<div class=shell><div class=topm><a class=logo href=/>{LOGO}</a></div><aside class=side><a class=logo href=/>{LOGO}</a>"
            f"<a class='nav on' href=/>{ICON_HOME}Dashboard</a><a class=nav href='/?new=1'>{ICON_PLUS}New set</a></aside>"
            f"<div class=content>"
            f"{'<div class=crumbs>' + crumbs + '</div>' if crumbs else ''}{body}</div></div>"
            f"<script>document.querySelectorAll('form[data-busy]').forEach(f=>f.addEventListener('submit',()=>"
            f"document.getElementById('busy').style.display='flex'))</script>")


def fmt(v):
    return "" if v is None else format(v, "g")


# ------------------------------------------------------------------ home: list of sets + create
def greeting() -> str:
    h = time.localtime().tm_hour
    return "Good morning" if h < 12 else "Good afternoon" if h < 17 else "Good evening"


@app.get("/", response_class=HTMLResponse)
def home(new: int = 0):
    sets = list_sets()
    total = sum(a + b for _, a, b in sets)
    attention = sum(b for _, _, b in sets)
    cards = "".join(
        f"<div class=set><div><h3>{html.escape(s)}</h3></div>"
        f"<div><div class=meta><span>{a} of {a + b} confirmed</span>{f'<span class=tag>{b} to review</span>' if b else ''}</div>"
        f"<div class=bar><i style='width:{100 * a // (a + b) if a + b else 0}%'></i></div></div>"
        f"<div class=row><a class='btn dark sm' href='/s/{s}/photo-link'>Upload photos</a><a class='btn alt sm' href='/s/{s}'>Open results</a></div></div>"
        for s, a, b in sets)
    listing = (f"<div class=sets>{cards}</div>" if cards else
               "<div class='card empty'><h2>No sets yet</h2><p>A set is one pile of answer sheets, for example one exam.</p>"
               "<button onclick=\"document.getElementById('new').showModal()\">Create your first set</button></div>")
    return page(f"""<div class=head><div><h1>{greeting()}</h1><p class=sub>Scan graded answer sheets and export the marks to Excel</p></div>
<button onclick="document.getElementById('new').showModal()"><span class=plus>+</span> New set</button></div>
<div class=tiles><div class=tile><b>{len(sets)}</b><span>Sets</span></div><div class=tile><b>{total}</b><span>Sheets scanned</span></div>
<div class='tile chk'><b>{attention}</b><span>Need your attention</span></div></div>
<div class=head style='margin:26px 0 12px'><h2>Your sets</h2></div>{listing}
<dialog id=new><form action=/sets method=post enctype=multipart/form-data>
<div class=head><h2>New set</h2><button type=button class='btn alt sm' onclick="this.closest('dialog').close()">Close</button></div>
<label class=f>Name</label><input type=text name=name placeholder="e.g. BCSE209L CAT-1 slot B2" required>
<label class=f>Class list (Excel or CSV with register number and name)</label><input type=file name=roster accept=".xlsx,.csv">
<label class=f>Questions in the paper</label><input type=number name=n_questions value=10 min=1 max=40>
<input type=hidden name=max_mark value=100><input type=hidden name=test_mode value=1>
<p class=mut>The class list lets the app match each sheet to a student and fill the Excel in your order.</p>
<button style='width:100%;margin-top:6px'>Create set</button></form></dialog>
<script>if(new URLSearchParams(location.search).get('new'))document.getElementById('new').showModal()</script>""")


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
               test_mode: str = Form(""), roster: UploadFile | None = File(None)):
    slug = slugify(name) or "set"
    save_meta(slug, {"n_questions": max(1, min(40, n_questions)), "max_mark": max_mark, "roster_file": "", "test_mode": bool(test_mode)})
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
        progress = (f"<div style='margin-top:14px'><div style='display:flex;justify-content:space-between;font-size:12px;color:var(--mut);margin-bottom:5px'>"
                    f"<span>{len(done)} of {len(roster)} students on the class list confirmed</span><span>{pct}%</span></div>"
                    f"<div class=bar><i style='width:{pct}%'></i></div></div>")
    shown = [r for r in rows if not filter or r["tier"] == filter]
    head = "".join(f"<th>Q{i}</th>" for i in range(1, n + 1))
    body = ""
    scores = {r["id"]: score(r, n) for r in rows}
    has_truth = any(scores.values())
    for r in shown:
        cls, label = TIER[r["tier"]]
        sc = scores[r["id"]]
        cells = "".join(
            f"<td class='n{' bad' if sc and i in sc['bad_q'] else ''}' title='read: {html.escape(r['raw'].get(i, ''), quote=True)}"
            f"{'; truth: ' + (fmt(sc['bad_q'][i]) or 'blank') if sc and i in sc['bad_q'] else ''}'>{fmt(r['qs'].get(i))}</td>"
            for i in range(1, n + 1))
        verdict = "" if not has_truth else "<td class=mut>-</td>" if sc is None else (
            "<td class=ok>correct</td>" if sc["ok"] else
            f"<td class=bad>wrong: {'reg ' + html.escape(sc['reg_true']) + ' ' if not sc['reg_ok'] else ''}"
            f"{'Q' + ','.join(map(str, sc['bad_q'])) if sc['bad_q'] else ''}</td>")
        total = sum(r["qs"].values()) if r["qs"] else None
        why = "; ".join(x.replace(CONFIRM, "") for x in r["reasons"])
        body += (f"<tr><td><span class='badge {cls}'>{label}</span></td>"
                 f"<td><a href='/s/{slug}/sheet/{r['id']}'>{html.escape(r['reg'] or '?')}</a></td>"
                 f"<td>{html.escape(r['name'])}</td>{cells}<td class=n><b>{fmt(total)}</b></td>{verdict}"
                 f"<td class=mut title='{html.escape(why, quote=True)}'>{html.escape(why[:90])}</td>"
                 f"<td><a href='/s/{slug}/sheet/{r['id']}'>{'view' if r['tier'] == 'confirmed' else 'open'}</a></td></tr>")
    table = (f"<div class=tw><table><tr><th>Status</th><th>Reg no</th><th>Name</th>{head}<th>Total</th>"
             f"{'<th>vs truth</th>' if has_truth else ''}<th>Note</th><th></th></tr>{body}</table></div>") if shown else "<div class=empty><h2>No sheets yet</h2><p>Scan sheets with your phone or upload photos.</p></div>"
    scored = [(r, scores[r["id"]]) for r in rows if scores[r["id"]]]
    report = ""
    if scored:
        k = len(scored)
        reg_ok = sum(sc["reg_ok"] for _, sc in scored)
        cells_all = sum(min(n, 10) for _ in scored)
        cells_bad = sum(len(sc["bad_q"]) for _, sc in scored)
        full = sum(sc["ok"] for _, sc in scored)
        auto = [(r, sc) for r, sc in scored if r["tier"] == "confirmed"]
        auto_bad = sum(not sc["ok"] for _, sc in auto)
        report = (f"<div class=card><details><summary>Accuracy against hand labels ({k} sheets)</summary><div class=tiles style='margin-top:14px'>"
                  f"<div class=tile><b>{100 * full // k}%</b><span>sheets fully correct ({full}/{k})</span></div>"
                  f"<div class=tile><b>{100 * reg_ok // k}%</b><span>register numbers ({reg_ok}/{k})</span></div>"
                  f"<div class=tile><b>{100 * (cells_all - cells_bad) // cells_all}%</b><span>question marks ({cells_all - cells_bad}/{cells_all})</span></div>"
                  f"<div class='tile {'bad' if auto_bad else 'ok'}'><b>{auto_bad} / {len(auto)}</b><span>auto-confirmed sheets wrong</span></div></div>"
                  f"<p class=mut>Red cells differ from the hand label (hover for the true value). Sheets without a label are not counted.</p></details></div>")
    roster_line = f"class list of {len(roster)} students" if roster else "no class list yet"
    bulk = (f"<form action='/s/{slug}/confirm_all' method=post onsubmit=\"return confirm('Confirm all {cnt['check']} sheets in to check "
            f"as correct without opening them?')\"><button class='btn warn sm'>Confirm all {cnt['check']} to check</button></form>"
            ) if cnt["check"] else ""
    def chip(key, label):
        return f"<a class='{'on' if filter == key else ''}' href='/s/{slug}{'?filter=' + key if key else ''}'>{label}</a>"
    return page(f"""<div class=head><div><h1>{html.escape(slug)}</h1><p class=sub>{n} questions &middot; {roster_line}</p></div>
<div class=row><a class='btn dark' href='/s/{slug}/photo-link'>Upload photos</a><a class=btn href='/s/{slug}/marks.xlsx'>Download Excel</a></div></div>{flash}
<div class=tiles><div class='tile ok'><b>{cnt['confirmed']}</b><span>Confirmed</span></div>
<div class='tile chk'><b>{cnt['check']}</b><span>To check</span></div>
<div class='tile bad'><b>{cnt['fix']}</b><span>Need fix</span></div>
<div class=tile><b>{len(rows)}</b><span>Sheets scanned</span></div></div>{progress}
{report}
<div class=card><div class=head style='margin-bottom:10px'><h2>Results</h2><div class=row>{bulk}</div></div>
<div class=chips style='margin-top:0'>{chip('', 'All')}{chip('check', 'To check')}{chip('fix', 'Need fix')}{chip('confirmed', 'Confirmed')}</div>{table}</div>
<div class=card><h2>Upload photos</h2><p class=mut style='margin-bottom:12px'>Have photos already? Each one is aligned, read and added to the results.</p>
<form action='/s/{slug}/upload' method=post enctype=multipart/form-data data-busy><p style='margin:0 0 12px'><input type=file name=files accept="image/*" multiple required></p>
<button class='btn alt' style='box-shadow:none'>Upload and read</button></form></div>
<div class=card><h2>Class list</h2><p class=mut style='margin-bottom:12px'>Upload or replace the Excel / CSV with register numbers and names.</p>
<form action='/s/{slug}/roster' method=post enctype=multipart/form-data><p style='margin:0 0 12px'><input type=file name=roster accept=".xlsx,.csv" required></p>
<button class='btn alt' style='box-shadow:none'>Upload class list</button></form></div>""",
                f"<a href='/'>Dashboard</a> &rsaquo; {html.escape(slug)}")


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
    return page(f"""<div class=head><div><h1>{html.escape(r['reg'] or 'Unknown')} <span class='badge {cls}'>{label}</span></h1><p class=sub>{html.escape(r['file'])}</p></div></div>
{('<div class=card><b>Notes</b><ul>' + why + '</ul></div>') if why else ''}
<div class=card><h2>Check the values against the sheet, then save</h2>
<form action='/s/{slug}/approve' method=post><input type=hidden name=sheet_id value={sheet_id}>
<div class=formgrid><div><label class=f>Register number</label><input type=text name=reg_no list=regs value='{html.escape(r['reg'])}'><datalist id=regs>{opts}</datalist></div>
<div><label class=f>Name</label><input type=text name=name value='{html.escape(r['name'])}'></div></div>
<label class=f>Marks</label><div class=grid>{fields}</div><p><button>Save as confirmed</button></p></form>
<p class=mut>Leave a box empty if that question has no marks. If the reg no is on the class list, the name is filled from it.</p></div>
<div class=card><h2>Sheet (aligned)</h2>{im}</div>""", f"<a href='/'>Dashboard</a> &rsaquo; <a href='/s/{slug}'>{html.escape(slug)}</a>")


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
        pending = [dict(reg=r["reg"], name=r["name"], qs=r["qs"]) for r in sheet_rows(slug) if r["tier"] == "check"]
        export_roster_excel(conn_for(slug), roster_of(slug), meta_of(slug)["n_questions"], path, pending)
    return FileResponse(path, filename=f"{slug}.xlsx")


# ------------------------------------------------------------------ phone photo upload (phone -> this PC)
@app.get("/s/{slug}/photo-link", response_class=HTMLResponse)
def photo_link(slug: str):
    if not (SETS / slug / "grades.db").exists():
        return RedirectResponse("/")
    photo = f"http://{phone.lan_ip()}:{phone.HTTP_PORT}/s/{slug}/photo"
    try:
        import segno
        qr = segno.make(photo, error="m").svg_inline(scale=6, border=2, dark="#000", light="#fff")
    except Exception:
        qr = ""
    return page(f"""<div class=head><div><h1>Upload photos</h1><p class=sub>Send photos of the sheets from your phone. They are read in the background.</p></div></div>
<div class=link>
<div class=card><h2>1. Open the page on your phone</h2>
<div class=pconly><p class=mut style='margin-top:6px'>Same Wi-Fi as this PC. Point the phone camera at the code:</p>
<div class=qrbox>{qr}</div><a class=addr href='{html.escape(photo)}'>{html.escape(photo)}</a></div>
<div class=onphone><p class=mut style='margin:6px 0 14px'>You are on your phone already, so go straight to the upload page:</p>
<a class=btn style='width:100%;padding:14px' href='{html.escape(photo)}'>Open photo upload</a></div></div>
<div class=card><h2>2. Send the sheets</h2><ol class=steps>
<li><span>Press <b>Take photo</b> for each sheet (the phone camera opens), or <b>Choose from gallery</b> to pick several at once.</span></li>
<li><span>Lay the sheet flat in good, even light and keep the whole page in the picture.</span></li>
<li><span>Each sheet is read in the background and appears in the list on the page.</span></li>
<li><span>Open the results here to review, then download the Excel.</span></li></ol>
<div class=row><a class='btn alt sm' href='/s/{slug}'>Back to results</a></div></div></div>""",
                f"<a href='/'>Dashboard</a> &rsaquo; <a href='/s/{slug}'>{html.escape(slug)}</a>")


@app.get("/s/{slug}/photo", response_class=HTMLResponse)
def photo_page(slug: str):
    """Photo mode: the phone's own camera app takes each sheet (full resolution)."""
    if not (SETS / slug / "grades.db").exists():
        return RedirectResponse("/")
    import html as _h
    return HTMLResponse((Path(__file__).parent / "assets" / "photo.html").read_text(encoding="utf-8").replace("__SLUG__", _h.escape(slug, quote=True)))


_jobs: dict = {}                    # capture id -> {"state": "reading" | "done" | "error", ...}
_reader = ThreadPoolExecutor(max_workers=1)   # sheets are read one at a time in the background so the phone is never kept waiting


def _read_photo(job: str, slug: str, data: bytes, name: str):
    try:
        with _lock:
            r = pipe_for(slug).process(data, name)
        prev = r["status"] == "duplicate"
        tier = tier_of(r["previous_status"] if prev else r["status"], r["reasons"])
        label = {"confirmed": "confirmed", "check": "to check", "fix": "needs fix"}[tier]
        who = r["reg_no"] or "sheet"
        say = f"{who} was already scanned" if prev else f"{who} read ({len(r['questions'])} marks, {label})"
        _jobs[job] = {"state": "done", "tier": tier, "label": label, "reg": r["reg_no"], "name": r["name"], "say": say}
    except Exception as e:
        _jobs[job] = {"state": "error", "msg": str(e)}


@app.post("/s/{slug}/photo/capture")
async def photo_capture(slug: str, request: Request):
    """Store the photo and return at once; the result is fetched from /photo/job/<id> when the sheet has been read."""
    data = await request.body()
    job = uuid.uuid4().hex
    _jobs[job] = {"state": "reading"}
    _reader.submit(_read_photo, job, slug, data, f"photo_{time.strftime('%Y%m%d_%H%M%S')}.jpg")
    return {"job": job}


@app.get("/s/{slug}/photo/job/{job}")
def photo_job(slug: str, job: str):
    return _jobs.get(job, {"state": "error", "msg": "unknown job"})


def warm_up():
    """Load the models at start so the first sheet is not slow."""
    global _pipe
    from grade_eval.vision import layout
    with _lock:
        if _pipe is None:
            _pipe = GradePipeline(db_path=":memory:", save_aligned=ALIGNED)
        layout._get_template()


@app.on_event("startup")
def _warm():
    threading.Thread(target=warm_up, daemon=True).start()
