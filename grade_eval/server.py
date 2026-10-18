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

from grade_eval import live
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


CSS = """
:root{--bg:#f4f6fa;--card:#fff;--tx:#18202b;--mut:#64708a;--ok:#0f8a3f;--bad:#c8420a;--chk:#a86400;--ln:#e1e6ee;--ac:#2f5bea;--acs:#e8eefd;--okb:#e6f6ec;--badb:#fdeee6;--chkb:#fdf3df;--sh:0 1px 2px #1b24351a,0 4px 14px #1b24350d}
@media(prefers-color-scheme:dark){:root{--bg:#0e1116;--card:#171c24;--tx:#e8ebf0;--mut:#98a3b8;--ok:#4ade80;--bad:#fb923c;--chk:#facc15;--ln:#283040;--ac:#7aa2ff;--acs:#1c2742;--okb:#12301f;--badb:#35200f;--chkb:#332a0c;--sh:none}}
*{box-sizing:border-box}html{-webkit-text-size-adjust:100%}
body{font-family:Inter,system-ui,-apple-system,"Segoe UI",sans-serif;background:var(--bg);color:var(--tx);margin:0;line-height:1.45}
.top{background:var(--card);border-bottom:1px solid var(--ln);position:sticky;top:0;z-index:5}
.top div{max-width:1180px;margin:auto;padding:12px 20px;display:flex;align-items:center;gap:8px;font-size:14px;flex-wrap:wrap}
.top .logo{font-weight:700;font-size:16px;color:var(--tx);text-decoration:none;margin-right:6px}
.top .logo b{color:var(--ac)}.top .sep{color:var(--mut)}
main{max-width:1180px;margin:auto;padding:8px 20px 48px}
h1{font-size:26px;margin:18px 0 4px;letter-spacing:-.01em}h2{font-size:16px;margin:0 0 10px}a{color:var(--ac)}
.sub{color:var(--mut);margin:0 0 14px;font-size:14px}
.card{background:var(--card);border:1px solid var(--ln);border-radius:14px;padding:18px;margin:14px 0;box-shadow:var(--sh)}
.ok{color:var(--ok)}.bad{color:var(--bad)}.chk{color:var(--chk)}.mut{color:var(--mut);font-size:13px}
button,.btn{background:var(--ac);color:#fff;border:0;border-radius:10px;padding:10px 18px;font-size:15px;font-weight:600;cursor:pointer;text-decoration:none;display:inline-block;transition:filter .15s,transform .05s}
button:hover,.btn:hover{filter:brightness(1.08)}button:active,.btn:active{transform:translateY(1px)}
.btn.alt{background:transparent;color:var(--ac);border:1px solid var(--ac)}
.btn.warn{background:var(--chk);color:#fff}
input[type=file]{font-size:14px;max-width:100%;padding:10px;border:1.5px dashed var(--ln);border-radius:10px;background:var(--bg);width:100%}
input[type=file]:hover{border-color:var(--ac)}
input[type=file]::file-selector-button{background:var(--acs);color:var(--ac);border:0;border-radius:8px;padding:7px 12px;margin-right:10px;font-weight:600;cursor:pointer}
input[type=text],input[type=number]{font-size:15px;padding:9px 11px;border:1px solid var(--ln);border-radius:9px;background:var(--bg);color:var(--tx);max-width:100%}
input:focus{outline:2px solid var(--ac);outline-offset:1px}
.formgrid{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:4px 22px}.formgrid label.f{font-size:13px;font-weight:600;color:var(--mut);display:block;margin-top:10px}
.formgrid input[type=text],.formgrid input[type=number]{width:100%;margin-top:4px}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(88px,1fr));gap:10px}.grid label{font-size:12px;font-weight:600;color:var(--mut)}.grid input{width:100%}
img.prev{width:100%;border-radius:10px;border:1px solid var(--ln)}
.badge{display:inline-block;padding:2px 10px;border-radius:99px;font-size:12px;font-weight:600;white-space:nowrap}
.badge.ok{background:var(--okb)}.badge.bad{background:var(--badb)}.badge.chk{background:var(--chkb)}
.tw{overflow:auto;max-height:70vh;border:1px solid var(--ln);border-radius:10px}
table{border-collapse:separate;border-spacing:0;width:100%;font-size:14px}
td,th{padding:9px 11px;border-bottom:1px solid var(--ln);text-align:left;white-space:nowrap}
th{color:var(--mut);font-weight:600;font-size:12px;text-transform:uppercase;letter-spacing:.03em;background:var(--card);position:sticky;top:0;z-index:1}
tbody tr:hover td,table tr:hover td{background:var(--acs)}tr:last-child td{border-bottom:0}
td.n{text-align:right;font-variant-numeric:tabular-nums}td.n.bad{background:var(--badb);font-weight:700}
.flash{background:var(--okb);border:1px solid var(--ok);border-radius:10px;padding:10px 14px;margin:12px 0;color:var(--tx)}
#busy{display:none;position:fixed;inset:0;background:#0009;color:#fff;align-items:center;justify-content:center;font-size:18px;z-index:9;backdrop-filter:blur(3px)}
#busy div:before{content:"";display:block;width:34px;height:34px;margin:0 auto 14px;border:4px solid #fff4;border-top-color:#fff;border-radius:50%;animation:sp .9s linear infinite}
@keyframes sp{to{transform:rotate(360deg)}}
.row{display:flex;gap:12px;flex-wrap:wrap;align-items:center}
.bar{height:10px;background:var(--ln);border-radius:6px;overflow:hidden;margin-top:6px}.bar i{display:block;height:100%;background:var(--ok);border-radius:6px}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin:0 0 14px}
.tile{border:1px solid var(--ln);border-radius:12px;padding:12px 14px;background:var(--bg)}
.tile b{display:block;font-size:26px;line-height:1.1;letter-spacing:-.02em}.tile span{font-size:12px;color:var(--mut);font-weight:600;text-transform:uppercase;letter-spacing:.03em}
.tile.ok{background:var(--okb);border-color:transparent}.tile.chk{background:var(--chkb);border-color:transparent}.tile.bad{background:var(--badb);border-color:transparent}
.chips{display:flex;gap:8px;flex-wrap:wrap;margin:6px 0 12px}.chips a{padding:6px 14px;border-radius:99px;border:1px solid var(--ln);text-decoration:none;font-size:13px;font-weight:600;color:var(--mut);background:var(--card)}
.chips a.on{background:var(--ac);border-color:var(--ac);color:#fff}
.hero{display:flex;gap:16px;flex-wrap:wrap;align-items:center;justify-content:space-between}
.empty{text-align:center;padding:34px 10px;color:var(--mut)}
code{background:var(--bg);padding:1px 6px;border-radius:5px;font-size:12px}
ol{padding-left:20px}li{margin:4px 0}
@media(max-width:640px){main{padding:4px 12px 40px}h1{font-size:22px}.card{padding:14px}.top div{padding:10px 12px}}
"""


def page(body: str, crumbs: str = "") -> str:
    return (f"<!doctype html><meta charset=utf-8><meta name=viewport content='width=device-width,initial-scale=1'>"
            f"<title>Grade eval</title><style>{CSS}</style>"
            f"<div id=busy><div>Reading sheet(s)… a few seconds each, please wait</div></div>"
            f"<div class=top><div><a class=logo href=/>Grade<b>Eval</b></a><a href=/>All sets</a>{crumbs}</div></div><main>{body}</main>"
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
        for s, a, b in list_sets()) or "<tr><td class=empty>No sets yet. Create one below.</td></tr>"
    return page(f"""<h1>Grade evaluation</h1>
<p class=sub>Turn photos of graded answer sheets into an Excel of register numbers, names and marks.</p>
<div class=card><h2>Your sets</h2><div class=tw><table>{items}</table></div></div>
<div class=card><h2>Start a new set of papers</h2>
<form action=/sets method=post enctype=multipart/form-data>
<div class=formgrid>
<div><label class=f>Name of this set</label><input type=text name=name placeholder="e.g. BCSE209L CAT-1 slot B2" required></div>
<div><label class=f>Class list (Excel or CSV with register number and name columns)</label><input type=file name=roster accept=".xlsx,.csv"></div>
<div><label class=f>Number of questions in the paper</label><input type=number name=n_questions value=10 min=1 max=40></div>
<div><label class=f>Maximum marks for any one question (safety check)</label><input type=number name=max_mark value=100 min=1></div></div>
<p><label><input type=checkbox name=test_mode value=1 checked> <b>Test mode</b>: accept exact class-list matches and confident marks without
extra confirmation, and score every sheet against <code>data/labels_all.csv</code> (photos must keep their <code>sheet_NNN</code> names)</label></p>
<button>Create set</button></form>
<p class=mut>The class list lets the app match each sheet to a student and fill the Excel in your order.</p></div>""")


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
        progress = (f"<p><b>{len(done)} of {len(roster)}</b> students on the class list are confirmed"
                    f"<div class=bar><i style='width:{pct}%'></i></div></p>")
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
             f"{'<th>vs truth</th>' if has_truth else ''}<th>Note</th><th></th></tr>{body}</table></div>") if shown else "<div class=empty>No sheets yet.<br>Upload photos above or use live scanning.</div>"
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
        report = (f"<div class=card><h2>Accuracy against hand labels ({k} sheets)</h2><div class=tiles>"
                  f"<div class=tile><b>{100 * full // k}%</b><span>sheets fully correct ({full}/{k})</span></div>"
                  f"<div class=tile><b>{100 * reg_ok // k}%</b><span>register numbers ({reg_ok}/{k})</span></div>"
                  f"<div class=tile><b>{100 * (cells_all - cells_bad) // cells_all}%</b><span>question marks ({cells_all - cells_bad}/{cells_all})</span></div>"
                  f"<div class='tile {'bad' if auto_bad else 'ok'}'><b>{auto_bad} / {len(auto)}</b><span>auto-confirmed sheets wrong</span></div></div>"
                  f"<p class=mut>Red cells differ from the hand label (hover for the true value). Sheets without a label are not counted.</p></div>")
    mode = ("<span class=chk>test mode on</span> &middot; " if meta["test_mode"] else "")
    roster_line = (f"{len(roster)} students" if roster else "none (register numbers will not be checked)")
    bulk = (f"<form action='/s/{slug}/confirm_all' method=post onsubmit=\"return confirm('Confirm all {cnt['check']} sheets in \\'to check\\' "
            f"as correct without opening them?')\"><button class='btn warn'>Confirm all {cnt['check']} “to check”</button></form>"
            ) if cnt["check"] else ""
    def chip(key, label):
        return f"<a class='{'on' if filter == key else ''}' href='/s/{slug}{'?filter=' + key if key else ''}'>{label}</a>"
    return page(f"""<h1>{html.escape(slug)}</h1>
<p class=sub>{mode}{n} questions &middot; class list: {roster_line}</p>{flash}
{report}<div class=card><div class=tiles>
<div class='tile ok'><b>{cnt['confirmed']}</b><span>confirmed</span></div>
<div class='tile chk'><b>{cnt['check']}</b><span>to check</span></div>
<div class='tile bad'><b>{cnt['fix']}</b><span>need fix</span></div>
<div class=tile><b>{len(rows)}</b><span>sheets scanned</span></div></div>{progress}</div>
<div class=card><div class=hero><div><h2>Add answer sheets</h2>
<p class=mut style='margin:0'>Pick one or many photos. Each is aligned, read, matched to the class list and added below.</p></div>
<a class=btn href='/s/{slug}/live-link'>&#128247; Open live scanning</a></div>
<form action='/s/{slug}/upload' method=post enctype=multipart/form-data data-busy style='margin-top:12px'>
<p><input type=file name=files accept="image/*" multiple required></p>
<button>Upload and read</button></form></div>
<div class=card><div class=hero><h2 style='margin:0'>Results</h2>
<div class=row><a class=btn href='/s/{slug}/marks.xlsx'>Download Excel</a>{bulk}</div></div>
<div class=chips>{chip('', 'All')}{chip('check', 'To check')}{chip('fix', 'Need fix')}{chip('confirmed', 'Confirmed')}</div>
<p class=mut><b>Confirmed</b>: cross-checked. <b>To check</b>: read fine, open it, glance and save (or confirm all).
<b>Need fix</b>: something unclear, correct it by hand. Hover a mark to see the raw read (e.g. 3+5+1).</p>{table}</div>
<div class=card><h2>Class list</h2><form action='/s/{slug}/roster' method=post enctype=multipart/form-data>
<p><input type=file name=roster accept=".xlsx,.csv" required></p><button class=alt>Upload / replace class list</button></form></div>""",
                f"<span class=sep>&rsaquo;</span> {html.escape(slug)}")


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
<div class=card><h2>Sheet (aligned)</h2>{im}</div>""", f"<span class=sep>&rsaquo;</span> <a href='/s/{slug}'>{html.escape(slug)}</a>")


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


# ------------------------------------------------------------------ live scanning (phone camera -> this PC)
@app.get("/s/{slug}/live-link", response_class=HTMLResponse)
def live_link(slug: str):
    if not (SETS / slug / "grades.db").exists():
        return RedirectResponse("/")
    ip = live.lan_ip()
    cert, key = live.cert_files()
    https = cert.exists() and key.exists()
    phone = f"https://{ip}:{live.HTTPS_PORT}/s/{slug}/live" if https else f"http://{ip}:{live.HTTP_PORT}/s/{slug}/live"
    try:
        import segno
        qr = segno.make(phone, error="m").svg_inline(scale=6, border=2, dark="#000", light="#fff")
    except Exception:
        qr = ""
    warn = "" if https else ("<p class=bad>No https certificate yet: phone cameras need https. Start the app with start.bat "
                             "(it creates one), then reload this page.</p>")
    return page(f"""<h1>Live scanning</h1>
<div class=card><h2>1. Connect the phone (same Wi-Fi as this PC)</h2>
<p>Scan this code with the phone camera, or type the address:</p>
<div style="background:#fff;padding:8px;display:inline-block;border-radius:8px;max-width:260px">{qr}</div>
<p><a href='{html.escape(phone)}'><b>{html.escape(phone)}</b></a></p>{warn}
<p class=mut>First time only: the phone shows a security warning because the certificate is made on this PC. Choose
<b>Advanced &rarr; Continue / Proceed</b>, then allow the camera.</p></div>
<div class=card><h2>2. Scan</h2><ol>
<li>Press <b>Start scanning</b> and hold the phone flat above the first sheet.</li>
<li>If the sheet is not lined up the phone says <b>Align the sheet</b>; when it turns green it captures by itself.</li>
<li>After each capture it says <b>NEXT</b>: swap in the next sheet. Sheets are read here on the PC.</li>
<li>Press <b>Stop scanning</b> when finished, then review the results on this page.</li></ol>
<p><a class=btn href='/s/{slug}/live' target=_blank>Or scan with this PC's webcam</a>
<a class='btn alt' href='/live-test' target=_blank>Test the camera only</a>
<a class='btn alt' href='/s/{slug}'>Back to results</a></p></div>""", f"<span class=sep>&rsaquo;</span> <a href='/s/{slug}'>{html.escape(slug)}</a>")


@app.get("/s/{slug}/live", response_class=HTMLResponse)
def live_page(slug: str):
    if not (SETS / slug / "grades.db").exists():
        return RedirectResponse("/")
    return HTMLResponse(live.page_html(slug))


@app.get("/live-test", response_class=HTMLResponse)
def live_test_page():
    """Camera framing test: no set needed, nothing is captured or stored (the last frame is kept in data/live_debug/)."""
    return HTMLResponse(live.page_html("test", test=True))


@app.get("/live-test/sheet", response_class=HTMLResponse)
def live_test_sheet():
    """Open on the PC and point the phone at the screen: a sample sheet shown as large as possible on white (screens are a poor stand-in for paper)."""
    return HTMLResponse("<!doctype html><meta charset=utf-8><title>Sample sheet</title><body style='margin:0;background:#fff'>"
                        "<img src='/live-test/sheet.jpg' style='display:block;margin:auto;height:100vh;max-width:100vw;object-fit:contain'>")


@app.get("/live-test/sheet.jpg")
def live_test_sheet_img():
    files = sorted((ROOT / "data" / "samples").glob("sheet_*.jp*g"))
    if not files:
        raise HTTPException(404)
    return FileResponse(files[0])


@app.post("/s/{slug}/live/frame")
async def live_frame(slug: str, request: Request):
    data = await request.body()

    def work():
        with _lock:                     # cv2's SIFT object is shared with the pipeline
            if slug == "test":
                live.save_debug_frame(data)
            return live.check_frame(data)
    return await run_in_threadpool(work)


_jobs: dict = {}                    # live capture id -> {"state": "reading" | "done" | "error", ...}
_reader = ThreadPoolExecutor(max_workers=1)   # sheets are read one at a time in the background so the phone is never kept waiting


def _read_live(job: str, slug: str, data: bytes, name: str):
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


@app.post("/s/{slug}/live/capture")
async def live_capture(slug: str, request: Request):
    """Store the photo and return at once; the result is fetched from /live/job/<id> when the sheet has been read."""
    data = await request.body()
    job = uuid.uuid4().hex
    _jobs[job] = {"state": "reading"}
    _reader.submit(_read_live, job, slug, data, f"live_{time.strftime('%Y%m%d_%H%M%S')}.jpg")
    return {"job": job}


@app.get("/s/{slug}/live/job/{job}")
def live_job(slug: str, job: str):
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
