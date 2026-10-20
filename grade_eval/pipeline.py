"""Photo -> aligned sheet -> register no. + marks -> roster match -> validate -> SQLite.

    pipe = GradePipeline(roster=load_roster("class.xlsx"))
    result = pipe.process(open("sheet.jpg", "rb").read(), "sheet.jpg")   # result["status"] in verified | needs_review | duplicate

Anything uncertain (could not align, unreadable/complex marks row, marks outside Q1-10, reg no. not on roster, failed
validation) is stored with status needs_review and reasons; only clean sheets write to students/marks.
The name reader is not built yet: the name comes from the roster when the register number matches, else it is empty.
"""
import json
import os
import re
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))   # train_marks (MarksNet, CTC decode) and finetune_header (TrOCR helpers) live in scripts/

from grade_eval import marks_solver, roster as roster_mod, totals as totals_mod
from grade_eval.exam_config import ExamConfig, QuestionLimit
from grade_eval.store import db
from grade_eval.validate import REG_NO_RE, validate_sheet
from grade_eval.vision import cells, header, layout
from grade_eval.vision.preprocess import load_image

MARK_CONF = 0.90        # per-row confidence needed to accept a marks read
MAX_Q_MARK = 30         # larger reads are almost certainly garbage
NAME_TIEBREAK_SIM = 0.4   # exact register match with a look-alike classmate: name must resemble the matched student at least this much ...
NAME_TIEBREAK_LEAD = 0.1  # ... and more than any look-alike classmate by this margin (0.4/0.1 accepted 48 of 73 such matches, 0 wrong, on the labelled sheets)
TOT_LR = 11.0           # nats of evidence that the total strips read this number rather than nothing (false-evidence rate ~1% per wrong number)
TOT_MARGIN = 3.0        # lead (nats) of the winning total hypothesis over the runner-up
GUESS_MIN = 0.25        # unclear rows show their best guess to the reviewer when it has at least this probability
SOLO_CONF = 0.97        # single-number rows without a sub-total cross-check must be at least this confident
CONFIRM = "please confirm: "   # reasons with this prefix mean "read fine, just glance at it"; anything else needs a fix
BLANK_INK = 0.003       # ink fraction of the a-column below which a row is blank


def default_config(max_mark: float = 100) -> ExamConfig:
    """Used when no exams/<COURSE>.yaml exists: only a sanity ceiling per question."""
    return ExamConfig("DEFAULT", None, {q: QuestionLimit(max=max_mark) for q in range(1, 41)})


def parse_sum(s: str) -> float | None:
    """'3+5+1' -> 9.0 (a|b entries are read as a+b); None when it is not a clean sum of numbers."""
    if not re.fullmatch(r"\d{1,2}(\.\d)?(\+\d{1,2}(\.\d)?)*", s or ""):
        return None
    return sum(float(t) for t in s.split("+"))


class GradePipeline:
    def __init__(self, db_path=db.DEFAULT_DB, roster=None, config: ExamConfig | None = None, device: str | None = None,
                 save_aligned: Path | None = None, n_questions: int = 10):
        from transformers import VisionEncoderDecoderModel
        import finetune_header as fh
        
        self.dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.fh = fh
        fh.DEV = self.dev
        reg_path = ROOT / "models" / "trocr_reg_small"
        self.tok = fh.load_tokenizer(str(reg_path))
        self.reg_model = VisionEncoderDecoderModel.from_pretrained(reg_path, torch_dtype=torch.float32).to(self.dev).eval()
        import torch.nn as nn
        import train_marks_sum as tms
        self.ctc_decode = tms.decode
        self.marks_net = tms.make_net()
        self.marks_net.load_state_dict(torch.load(ROOT / "models" / "marks_sum.pt", map_location="cpu"))
        self.marks_net.eval()
        self.hard_net = None        # specialist for rows with a|b entries or written sums; simple rows stay on marks_sum.pt
        hard_path = ROOT / "models" / "hard_b_s0.pt"
        if hard_path.exists():
            import train_marks_hard as th
            self.th = th
            ck = torch.load(hard_path, map_location="cpu")
            self.hard_net = tms.make_net()
            self.hard_net.load_state_dict(ck["state"])
            self.hard_net.eval()
            self.hard_coord = ck["coord"]
        self.n_questions = n_questions
        self.total_reader = None    # reads the written total (Sub-Total cell / Grand Total box) to cross-check the marks
        suffix = os.environ.get("GRADE_TOTAL_SUFFIX", "")     # "_eval" = readers trained on the train split only (honest scoring)
        sp, gp = ROOT / "models" / f"total_sub{suffix}.pt", ROOT / "models" / f"total_grand{suffix}.pt"
        if sp.exists() and gp.exists() and os.environ.get("GRADE_NO_TOTAL") != "1":
            self.total_reader = totals_mod.TotalReader(sp, gp, self.dev)
        self.name_model = None
        name_path = ROOT / "models" / "trocr_name_small"      # weak free-text reader: used only as a roster hint
        if name_path.exists():
            self.name_model = VisionEncoderDecoderModel.from_pretrained(name_path, torch_dtype=torch.float32).to(self.dev).eval()
            self.name_tok = fh.load_tokenizer(str(name_path))
        self.conn = db.connect(db_path)
        self.roster = roster or []
        self.config = config or default_config()
        self.save_aligned = save_aligned
        self.test_mode = False      # relaxes the two "please confirm" rules below; the front end scores against ground truth

    # --- readers -------------------------------------------------------------------------------------------------
    def read_reg(self, aligned: np.ndarray) -> str:
        strip = torch.tensor(header.ink_strip(aligned)).unsqueeze(0).unsqueeze(0)
        return self.fh.decode(self.reg_model, self.tok, strip.to(self.dev), "reg")[0]

    def read_name(self, aligned: np.ndarray) -> str:
        if self.name_model is None:
            return ""
        strip = torch.tensor(header.name_strip(aligned)).unsqueeze(0).unsqueeze(0)
        return self.fh.decode(self.name_model, self.name_tok, strip.to(self.dev), "name")[0]

    @staticmethod
    def _value_conf_c(c: dict, s: str, frame_conf: float) -> float:
        v = 0.0 if s == "" else parse_sum(s)
        if v is None:
            return frame_conf
        return float(np.exp(c[v])) if v in c else frame_conf

    @staticmethod
    def _value_conf(lg: np.ndarray, s: str, frame_conf: float) -> float:
        v = 0.0 if s == "" else parse_sum(s)
        if v is None:
            return frame_conf
        c = marks_solver.candidates(lg, keep=4)
        return float(np.exp(c[v])) if v in c else frame_conf

    def read_marks(self, aligned: np.ndarray, n_questions: int | None = None):
        """-> (questions {q: {"a", "raw", "conf"}}, sub_total|None (single group only), confidences, reasons, crosschecked).

        Rows are read with the sum-capable reader: '3+5+1' and a|b entries are added. Anything unclear is not
        guessed; it is listed in reasons so the sheet goes to review. Each group of 10 questions is its own column
        block (same layout shifted right), checked against that block's written sub-total.
        """
        n = n_questions or self.n_questions
        mask = cells.ink_mask(aligned)
        groups = (n + 9) // 10
        strips, keys, simple = [], [], []
        for g in range(groups):
            dx = layout.GROUP_X[g][0] - layout.GROUP_X[0][0]
            xr = (cells.SPILL_X[0] + dx, cells.SPILL_X[1] + dx)
            for i in range(11):
                x0, y0, x1, y1 = cells.row_box(i, xr)
                m = mask[y0:y1, x0:x1].astype(np.uint8) * 255
                strips.append(255 - cv2.resize(m, (cells.W, cells.H), interpolation=cv2.INTER_AREA))
                keys.append((g, i, (x0, y0, x1, y1)))
                simple.append(i == 10 or cells.ink_fraction(mask, i, (cells.B_X[0] + 8 + dx, cells.SPILL_X[1] + dx)) < 0.004)
        X = torch.tensor(np.stack(strips).astype(np.float32) / 255).unsqueeze(1)
        with torch.no_grad():
            logits = self.marks_net(X)
            if self.hard_net is not None and os.environ.get("GRADE_ENS", "1") == "1":
                hl = self.th.forward(self.hard_net, X, self.hard_coord)      # average the two readers on every row
                logits = torch.log(0.5 * logits.softmax(-1) + 0.5 * hl.softmax(-1) + 1e-9)
            elif self.hard_net is not None and not all(simple):
                logits = torch.where(torch.tensor(simple).view(-1, 1, 1), logits, self.th.forward(self.hard_net, X, self.hard_coord))
            reads = self.ctc_decode(logits)
            # confidence = probability of the read VALUE (all spellings/alignments that give it), not the weakest single
            # frame: a correct but loosely-written '3' scores ~0.6 on the frame rule yet ~0.9+ on the value
            cand_list = [marks_solver.candidates(lg.numpy(), keep=6) for lg in logits]
            self.last_cands, self.last_keys = cand_list, keys                 # for tentative read-outs and analysis
            reads = [(sx, self._value_conf_c(c, sx, cf)) for c, (sx, cf) in zip(cand_list, reads)]
        questions, confs, reasons, subs = {}, [], [], {}
        self.last_guesses = {}
        for idx, ((g, i, box), (s, conf)) in enumerate(zip(keys, reads)):
            if i == 10:
                if s and conf >= MARK_CONF:
                    v = parse_sum(s)
                    if v is not None:
                        subs[g] = v
                continue
            q = g * 10 + i + 1
            if q > n:
                continue
            x0, y0, x1, y1 = box
            ink = float(mask[y0:y1, x0:x1].mean())
            if s == "" and (conf >= MARK_CONF or ink < BLANK_INK):
                continue                                           # blank row: question not marked
            v = parse_sum(s)
            if v is None or conf < MARK_CONF or v > MAX_Q_MARK:
                reasons.append(f"Q{q}: unclear read '{s}' (conf {conf:.2f})")
                cd = self.last_cands[idx]
                if cd:                                              # best guess, shown highlighted for the reviewer; never auto-accepted
                    gv, glp = max(cd.items(), key=lambda kv: kv[1])
                    if np.exp(glp) >= GUESS_MIN:
                        self.last_guesses[q] = {"a": gv, "p": round(float(np.exp(glp)), 2)}
                continue
            questions[q] = {"a": v, "b": None, "raw": s, "conf": round(conf, 3)}
            confs.append(conf)
        tot_note = None
        if self.total_reader is not None and groups == 1:
            questions, reasons, confs, sub, crosschecked, tot_note = self._cross_check_total(
                mask, keys, reads, logits, questions, reasons, confs, n)
            return questions, sub, confs, reasons, crosschecked
        for g in range(groups):                                    # per-block written sub-total check
            if g in subs:
                tot = sum(v["a"] for q, v in questions.items() if (q - 1) // 10 == g)
                missing = any(r.startswith(f"Q{q}:") for r in reasons for q in range(g * 10 + 1, g * 10 + 11))
                if abs(tot - subs[g]) > 1e-6 and not missing:
                    reasons.append(f"sub-total of Q{g * 10 + 1}-{g * 10 + 10} reads {subs[g]:g} but questions add to {tot:g}")
        sub = subs.get(0) if groups == 1 else None
        used = {(q - 1) // 10 for q in questions}
        crosschecked = bool(used) and all(
            g in subs and abs(sum(v["a"] for q, v in questions.items() if (q - 1) // 10 == g) - subs[g]) < 1e-6 for g in used)
        return questions, sub, confs, reasons, crosschecked


    def _cross_check_total(self, mask, keys, reads, logits, questions, reasons, confs, n):
        """Confirm the question marks with the written total (Sub-Total cell / Grand Total box).

        For every sum the rows could reach (their best readings and runner-ups) we score how well the total strips match that
        number against "nothing written" (CTC likelihood ratio, `totals.evidence`) and add the rows' own log-probability.
        Only when the BEST reading of every row already adds up to the winning number, the evidence is strong (TOT_LR) and the
        winner leads the next hypothesis clearly (TOT_MARGIN) is the sheet cross-checked: unclear-looking rows are then
        accepted. Changing rows to fit a total was tried and was wrong on every held-out case, so it is not done.
        """
        sl, gl = self.total_reader.logits(mask)
        rows, ink_blank = {}, {}
        for idx, ((g, i, box), (sx, cf)) in enumerate(zip(keys, reads)):
            if i == 10 or i + 1 > n:
                continue
            x0, y0, x1, y1 = box
            ink_blank[i + 1] = float(mask[y0:y1, x0:x1].mean()) < BLANK_INK
            rows[i + 1] = (self.last_cands[idx], sx, cf)
        if not rows or any(not c for c, _, _ in rows.values()):
            return questions, reasons, confs, None, False, None
        dp = {0: (0.0, ())}
        for q in sorted(rows):
            nd = {}
            for sm, (lp, asg) in dp.items():
                for v, l in rows[q][0].items():
                    k = sm + int(round(v * 10))
                    if k not in nd or lp + l > nd[k][0]:
                        nd[k] = (lp + l, asg + (v,))
            dp = nd
        sums = sorted(dp)
        values = [k / 10 for k in sums]
        E = totals_mod.evidence(sl, gl, values)
        # the marks readers also read digits in the Sub-Total cell (never trained on totals): extra, independent-ish evidence
        sx = torch.tensor(totals_mod.sub_strip(mask).astype(np.float32) / 255).view(1, 1, 64, 192)
        with torch.no_grad():
            extra = [self.marks_net(sx)[0].numpy()]
            if self.hard_net is not None:
                extra.append(self.th.forward(self.hard_net, sx, self.hard_coord)[0].numpy())
        for lg in extra:
            E = np.maximum(E, totals_mod.evidence(lg, np.zeros((0,) + lg.shape, np.float32), values))
        H = np.array([dp[k][0] for k in sums]) + E
        order = np.argsort(-H)
        j = int(order[0])
        margin = float(H[j] - H[order[1]]) if len(order) > 1 else 1e9
        best = {q: max(c.items(), key=lambda kv: kv[1])[0] for q, (c, _, _) in rows.items()}
        if E[j] >= TOT_LR and margin >= TOT_MARGIN and abs(sum(best.values()) - values[j]) < 1e-6:
            newq, newconf = {}, []
            for q, v in best.items():
                c, sx, cf = rows[q]
                if (ink_blank[q] and v == 0.0) or (v == 0.0 and sx == ""):
                    continue
                pv = float(np.exp(c[v]))
                raw = sx if marks_solver.parse_sum(sx) == v else f"{v:g}"
                newq[q] = {"a": v, "b": None, "raw": raw, "conf": round(pv, 3)}
                newconf.append(pv)
            keep = [r for r in reasons if not (r.startswith("Q") and "unclear read" in r)]
            self.last_guesses = {}
            return newq, keep, newconf, values[j], True, "agree"
        return questions, reasons, confs, None, False, None

    # --- main entry ------------------------------------------------------------------------------------------------
    def process(self, data: bytes, filename: str = "") -> dict:
        h = db.hash_image(data)
        out = {"file": filename, "hash": h, "reg_no": "", "name": "", "questions": {}, "sub_total": None, "reasons": []}
        row = self.conn.execute("SELECT * FROM sheets WHERE image_hash = ?", (h,)).fetchone()
        if row is not None:                                  # same photo seen before: report the stored result
            ex = json.loads(row["extracted"])
            out.update(status="duplicate", previous_status=row["status"], reg_no=row["reg_no"] or "",
                       name=ex.get("name", ""), reg_raw=ex.get("reg_raw", ""), sub_total=ex.get("sub_total"),
                       questions={int(q): v for q, v in ex.get("questions", {}).items()},
                       reasons=json.loads(row["reasons"]), image=h)
            return out

        img = load_image(data)
        res = layout.align(img)
        if res is None:
            out["reasons"] = ["could not align photo to the form (retake: flat, whole page, good light)"]
            out["status"] = db.save_sheet(self.conn, h, "", "", "", {}, out["reasons"], {"file": filename})
            return out
        aligned = res.image
        if self.save_aligned:
            self.save_aligned.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(self.save_aligned / (h + ".png")), aligned)
            out["image"] = h

        reg_raw = self.read_reg(aligned)
        questions, sub_total, confs, reasons, crosschecked = self.read_marks(aligned)
        reg_no, name = reg_raw, ""
        reg_exact = False

        name_read = self.read_name(aligned) if self.roster else ""
        if self.roster:
            m = roster_mod.match_sheet(reg_raw, name_read, self.roster)
            if m.row is not None and m.status != "review":
                reg_no, name = m.row.reg_no, m.row.name
                # exact match is trusted when the name agrees, or when no OTHER student on the list is one slip away
                # (then a single misread digit cannot silently turn into a classmate)
                near = [r for r in self.roster if r.reg_no != reg_no and roster_mod.lev(reg_raw, r.reg_no) <= 1]
                reg_exact = reg_no == reg_raw and (m.status == "auto" or (not near and os.environ.get("GRADE_REG_UNIQUE", "1") == "1"))
                if reg_no == reg_raw and not reg_exact and near and name_read and os.environ.get("GRADE_NAME_TIEBREAK", "1") == "1":
                    # a classmate is one slip away: the handwritten name must favour the matched student over every such neighbour
                    s_m = roster_mod.name_similarity(name_read, m.row.name)
                    s_n = max(roster_mod.name_similarity(name_read, r.name) for r in near)
                    reg_exact = s_m >= NAME_TIEBREAK_SIM and s_m - s_n >= NAME_TIEBREAK_LEAD
                if reg_no != reg_raw:
                    reasons.append(CONFIRM + f"register no. read '{reg_raw}', matched {reg_no} on the class list "
                                   f"({'name agrees' if m.status == 'auto' else 'name not readable'})")
            else:
                reasons.append("register no. not matched to the class list: " + "; ".join(m.reasons))
        elif not REG_NO_RE.match(reg_raw):
            pass  # validate_sheet reports it

        reasons += validate_sheet(reg_no, questions, None, self.config, confs)   # sub-total is checked per block in read_marks
        if not reasons:                                      # nothing wrong, but is it safe to write without a human glance?
            if not questions:
                reasons.append(CONFIRM + "no marks were found on this sheet")
            elif not self.test_mode and not crosschecked and (any("+" in v["raw"] for v in questions.values()) or min(v["conf"] for v in questions.values()) < SOLO_CONF):
                reasons.append(CONFIRM + "marks read, but no written sub-total to cross-check them against")
            if self.roster and not reg_exact and not (self.test_mode and reg_no == reg_raw):
                reasons.append(CONFIRM + "register number matched the class list, but a classmate with a very similar number is on the list and the handwritten name did not clearly tell them apart")
            elif not self.roster:
                reasons.append(CONFIRM + "register number read by OCR and no class list uploaded to check it")
        out.update(reg_no=reg_no, name=name, questions=questions, sub_total=sub_total, reasons=reasons, reg_raw=reg_raw)
        extracted = {k: out[k] for k in ("file", "reg_raw", "reg_no", "name", "sub_total")}
        extracted["name_read"] = name_read
        extracted["questions"] = {str(q): v for q, v in questions.items()}
        extracted["guesses"] = {str(q): v for q, v in getattr(self, "last_guesses", {}).items() if q not in questions}
        extracted["crosschecked"] = crosschecked
        out["status"] = db.save_sheet(self.conn, h, reg_no, name, self.config.course_code, questions, reasons, extracted)
        return out
