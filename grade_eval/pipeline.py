"""Photo -> aligned sheet -> register no. + marks -> roster match -> validate -> SQLite.

    pipe = GradePipeline(roster=load_roster("class.xlsx"))
    result = pipe.process(open("sheet.jpg", "rb").read(), "sheet.jpg")   # result["status"] in verified | needs_review | duplicate

Anything uncertain (could not align, unreadable/complex marks row, marks outside Q1-10, reg no. not on roster, failed
validation) is stored with status needs_review and reasons; only clean sheets write to students/marks.
The name reader is not built yet: the name comes from the roster when the register number matches, else it is empty.
"""
import json
import re
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))   # train_marks (MarksNet, CTC decode) and finetune_header (TrOCR helpers) live in scripts/

from grade_eval import roster as roster_mod
from grade_eval.exam_config import ExamConfig, QuestionLimit
from grade_eval.store import db
from grade_eval.validate import REG_NO_RE, validate_sheet
from grade_eval.vision import cells, header, layout
from grade_eval.vision.preprocess import load_image

MARK_CONF = 0.90        # per-row confidence needed to accept a marks read
MAX_Q_MARK = 30         # larger reads are almost certainly garbage
EXTRA_INK = 0.004       # ink in groups 2-4 (Q11+) above this means marks the reader does not cover
BLANK_INK = 0.003       # ink fraction of the a-column below which a row is blank


def default_config(max_mark: float = 100) -> ExamConfig:
    """Used when no exams/<COURSE>.yaml exists: only a sanity ceiling per question."""
    return ExamConfig("DEFAULT", None, {q: QuestionLimit(max=max_mark) for q in range(1, 11)})


def _parse_mark(s: str) -> float | None:
    if not re.fullmatch(r"\d{1,2}(\.\d)?", s):
        return None
    v = float(s)
    return v if v <= MAX_Q_MARK else None


class GradePipeline:
    def __init__(self, db_path=db.DEFAULT_DB, roster=None, config: ExamConfig | None = None, device: str | None = None,
                 save_aligned: Path | None = None):
        from transformers import VisionEncoderDecoderModel
        import finetune_header as fh
        from train_marks import MarksNet

        self.dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.fh = fh
        fh.DEV = self.dev
        reg_path = ROOT / "models" / "trocr_reg_small"
        self.tok = fh.load_tokenizer(str(reg_path))
        self.reg_model = VisionEncoderDecoderModel.from_pretrained(reg_path, torch_dtype=torch.float32).to(self.dev).eval()
        self.marks_net = MarksNet(pretrained=False)
        self.marks_net.load_state_dict(torch.load(ROOT / "models" / "marks.pt", map_location="cpu"))
        self.marks_net.eval()
        self.conn = db.connect(db_path)
        self.roster = roster or []
        self.config = config or default_config()
        self.save_aligned = save_aligned

    # --- readers -------------------------------------------------------------------------------------------------
    def read_reg(self, aligned: np.ndarray) -> str:
        strip = torch.tensor(header.ink_strip(aligned)).unsqueeze(0).unsqueeze(0)
        return self.fh.decode(self.reg_model, self.tok, strip.to(self.dev), "reg")[0]

    def read_marks(self, aligned: np.ndarray):
        """-> (questions {q: {"a": float}}, sub_total|None, confidences, reasons)."""
        from train_marks import decode as ctc_decode
        mask = cells.ink_mask(aligned)
        X = torch.tensor(np.stack([cells.row_strip(mask, i) for i in range(11)]).astype(np.float32) / 255).unsqueeze(1)
        with torch.no_grad():
            reads = ctc_decode(self.marks_net(X))
        questions, confs, reasons = {}, [], []
        for i in range(10):
            ink_a = cells.ink_fraction(mask, i, cells.A_X)
            s, conf = reads[i]
            simple = cells.is_simple(mask, i)
            if simple and (ink_a < BLANK_INK or (s == "" and conf >= MARK_CONF)):
                continue                                   # blank row: question not attempted / not marked
            if not simple:
                reasons.append(f"Q{i + 1}: written sum or a|b entry (not read automatically)")
                continue
            val = _parse_mark(s)
            if val is None or conf < MARK_CONF:
                reasons.append(f"Q{i + 1}: unclear read '{s}' (conf {conf:.2f})")
                continue
            questions[i + 1] = {"a": val, "b": None}
            confs.append(conf)
        sub = None
        s, conf = reads[10]
        if s and conf >= MARK_CONF:
            if re.fullmatch(r"\d{1,3}(\.\d)?", s):
                sub = float(s)
        # marks written in groups 2-4 (Q11-40) are outside what the reader covers
        for gx in layout.GROUP_X[1:]:
            for i in range(10):
                x0, y0, x1, y1 = gx[1], layout.ROW_Y[i] + 6, gx[3], layout.ROW_Y[i + 1] - 6
                if mask[y0:y1, x0:x1].mean() > EXTRA_INK:
                    reasons.append("marks written beyond Q10 (not read)")
                    return questions, sub, confs, reasons
        return questions, sub, confs, reasons

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
        questions, sub_total, confs, reasons = self.read_marks(aligned)
        reg_no, name = reg_raw, ""

        if self.roster:
            m = roster_mod.match_sheet(reg_raw, "", self.roster)
            if m.row is not None and m.status != "review":
                reg_no, name = m.row.reg_no, m.row.name
                if reg_no != reg_raw:
                    reasons.append(f"register no. read '{reg_raw}', snapped to roster {reg_no}")
            else:
                reasons.append("register no. not matched to roster: " + "; ".join(m.reasons))
        elif not REG_NO_RE.match(reg_raw):
            pass  # validate_sheet reports it

        reasons += validate_sheet(reg_no, questions, sub_total, self.config, confs)
        out.update(reg_no=reg_no, name=name, questions=questions, sub_total=sub_total, reasons=reasons, reg_raw=reg_raw)
        extracted = {k: out[k] for k in ("file", "reg_raw", "reg_no", "name", "sub_total")}
        extracted["questions"] = {str(q): v for q, v in questions.items()}
        out["status"] = db.save_sheet(self.conn, h, reg_no, name, self.config.course_code, questions, reasons, extracted)
        return out
