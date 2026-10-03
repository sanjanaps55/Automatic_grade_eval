import re

from .exam_config import ExamConfig

REG_NO_RE = re.compile(r"^\d{2}[A-Z]{3}\d{4}$")
MIN_CONFIDENCE = 0.80

# Characters OCR commonly confuses, keyed by whether the slot should be a digit or a letter.
_TO_DIGIT = {"O": "0", "I": "1", "L": "1", "S": "5", "B": "8", "Z": "2", "G": "6"}
_TO_LETTER = {v: k for k, v in {"O": "0", "I": "1", "S": "5", "B": "8", "Z": "2"}.items()}


def repair_reg_no(raw: str) -> str:
    """Fix OCR confusions using the fixed format: 2 digits, 3 letters, 4 digits."""
    s = re.sub(r"\s+", "", raw).upper()
    if len(s) != 9:
        return s
    out = []
    for i, ch in enumerate(s):
        if 2 <= i < 5:
            out.append(_TO_LETTER.get(ch, ch))
        else:
            out.append(_TO_DIGIT.get(ch, ch))
    return "".join(out)


def validate_sheet(
    reg_no: str,
    questions: dict[int, dict],
    sub_total: float | None,
    config: ExamConfig | None,
    confidences: list[float] | None = None,
) -> list[str]:
    """Return a list of reasons the sheet needs manual review (empty list = clean).

    `questions` maps q_no -> {"a": float|None, "b": float|None}; the question mark is a + b.
    """
    reasons = []
    if not REG_NO_RE.match(reg_no):
        reasons.append(f"register number '{reg_no}' does not match pattern")
    if confidences and min(confidences) < MIN_CONFIDENCE:
        reasons.append("low recognition confidence")

    q_totals = {q: (p.get("a") or 0) + (p.get("b") or 0) for q, p in questions.items()}
    total = sum(q_totals.values())

    if config is None:
        reasons.append("no exam config")
    else:
        for q_no, parts in questions.items():
            limit = config.questions.get(q_no)
            if limit is None:
                reasons.append(f"Q{q_no} not in exam config")
                continue
            if q_totals[q_no] > limit.max:
                reasons.append(f"Q{q_no} marks {q_totals[q_no]} exceed max {limit.max}")
            if limit.a_max is not None and (parts.get("a") or 0) > limit.a_max:
                reasons.append(f"Q{q_no} part a exceeds max {limit.a_max}")
            if limit.b_max is not None and (parts.get("b") or 0) > limit.b_max:
                reasons.append(f"Q{q_no} part b exceeds max {limit.b_max}")
        if config.total_max is not None and total > config.total_max:
            reasons.append(f"total {total} exceeds max {config.total_max}")

    if sub_total is not None and sub_total != total:
        reasons.append(f"sub-total {sub_total} != sum of questions {total}")
    return reasons
