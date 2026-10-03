"""Per-exam max marks, typed once by the user and keyed by course code.

File: exams/<COURSE_CODE>_<EXAM>.yaml, e.g. exams/MCSE602L_CAT1.yaml

    course_code: MCSE602L
    total_max: 50
    questions:
      1: {max: 10}
      2: {a_max: 5, b_max: 5}   # question with parts; max defaults to a_max + b_max
"""
from dataclasses import dataclass, field
from pathlib import Path

import yaml

EXAMS_DIR = Path(__file__).resolve().parent.parent / "exams"


@dataclass
class QuestionLimit:
    max: float
    a_max: float | None = None
    b_max: float | None = None


@dataclass
class ExamConfig:
    course_code: str
    total_max: float | None
    questions: dict[int, QuestionLimit] = field(default_factory=dict)


def parse_config(data: dict) -> ExamConfig:
    questions = {}
    for q_no, spec in (data.get("questions") or {}).items():
        a_max, b_max = spec.get("a_max"), spec.get("b_max")
        q_max = spec.get("max")
        if q_max is None:
            q_max = (a_max or 0) + (b_max or 0)
        questions[int(q_no)] = QuestionLimit(max=q_max, a_max=a_max, b_max=b_max)
    return ExamConfig(
        course_code=str(data["course_code"]).upper(),
        total_max=data.get("total_max"),
        questions=questions,
    )


def load_config(course_code: str, exams_dir: Path = EXAMS_DIR) -> ExamConfig | None:
    """Return the config for a course code, or None if the user hasn't added one."""
    code = course_code.strip().upper()
    for path in sorted(exams_dir.glob(f"{code}*.yaml")):
        return parse_config(yaml.safe_load(path.read_text(encoding="utf-8")))
    return None
