# Automatic_grade_eval

Reads graded VIT answer-sheet cover pages photographed with a phone and writes the marks to Excel, locally.

**Extracted fields:** register number, student name, marks per question (columns a and b added together), total marks.

## Pipeline
Phone photo -> upload to local system -> align to template -> read handwriting -> validate + duplicate check -> SQLite -> `marks.xlsx`.
Anything uncertain is held for manual review instead of being written to Excel.

## Layout
- `grade_eval/vision/` - image loading, alignment to the form template, cell cropping
- `grade_eval/validate.py` - register-number format, max-mark and sub-total checks
- `grade_eval/store/` - SQLite storage (one record per student, duplicate-image detection) and Excel export
- `grade_eval/exam_config.py` - per-exam max marks from `exams/*.yaml`
- `data/samples/` - sheet photos (`sheet_001.jpeg` ...); `data/labels.csv` - hand-labelled marks (both git-ignored)
- `tests/` - unit tests

## Setup and tests
```
pip install -r requirements.txt
pytest tests/
```

See `PLAN.md` for the roadmap, model choices and current status.
