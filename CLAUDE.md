# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Start here
Read `PLAN.md` first: it holds the roadmap, model decisions and current status. At the end of each working session, update its Status section and tick off finished steps so context is not lost.

## Project
Local pipeline that turns phone photos of graded VIT answer sheets into an Excel file of register no., name, per-question marks (a + b summed) and total. Package: `grade_eval/` (`vision/`, `store/`, `validate.py`, `exam_config.py`). See `README.md`.

## Commands
- Install: `pip install -r requirements.txt`
- Tests: `pytest tests/` (single test: `pytest tests/test_core.py::test_excel_export`)
- Dataset: `python scripts/build_cells.py` (aligns `data/labels_all.csv` sheets -> `data/cells.npz`). Train marks reader: `python scripts/train_marks.py --epochs 25` (CPU ~20 min). Labelling panels: `scripts/make_label_sheet.py`; alignment check: `scripts/check_alignment.py`.
- No pipeline / server commands yet.

## Data
`data/` is git-ignored: `data/samples/sheet_NNN.jpeg` (photos), `data/labels.csv` (hand labels, file column uses the sheet_NNN names), `data/rename_map.csv` (old -> new names).
