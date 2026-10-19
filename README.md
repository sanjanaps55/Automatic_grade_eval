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
pip install sentencepiece        # needed by the TrOCR-small tokenizer
pytest tests/
```

## Models (weights are not in git)
`models/`, `data/` and `dist/` are git-ignored; weights are shared as a zip (GitHub Release / Drive link: ask the owner).
1. Unzip `grade_eval_models.zip` into the project root so the files land in `models/`. Check the sha256 values in `MANIFEST.txt`.
2. What is inside:

| Job | File | Status |
|---|---|---|
| Marks per question (simple rows) | `models/marks_sum.pt` (ResNet18 + CTC, reads sums) | in use |
| Marks per question (hard rows: a\|b entries, written sums) | `models/hard_b_s0.pt` (same net + position channels) | in use; ~62% vs 46% exact on held-out hard rows |
| Register number | `models/trocr_reg_small/` (TrOCR-small) | in use; val 84% / test 75% exact, ~1 s/strip CPU |
| Older models | `models/not_used_models/` | not loaded by the app, kept for reference; not in the shared zip |
| Name | `models/trocr_name_small/` | weak free-text reader (8% exact): used only as a hint when matching the register number to the class list |

3. Check the register-number model loads and reproduces the numbers (needs `data/`): `python scripts/eval_header.py --field reg`.

The sheet photos, labels and class roster (`data/`) are private and are not shared with the weights. Ask the owner for them, or use your own sheets of the same VIT cover form.

## How to work on it
**Data pipeline** (needs `data/samples/`, `data/labels_all.csv`):
1. `python scripts/make_split.py` - one train/val/test split by student (`grade_eval/split.py`). Every model uses it, so never re-split by hand.
2. `python scripts/build_cells.py` - aligns the sheets and writes the marks row strips (`data/cells.npz`).
3. Marks reader: `python scripts/train_marks.py --epochs 25` (CPU ~20 min) -> `models/marks.pt`.
4. Header readers (register no. / name): `python scripts/export_reg_data.py`, `python scripts/export_name_data.py`, then train on a GPU (Kaggle/Colab): `python -u scripts/finetune_header.py --field reg|name --model microsoft/trocr-small-handwritten --save`. Kaggle/Colab notebooks: `kaggle_header_bakeoff.ipynb`. It prints train-vs-val gap each epoch, early-stops on val, and scores the test set once at the end - do not tune on test.
5. PaddleOCR PP-OCRv5 mobile: `python scripts/prep_paddle_data.py` builds its data; training steps are in `PLAN.md` / the Kaggle notebook. Install Paddle in its own session/venv (it breaks Kaggle's torch, and on Windows use a short venv path like `C:\pdl_env`).
6. Class roster matching: `grade_eval/roster.py` (csv/xlsx with register-number and name columns). `python scripts/dump_header_preds.py` writes each model's predictions; `python scripts/score_roster.py [preds.csv] [roster]` scores auto-correct / auto-wrong / review (add `--require-name` for the safe mode).
7. To share updated weights: `python scripts/package_models.py [--with-resnet-reg]` -> `dist/grade_eval_models.zip` (upload as a Release asset, never commit).

**Rules of the project**
- Uncertain reads go to manual review, never straight into Excel. The number that matters most is *auto-accepted wrong*, keep it ~0.
- Judge models by roster-matching results (wrong / review counts), not only character error rate. The held-out sets are small (25 val, 12 test sheets), so differences of a few points are noise.
- Update the Status section in `PLAN.md` at the end of each session.

## What is left
- **Name reader** (main open problem): fine-tune PP-OCRv5 mobile on Kaggle and score it with `score_roster.py`; also try scoring roster names by TrOCR likelihood and training on synthetic roster-name strips.
- **Final pick** for register number (TrOCR-small vs Paddle mobile) once Paddle is scored the same way.
- **Marks**: retrain on the 3-way split; a|b columns and written sums (e.g. `3+2+1+2`) are not read yet (sent to review); 8 curled sheets fail to align.
- **Pipeline** (`grade_eval/pipeline.py`): hash -> dedupe -> align -> read -> roster match -> validate -> save to SQLite.
- **FastAPI server** with a phone photo-upload page, and a **review UI** for sheets held for review (`db.approve_sheet`), Excel via `export_excel`.
- **Exam YAMLs** (per-question max marks, a/b limits) and the real class roster.
- **~30-50 fresh sheets** from different students for a true unseen test, and more labelled name strips.

See `PLAN.md` for the roadmap, results of every experiment and the current status.

## Using it (teacher guide)
1. Double-click `start.bat`, open the address it prints (PC: `http://localhost:8000`; phone on the same Wi-Fi: the IP shown).
2. **Start a new set**: give it a name, upload the class Excel/CSV (columns: register number, name), set the number of questions.
3. **Upload photos** of the graded cover pages (camera or gallery, many at once).
4. Each sheet becomes one of three tiers:
   - **confirmed**: the question marks add up to the written sub-total (or are clear single numbers) and the register number matches the class list. Written to the Excel.
   - **to check**: read fine but nothing to cross-check it against. Open it, glance at the photo next to the values, press *Save as confirmed* (or *Confirm all* when you trust the batch).
   - **needs fix**: something unclear. Correct the boxes by hand and save.
5. **Download Excel**: your class list in your order with the marks in each student's row; absent students stay blank; a second sheet lists what still needs attention.

Written sums such as `3+5+1` and a|b entries are read and added; the raw read is shown on hover. Nothing is guessed: unclear rows are held back.

## Developer notes (branch `naveen`)
- `grade_eval/pipeline.py`: photo -> align -> reg no. (TrOCR) + marks (ResNet+CTC with `+`) -> roster match -> validate -> SQLite.
- `grade_eval/server.py`: the web app (sets under `data/sets/<name>/`). `scripts/run_pipeline.py`: batch CLI.
- `scripts/train_marks_sum.py` trains the sum-capable reader -> `models/marks_sum.pt` (needs `data/cells.npz`, `data/label_work/records.txt`).
- `scripts/score_pipeline.py` scores the readers on `data/labels_all.csv` (row accuracy, auto-wrong, sheet-level acceptance).

## Running the pipeline (branch `naveen`)
Weights go in `models/` (see above). Then:
```
# batch over a folder -> data/grades.db + marks.xlsx (review items listed with reasons, not written to Excel)
python scripts/run_pipeline.py data/samples_new [--roster class.xlsx] [--db data/grades.db] [--xlsx marks.xlsx]

# phone upload + review page (open http://<pc-ip>:8000 on the phone, same Wi-Fi)
set ROSTER=class.xlsx            # optional
python -m uvicorn grade_eval.server:app --host 0.0.0.0 --port 8000
```
Reads register no. (TrOCR) and Q1-Q10 marks (ResNet+CTC). Rows with written sums / a|b, low confidence, marks beyond Q10 or an
unaligned photo go to the `/review` page. The name reader is not built: the name comes from the roster when matched.
