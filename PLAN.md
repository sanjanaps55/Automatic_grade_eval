# PLAN.md — Automatic grade evaluation

Living plan. **Update the Status section at the end of every working session** so context is not lost.

Extract only: **register number, name, marks per question (a + b summed), total.**

## Context
Evaluators copy name, register no. and question-wise marks from graded VIT answer sheets into Excel by hand. This system does it locally: phone photo -> alignment -> handwriting recognition (CNN) -> validation / duplicate check -> SQLite -> Excel. Uncertain results go to manual review, never straight into Excel. Model choice is driven by the real dataset (`data/samples`), then fine-tuned.

## Status (2026-10-11)
- Done + tested (`tests/test_core.py`): `grade_eval/exam_config.py`, `validate.py`, `store/db.py`, `store/excel.py`.
- Written: `vision/preprocess.py`, `vision/layout.py`, `vision/make_template.py` (+ `assets/template.png`). `layout.align` verified on real data, see step 1.
- Missing: recognizer, pipeline, FastAPI phone upload, review UI, roster matching, exam YAML.
- Data: `data/samples/` = 163 images `sheet_001..165.jpeg` (old names in `data/rename_map.csv`; 2 duplicate photos in `data/removed_duplicates/`). `data/labels.csv` = 36 labelled sheets (per-question totals q1..q10, `sub_total` still empty). ~127 unlabelled sheets are different students. `data/` is git-ignored.

## Session note (2026-10-11, later)
- Removed 20 re-photos (same reg_no, near pixel-identical) from `data/samples/` -> `data/removed_duplicates/`; `labels_draft.csv` now 74 rows (backup `.bak`), samples 143. Kept the sheet already in labels.csv, else lowest number.
- Reg no format: 2 digits + 3 capital letters + 4 digits. Draft rows 056, 059, 102, 164 had truncated reg nos (still `?`).
- Next: user uploads any missing sheets to `data/samples/` as `sheet_166.jpeg`+; label sheets with no label (002-043 not in labels.csv, 049); skip re-photos; then merge draft into labels.csv.

- 2026-10-11 (later): user's `missing_samples/` upload = same 50 sheets (001-050) at higher res (~2300 px wide). Swapped into `data/samples/` (low-res backups in `data/samples_lowres/`, originals in `data/missing_samples_orig/`). Labelled the 33 previously unlabelled sheets into `labels_draft.csv` (now 105 rows, `?` for unsure; header strips re-read from wider crops). Removed re-photos 004 (=005), 034 (=033). Same reg_no, different marks (probably different exams): sheet_007 vs 139, sheet_039 vs 001 -> decide how DB keys exams (one row per reg_no per exam?). sheet_042 still unaligned (reg no unsure).
- Panels for the new rows: `data/label_panels_new/`.

- 2026-10-11 (end): user cleared all `?`. Built `data/labels_all.csv` = labels.csv (36) + labels_draft.csv (104) = **140 labelled sheets**, every reg_no matches `^\d{2}[A-Z]{3}\d{4}$`, names capitalised, no `?`. Removed re-photo sheet_164 (=092); samples now 140 = all labelled. Same student on two sheets with different marks (23BAI1422: 007/139, 24MCS1074: 001/039) is fine: different exams, extraction only. 4 sheets whose written sub-total != sum of questions are flagged in `needs_check` (049, 120, 127, 132): recheck or treat as teacher arithmetic errors. Next: step 3 (`cells.py`), split train/val by student.

## Session note (2026-10-12): digits + header
- Marks reader retrained with synthetic 2-digit strips (real single digits pasted together, train students only): held-out filled rows exact 97.4% (was 85.5%), confident-wrong at conf>=0.9: 1 (was 6). Blank rows 97.8%. a|b and written-sum rows still go to review. Weights `models/marks.pt` (git-ignored).
- Register no.: zero-shot TrOCR-base scored 0/132 (reads whole lines, ignores format). Local CPU CTC attempt (`scripts/train_reg.py`, synthetic fonts then real fine-tune) collapsed to a constant guess: too few steps on CPU (~2.5 min/epoch). Next: train on a Colab GPU. `colab_reg.zip` (code + `data/reg_synth.npz` 30k synthetic strips rendered with Windows handwriting fonts + `data/reg_real.npz` 132 real strips). Colab: GPU runtime, upload zip, `!unzip -q colab_reg.zip`, `!python -u scripts/train_reg.py --npz --syn-epochs 40 --ft-epochs 60`, download `models/reg.pt`. Then add grammar-constrained decoding (2 digits + 3 letters + 4 digits) and roster snapping; name reader is secondary.
- Live capture (not trained, classical): align every frame, auto-capture when aligned + steady + sharp, reject otherwise; to do with the server (step 7).

## Session note (2026-10-14): header-reader plan
- Decision: user has no GPU, names are cursive -> bake-off of fine-tuned **TrOCR-small** vs **PaddleOCR PP-OCRv5 rec** (TrOCR-base = accuracy reference: val exact 92%, test 75% n=12, 12 s/strip CPU). Train on Kaggle/Colab; inference CPU. Full plan: `~/.claude/plans/okay-so-i-need-velvet-iverson.md` (roster joint reg+name rule, Excel gets roster values, unseen-sheet test, 5-fold CV).
- Done: `grade_eval/split.py` = single train/val/test split by student (101/25/14 sheets, 138 students), `scripts/make_split.py` -> `data/split.csv`; `build_cells.py` (cells.npz now has `split`) and `train_reg.py` use it. `tests/test_split.py`.
- Next: name strips in `header.py` + export; `--field reg|name` in `train_trocr.py`; Paddle rec fine-tune; `grade_eval/roster.py`; overfit report (train vs val gap) in training scripts; need ~30-50 fresh sheets for the unseen test. Re-run `scripts/build_cells.py` to refresh `cells.npz` with the `split` column (marks model not yet retrained on 3-way split).

- 2026-10-14 (later): added `header.name_strip` (`NAME_CROP` past the "Mr/Ms" label, before "Course Title"), `scripts/export_name_data.py` -> `data/name_real.npz` (132 strips: 95/25/12 train/val/test) + `data/name_preview.png`, and `scripts/finetune_header.py --field reg|name --model <trocr-small|base>` (shared student split, frozen encoder first epochs, early stopping on val CER, best-on-val weights, train-vs-val gap each epoch, test scored once). Smoke-tested on CPU. Needs `pip install sentencepiece` (small model tokenizer). Still to do: run on Kaggle/Colab GPU, Paddle rec fine-tune, compare.

### Pending / to figure out tomorrow
- Kaggle bake-off run (`kaggle_header_bakeoff.ipynb` + `kaggle_train.zip`, both git-untracked/rebuilt as needed): data loaded fine, but `finetune_header.py` failed at tokenizer load on Kaggle's newer transformers (needs `sentencepiece`/`protobuf`; patched to try `XLMRobertaTokenizer` first, not yet re-run on Kaggle). Re-upload zip or apply the patch cell, restart kernel, re-run.
- Run TrOCR-small on reg + name, then PaddleOCR PP-OCRv5 rec on both (Paddle install / config / pretrained-weights URL are untested). Compare on val CER, train-vs-val gap, CPU speed; test set (12 sheets) is only a sanity check.
- Then: `grade_eval/roster.py` (joint reg+name snapping), unseen-sheet test (~30-50 fresh sheets), retrain marks model on the 3-way split.

## Dataset findings
- One fixed VIT cover form: 4 groups x 10 question rows (a|b) + sub-total row, printed in blue.
- Marks are written in red/green pen, grid is blue, header handwriting blue/black -> separable by colour.
- Mostly column a, sometimes a and b (add them). Values 0-14, multi-digit, some x.5.
- Hard cases: crossed-out / overwritten / circled marks, strokes across cells.
- Phone photos: tilt, curl, clutter, 960x1280 up to 3024x4032, WhatsApp-compressed. Header is handwritten.

## Models
1. Page/grid alignment: OpenCV SIFT + homography (classical, written).
2. Ink separation: HSV colour threshold (classical).
3. **Marks cells: pretrained small CNN backbone (ResNet18/MobileNet) + CTC head, fine-tuned, not trained from scratch** (~160 sheets is too little). Warm up on synthetic EMNIST-digit strings with pen styling, then fine-tune on real labelled cells; hold out whole students. "Unreadable / crossed-out" output routes to review. Compare against fine-tuned TrOCR-small-handwritten and zero-shot TrOCR; keep the best on held-out students.
4. Register no. / name: pretrained TrOCR + snap to class roster (CSV from user).
5. Validation: `validate.py` rules (a+b, per-question max, sub-total check).

## Steps
0. [x] Write `PLAN.md` + `README.md`, point `CLAUDE.md` here.
1. [x] `scripts/check_alignment.py` (run 2026-10-11): 155/163 aligned (95%); inliers min 35, median 113. Overlays in `data/alignment_check/` (git-ignored). **Overlay images were saved but NOT actually viewed (image reads failed on request limit), so "cells land correctly" is UNVERIFIED; open `data/alignment_check/` and check by eye.** The label panels (steps below) did show the aligned table crops looking right for sheets 042-165. Findings:
   - **8 failures** (sheet_042, 093, 113, 116, 117, 118, 119, 120): curled pages (batch 113-120, course MCSE502) or small page with clutter. Tried `preprocess.find_grid_corners` as fallback: wrong corners on curled pages (stretched grid), not adopted. Pipeline sends these to review ("could not align").
   - **Curled pages that do align** (e.g. sheet_108, 114, 088) drift a few px toward right/bottom columns; fine for top rows, may hurt sub-total row. Possible later fix: refine each cell box by snapping to detected grid lines, or a local re-align.
   - `REGISTER_BOX` / `NAME_BOX` include the printed label ("Register No", "Mr/Ms"); tighten x0 past the label before header OCR.
2. [~] Labelling helper `scripts/make_label_sheet.py` (panels in `data/label_panels/`, 2 sheets/image). **Draft labels: `data/labels_draft.csv`** = Claude's reads of 95 sheets (sheet_044..165, read from panels actually viewed); unsure cells are `?` (no guessed values), unsure reg_no/name end with `?`. `needs_check` column lists why a row needs the user's recheck (unsure digits, cut-off names, struck-out marks, sum != written sub-total, re-photo of an earlier student). Per-question values are a+b summed (and written sums like 3+2+1+2 added); original a|b/x+y notation is in `data/label_work/records.txt`. NOT merged into `labels.csv` (user's verified file untouched).
   - **Still unlabelled: 32 sheets (sheet_002..043)**; their panels (000-015) would not load in the last session, so no labels were written for them (earlier guesses were deleted). Retry when images load.
   - Many sheets 144-165 are re-photos of earlier students (same marks): decide whether to dedupe.
   - Header crop cuts off long names ("Prashant Kum"): widen `NAME_BOX` before header OCR.
   - Per-cell a/b labelling is not done; drafts only have per-question totals.
3. [x] `grade_eval/vision/cells.py` + `scripts/build_cells.py`: aligned image (cached in `data/aligned/`) -> Lab a* ink mask (the HSV-saturation mask missed pale pen) -> 64x192 ink-only row strips + labels in `data/cells.npz`; val = 20% of students by reg_no hash. 132/140 sheets align (8 skipped: 042,093,113,116-120). 661 filled rows, 404 'simple' (mark in column a only); the rest (a|b, written sums like 3+3+1+2) have no per-cell labels, so they are excluded from training and must go to review for now.
4. [~] `scripts/train_marks.py` (ResNet18 ImageNet + CTC, CPU ~20 min, weights `models/marks.pt`). First result on 24 held-out students (simple rows): filled rows exact 85.5% (76), blank rows correct 99.3% (136); 6 confident-wrong at conf>=0.9, mostly '11'->'1' (x3), '25'->'5', '31'->'3': repeated/adjacent digits collapse in CTC. To do: more width resolution / synthetic multi-digit strings, handle a|b + sums (need per-cell labels or a sum parser), `scripts/eval_recognizer.py`, compare TrOCR.
5. [ ] `grade_eval/roster.py` + tests.
6. [ ] `grade_eval/pipeline.py`: hash -> dedupe -> load -> align -> recognize -> roster match -> validate -> save.
7. [ ] FastAPI `server.py`: `/upload`, mobile capture page over local Wi-Fi.
8. [ ] Review UI for `needs_review` (`db.approve_sheet`); Excel via `export_excel`.
9. [ ] Exam YAMLs (MCSE602L, BCSE209L), docs, commits.

## Verification
- `pytest tests/` passes (plus roster / pipeline tests).
- >=95% of images align; overlays correct.
- Held-out students: per-cell accuracy, per-sheet exact match, near-zero confident-wrong reads.
- End to end from phone: verified / needs_review; same photo -> duplicate; re-photo -> one row per reg_no; `marks.xlsx` correct.

## Open items for the user
- Roster file (reg_no -> name).
- Exam course codes, per-question max marks and a/b limits.
- Label remaining sheets.
- How often column b is used.
