# PLAN.md — Automatic grade evaluation

Living plan. **Update the Status section at the end of every working session** so context is not lost.

Extract only: **register number, name, marks per question (a + b summed), total.**

## Context
Evaluators copy name, register no. and question-wise marks from graded VIT answer sheets into Excel by hand. This system does it locally: phone photo -> alignment -> handwriting recognition (CNN) -> validation / duplicate check -> SQLite -> Excel. Uncertain results go to manual review, never straight into Excel. Model choice is driven by the real dataset (`data/samples`), then fine-tuned.

## Status (2026-10-15)
- **Done + tested** (`pytest tests/`, 16 pass): `exam_config.py`, `validate.py`, `store/db.py`, `store/excel.py`, `split.py` (one train/val/test split by student, 101/25/14 sheets), `roster.py` (joint reg-no + name matching against the uploaded class list; csv/xlsx).
- **Done, no tests**: alignment (`vision/layout.py`, 95% of photos), cell extraction + marks reader (`scripts/build_cells.py`, `train_marks.py`), header strips for reg no. and name (`vision/header.py`), TrOCR fine-tune (`scripts/finetune_header.py`), Paddle data prep, prediction dump + roster scoring.
- **Model decisions so far** (provisional until Paddle bake-off finishes):
  | Job | Model | Status |
  |---|---|---|
  | Marks per question | ResNet18 + CTC (`models/marks.pt`, 43 MB) | keep; 97.4% exact on held-out filled rows. Retrain on the 3-way split still to do |
  | Register no. | TrOCR-small fine-tuned (`models/trocr_reg_small`, 122 MB) | **chosen so far**: val 84% / test 75% exact, CER 0.02-0.07, ~1 s/strip CPU. Roster matching fixes most slips. Only replace if Paddle mobile beats it |
  | Name | none yet | TrOCR-small name = unusable (val 8% exact). Candidates: PP-OCRv5 mobile fine-tune (Kaggle run pending, zero-shot 0.06-0.2 s/strip CPU), closed-set scoring of roster names by TrOCR likelihood, synthetic roster-name training data |
  | Class list | `roster.py` | built; auto-accept only when reg + name agree, else review |
  Dropped: TrOCR-base (12 s/strip CPU), Paddle server rec (14-29 s/strip CPU), old ResNet reg readers (`models/reg*.pt`) and the 640 MB `models/trocr_reg` base checkpoint: reference only.
- **Missing**: pipeline, FastAPI phone upload, review UI, name reader, exam YAML, ~30-50 fresh sheets for the unseen test.
- Data: `data/samples/` = 163 images `sheet_001..165.jpeg` (old names in `data/rename_map.csv`; 2 duplicate photos in `data/removed_duplicates/`). `data/labels.csv` = 36 labelled sheets (per-question totals q1..q10, `sub_total` still empty). ~127 unlabelled sheets are different students. `data/` is git-ignored.

## Status (2026-10-16, branch `naveen`)
- **Built**: `grade_eval/pipeline.py` (photo -> read -> match -> validate -> SQLite), `grade_eval/server.py` (sets, class-list upload, number of questions, results table, per-sheet fix page, confirm-all, Excel in class-list order), `scripts/run_pipeline.py`, `start.bat`, `scripts/score_pipeline.py`, `scripts/train_marks_sum.py` (`models/marks_sum.pt`, reads `3+5+1` / a|b as sums). 19 tests pass.
- **Baseline vs now** on the 140 labelled sheets (Q1-Q10 rows, `score_pipeline.py --split all`): old reader flagged 338/1320 rows (255 were written sums) and only 14 sheets had no flag; sum reader flags 196, 33 sheets unflagged, but 17 rows auto-read wrong (was 3) because the model is overconfident when it misreads a sum or a tick.
- **Safety design**: model confidence is NOT trusted (wrong reads at 0.95-1.0). A sheet is auto-written only when the question marks match the written sub-total (or clear single numbers >= 0.97) AND the register number matches the class list exactly. Everything else is "to check" (glance + save) or "needs fix". Only 4/132 sheets carry a matching written sub-total, so most sheets land in "to check" by design.
- **Open**: more sum/a|b training rows (46 real), ticks/strikes read as digits, name reader (see below), 8 curled sheets fail to align, Paddle bake-off not run.

## Status (2026-10-20, branch `overnight`): see OVERNIGHT_REPORT.md
- Added: reader ensemble (marks_sum + hard_b averaged), written-total cross-check by likelihood ratio (`grade_eval/totals.py`, `models/total_sub.pt`, `total_grand.pt`; total locations labelled in `data/overnight/total_loc.csv`), unique-exact register rule, alignment upscaling for small/compressed photos, pre-filled row guesses, upload-page quality warning, corrections log (`data/corrections/`), end-to-end scorer `scripts/eval_e2e.py`.
- Held-out (37 sheets): confirmed 5 -> 6 (orig), 3 -> 4 (q85), 0 -> 1 (q60); confirmed-wrong 0 (was 1 at q85). Most sheets still need fixing: sum rows without a readable total stay unconfirmed by design. Next best lever: more labelled totals / sum rows (use the corrections log) and a fresh 30-50 sheet test set.

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

### TrOCR-small results (Kaggle 2026-10-15; models in `models/trocr_{reg,name}_small`, logs in `data/kaggle_logs/`, `scripts/eval_header.py` re-scores on CPU)
- **Register no.**: val CER 0.022, exact 84% (n=25); TEST CER 0.065, exact 75% (n=12); gap +0.02 (no overfit). ~1.1 s/strip CPU. Errors are single-digit slips (1137 vs 1137->BRS, 1072->1672), so roster snapping should fix most.
- **Name**: val CER 0.585, exact 8%; TEST CER 0.70, exact 0%; gap +0.16..0.43, early-stopped at ep 16; many empty outputs; 2.5-4.5 s/strip. Not usable as a free-text reader: only 95 train strips. Use as a roster-matching hint at best; next: Paddle, more name data, or sheet-level roster match from reg no. alone.
- Paddle: pretrained URL is `paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv5_mobile_rec_pretrained.pdparams`; installing paddlepaddle-gpu breaks Kaggle's torch (NCCL), so run Paddle in its own session; set FLAGS_allocator_strategy=auto_growth.

### Roster matching (2026-10-15)
- `grade_eval/roster.py` (`load_roster` csv/xlsx, `match_sheet`: reg edit distance + name similarity vote for one roster row; statuses auto / auto_reg_only / review) + `tests/test_roster.py` (16 tests pass). `scripts/dump_header_preds.py` -> `data/preds_trocr.csv`; `scripts/score_roster.py [preds.csv] [roster] [--require-name]` scores any engine's predictions (same csv columns) so Paddle plugs in.
- TrOCR on val+test (37 sheets, dense 759-row test roster = real students + invented neighbours): reg only auto-correct 30 / auto-WRONG 3 / review 4; reg+name 29 / 2 / 6; `--require-name` (auto only when name agrees) 9 / **0** / 28. Auto-wrong cases = exact hit on a *valid neighbouring* student with an unreadable name. Name reader too weak to confirm, so safe mode sends most sheets to review. Better name reader (Paddle?) is what unlocks both accuracy and safety. Weights not tuned (37 sheets is too few).

### Pending / to figure out tomorrow
- Kaggle bake-off run (`kaggle_header_bakeoff.ipynb` + `kaggle_train.zip`, both git-untracked/rebuilt as needed): data loaded fine, but `finetune_header.py` failed at tokenizer load on Kaggle's newer transformers (needs `sentencepiece`/`protobuf`; patched to try `XLMRobertaTokenizer` first, not yet re-run on Kaggle). Re-upload zip or apply the patch cell, restart kernel, re-run.
- Run TrOCR-small on reg + name, then PaddleOCR PP-OCRv5 rec on both (Paddle install / config / pretrained-weights URL are untested). Compare on val CER, train-vs-val gap, CPU speed; test set (12 sheets) is only a sanity check.
- Then: `grade_eval/roster.py` (joint reg+name snapping), unseen-sheet test (~30-50 fresh sheets), retrain marks model on the 3-way split.

### Sharing the models (weights are NOT in git: `models/`, `data/`, `dist/` are ignored)
- GitHub rejects files >100 MB in a repo, so never commit weights. Run `python scripts/package_models.py` -> `dist/grade_eval_models.zip` (~165 MB: marks.pt + trocr_reg_small, with `MANIFEST.txt` sha256 checksums; `--all` adds the experimental name model).
- Share that zip by: (1) a **GitHub Release** asset on this repo (up to 2 GB per file; Releases page -> Draft a new release -> attach zip), or (2) a private **Hugging Face Hub** model repo, or (3) Google Drive / OneDrive link. Put the link in README.
- Recipient: `git clone`, `pip install -r requirements.txt` (+ `sentencepiece`), unzip into the project root so files land in `models/`, check the sha256 in `MANIFEST.txt`, then `python scripts/eval_header.py --field reg` to confirm it loads. The `data/` folder (photos, labels) is private and not shared.

### Pending / to figure out tomorrow
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

## Status (2026-10-18): live scanning + test scoring
- `grade_eval/live.py` + `assets/live.html` + routes `/s/<set>/live-link`, `/live`, `/live/frame`, `/live/capture`: phone streams frames over https, server answers align / back / blurry / ok, phone auto-captures after 2 ok frames and says NEXT; Start/Stop scanning. `scripts/serve.py` (used by `start.bat`) serves http :8000 + https :8443 (self-signed cert in `data/certs/`, phones only allow the camera on https). Thresholds (`MAX_OUTSIDE`, `MIN_SHARP`) were set from still photos only: **not yet tried with a real phone camera**; the page has a manual "Capture now" fallback.
- Test mode (per set) + accuracy panel against `data/labels_all.csv` (photos must keep `sheet_NNN` names); `data/roster_labelled.xlsx` = 138 students for testing.
- Held-out test split (14 sheets): 12 aligned, reg exact 9/12, 103/120 marks rows auto-correct, 1 auto-wrong (written sum `3+4+8` read as 9). Roster matching on val+test (37): 30 correct, 0 wrong, 7 to review.
- Live scanning rework (tested on synthetic frames / a phone aimed at a monitor, **not yet on real paper**): phone asks for 4K and sends 1800 px frames (the form needs ~900 px of sheet width to match; cheaper SIFT/ORB/AKAZE settings failed); server returns the sheet outline (drawn on the video) and one specific instruction (cut off at left / move closer / tilt / rotated / blurry); full-screen green "NEXT sheet" sign with beep. `/live-test` = framing diagnostics only (nothing saved, last frame kept in `data/live_debug/`), `/live-test/sheet` = sample sheet to point the phone at. Capture is asynchronous (`/live/capture` returns a job id, `/live/job/<id>` is polled) and models load at server start. Timing per sheet on CPU: ~1.7 s without class list, ~2.5 s with (align 0.8, reg reader 0.8, name 0.6, marks 0.3). Idea not done: reuse the live-frame alignment at capture to save ~0.8 s (needs an accuracy check first).
- 2026-10-19 marks reader analysis (held-out val+test, 37 sheets / 370 rows): roster match 35/37 correct, 0 wrong auto. Marks rows 89% exact; single-number rows 98-99%, hard rows (a|b, written sums) only 46% (old `marks.pt` 34%, so reverting is no help). Only 12/37 sheets have no hard rows (avg 1.7/sheet); sheets fully correct: 14/37. Raising/lowering `MARK_CONF` cannot fix it (accepting >=0.5 makes ~half of accepted sheets wrong). Plan: (1) cell-level review (auto-accept simple rows, only hard cells highlighted + pre-filled), (2) retrain for hard rows: Kaggle bake-off `kaggle_marks_bakeoff.ipynb` + `kaggle_marks.zip` (`scripts/package_kaggle_marks.py`): ResNet18-CTC + MNIST glyphs vs TrOCR-small-handwritten (`scripts/train_marks_trocr.py`), both smoke-tested on CPU only; next dataset to add: CROHME/HME100K expressions filtered to digits and `+`; (3) save corrected sheets from the app as new training data; (4) scan real paper. Marks are not yet using any pretrained digit/handwriting model in production (ImageNet ResNet18 only).
- 2026-10-19 hard-row work. Findings: a MNIST-augmented ResNet (`models/marks_sum_mnist.pt`) and TrOCR-small (`models/marks_trocr_ex/`) were NOT better than `marks_sum.pt` on the 37 held-out sheets (hard rows 54% / 51% vs 46%; TrOCR worse on simple rows 96% vs 99%, +2.6 s/sheet). ResNet+TrOCR agreeing on a row = 98% right (322/370 rows; 14/37 sheets agree on every row, 11 of those fully correct) -> candidate fallback/cascade check. Cell-level review alone does not cut the number of sheets reviewed (~27/30 sheets would still have >=1 cell). Real bottleneck = data: only 94 hard rows had written notation, 131 had totals only. Read 89 more hard rows from row crops by eye (`data/label_work/hard_notation.txt`, every one sums to the labelled total; sheet 041/102 skipped: row crops misaligned; scribbled/ambiguous rows skipped) -> 183 labelled hard rows (train 136 / val 33 / test 14; split unchanged = by student). Most "hard" rows are plain numbers with a grader tick stroke across the column line, not real sums.
- New: `scripts/train_marks_hard.py` (specialist; warm start marks_sum.pt, hard rows oversampled x3, realistic synthetic hard rows from TRAIN digits only incl. ticks/scribbles/a|b, feature dropout, early stop on val, train-vs-val gap printed, test scored once, optional `--coord` a|b position channels) + `kaggle_marks_hard.ipynb` + `scripts/package_kaggle_marks.py` (kaggle_marks.zip, 84 MB). Local 30-epoch CPU run -> `models/marks_hard_local.pt`. Next: compare on the held-out sheets with the same harness, then cascade (specialist -> TrOCR/other model agreement) instead of "review". Test split has only 14 hard rows: judge by val+test together and by seed stability; consider grouped CV.

## Status: marks reader routing (latest)
- Marks rows are routed: simple rows -> `models/marks_sum.pt`, hard rows (a|b / sums, `cells.is_simple` false) -> `models/hard_b_s0.pt` (wired in `GradePipeline.read_marks`). Held-out (37 sheets): 92.4% cells exact vs 89.7% before; 19/37 sheets fully right.
- Through the real pipeline on the 37 held-out sheets: 312/370 cells accepted-and-right, 4 silently wrong, the rest flagged for review.
- `grade_eval/marks_solver.py` (sub-total search) is built but NOT wired in: the sub-total cell reads right on only 2/15 held-out sheets (the readers were never trained on sub-total cells). Needs a small sub-total reader (59 labelled + ~21 to label) trained on Kaggle first. `scripts/eval_subtotal.py` re-tests it.
- Unused models moved to `data/not_used_models/`. `scripts/package_models.py` still lists marks.pt / reg*.pt.
- Confidence is now the probability of the read VALUE (`GradePipeline._value_conf`, via `marks_solver.candidates`), not the weakest frame. Held-out through the pipeline: accepted-and-right cells 312 -> 333 of 370, flagged 35 (was ~58), silent wrong 4 (unchanged); +0.5 s/sheet.

## Status: phone-quality robustness (latest)
- Finding: the marks reader degrades on re-compressed / downsized copies of the same photo (held-out cells right-and-accepted 333 -> 302 (2000px q85) -> 275 (1600px q75); silent wrong 4 -> 10). Gallery uploads from a phone are often such copies; live frames are JPEG-compressed too (live quality raised 0.92 -> 0.98). Smoothing filters before the ink mask did not help.
- Fix in progress: `scripts/build_cells_degraded.py` -> `data/cells_deg.npz` (train sheets: 4 random variants; val/test: 2 fixed variants for scoring only); `train_marks_hard.py --deg data/cells_deg.npz` swaps in a copy for half the real train rows each epoch and selects on original + phone-quality validation. Notebook: `kaggle_marks_robust.ipynb` -> `models/hard_c_s*.pt`. Current models are untouched (`hard_b_s0.pt`, `marks_sum.pt`).
- Note: ~33 of ~450 degraded copies failed to align at all (e.g. all 4 of sheet_042), so "could not align" is another failure mode on low-quality uploads.
- Next: run the notebook on Kaggle, then compare hard_c vs hard_b on original AND phone-quality held-out sheets (scripts/eval or data/robust_test.py style) before swapping it in.
- Ink detection made compression-robust (`cells.ink_mask`: extra rule = clearly darker + blurred red shift + not blue; only adds pixels). No retraining. Through the real pipeline on the 39 held-out sheets (cells right-and-accepted / silently wrong / flagged): original 333/4/53 -> 336/6/48; 2000px q85 copy 302/10/78 -> 331/6/53; 1600px q75 copy 275/9/106 -> 310/7/73; 1280px q60 copy 243/11/136 (no earlier baseline). Cost: ink_mask 0.33 s/call. Robust-training run C (hard_c_s0) gave ~no gain on top of this idea alone (its strips were mostly empty of ink); re-evaluate hard_c with the new mask if worth it. Scratch scripts: data/mask_exp.py, data/robust_test.py (data/ is git-ignored).
- Live scanning removed (grade_eval/live.py, assets/live.html, /live*, /live-test*). Phone path is now only the photo-upload page: `/s/<set>/photo-link` (QR + steps, plain http on :8000, no certificate warning), `/s/<set>/photo` (Take photo / Choose from gallery, multiple), `/s/<set>/photo/capture` + `/photo/job/<id>`. Network helpers moved to `grade_eval/phone.py` (https server still runs but is optional).
