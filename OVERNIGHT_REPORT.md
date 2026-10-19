# Overnight report (branch `overnight`, 2026-10-19/20)

Nothing was pushed, deleted or changed outside this project. `main` and the original models are untouched; new models have new names
(`models/total_sub.pt`, `models/total_grand.pt`, plus `_eval` variants). Every new behaviour can be switched off (see "Switches").

## What I was asked
More sheets **confirmed** (fewer "unconfirmed"), without more wrong ones, and **do not overfit**. All choices were made on the train split /
cross-validation; the 39 held-out sheets (val + test) were only scored, never tuned on. Held-out = 37 aligned sheets, so differences of one or
two sheets are noise.

## Result on the 39 held-out sheets (end-to-end through the real pipeline, with a class list)
Tiers: **confirmed** (written without a human glance) / **to check** (read, one click) / **needs fix**. "wrong" = at least one value in the sheet is wrong.

| Photo quality | Before (main) | After (this branch) |
|---|---|---|
| Original photos | confirmed 5 ok, **0 wrong** | confirmed 6 ok, **0 wrong** |
| Re-saved 2000px q85 (phone gallery) | confirmed 3 ok, **1 wrong** | confirmed 4 ok, **0 wrong** |
| Re-saved 1280px q60 (WhatsApp) | 9 sheets could not be aligned, confirmed 0 | 5 not aligned, confirmed 1 ok, 0 wrong |

Sheets whose marks are fully cleared and correct (reg number aside): original 12 -> 14, q85 5 -> 7, q60 2 -> 2.
The gain in *confirmed* sheets is small (+1 per variant) and I will not oversell it: most sheets are still "needs fix" (22 of 37) because
sum rows such as `3+3+1+2` read with low confidence and there is no total to check them against.

## What changed
1. **Reader ensemble (default on):** the base marks reader and the hard-row specialist are averaged on every row. On compressed photos this removed the
   one confirmed-but-wrong sheet and cut wrong "to check" sheets from 4 to 1 at WhatsApp quality; unchanged at original quality.
2. **Written total cross-check (default on, `models/total_sub.pt`, `total_grand.pt`):** the total is written in the Sub-Total cell on ~50% of sheets and in the
   Grand Total box on ~47% (I labelled where it is written on all 132 aligned sheets: `data/overnight/total_loc.csv`; 15 scribbled, 4 blank).
   Decoding the total directly was NOT reliable (circled 22 read as 2; wrong numbers got as much evidence as right ones). So the check never trusts a decoded
   total. Instead it scores how well the total strips match each sum the rows could reach (CTC likelihood ratio vs "nothing written"), and confirms only when
   the rows' BEST reading already adds up to that number, evidence >= 11 nats and a clear margin. Held-out: 0 new wrong confirmations.
   Tried and rejected: changing rows to fit the total (0 of 4 right on held-out), a mismatch-blocks-the-sheet rule.
3. **Unique exact register match:** an exact class-list match is trusted when no other student is one slip away (confirmed 5 -> 6, 0 wrong). With a dense or
   look-alike class list it does not fire and the name check is still needed.
4. **Alignment of small / compressed photos:** upscale before feature matching (WhatsApp-quality unaligned 9 -> 5 of 39; the full set 109 -> 124 of 132).
5. **Best guess pre-filled** for unclear rows (yellow, never auto-accepted). On held-out its top guess is right ~40% of the time, so it saves typing but is a guess.
6. **Phone upload page warns** about small / compressed copies. **Corrections are logged** to `data/corrections/` (aligned image + final marks) for future training.
7. `scripts/eval_e2e.py` (end-to-end scoring, 3 photo qualities), `scripts/score_pipeline.py`, `scripts/train_total.py`, `scripts/build_total_data.py`, `scripts/align_experiment.py`.

## About "same image on PC vs phone"
Same code path; the bytes differ. EXIF rotation is handled correctly (tested 4 orientations). The cause is re-compression / downsizing by the phone gallery or
WhatsApp: on `sheet_003` the same photo read all 5 marks at full size but flagged a different row at 2000px q85 and at 1280px q60. Mitigations: ensemble (fewer silent
errors), alignment upscaling, and a warning on the upload page. Best practice: use "Take photo" (full resolution), not a forwarded copy.

## What did NOT work / is still open
- Sum rows with no readable total stay unconfirmed. This is by design (nothing independent to check them against).
- The total readers are weak with ~80 training sheets (total evidence >= 11 on only ~1/3 of sheets with a readable total). More labelled totals would help most.
- 15 sheets have scribbled totals, 4 have none, 9 have a total that disagrees with the rows: those can never auto-confirm.
- Held-out set is only 37 sheets. A fresh set of 30-50 labelled sheets is the real test.
- 8 curled sheets still fail to align at original quality.

## Switches (environment variables)
`GRADE_NO_TOTAL=1` off total check | `GRADE_ENS=0` off ensemble | `GRADE_ALIGN_UP=0` off upscaling | `GRADE_REG_UNIQUE=0` off unique-reg rule |
`GRADE_TOTAL_SUFFIX=_eval` use the train-split-only total models (honest scoring).

## To use
Merge or check out `overnight`; models needed in `models/`: `marks_sum.pt`, `hard_b_s0.pt`, `total_sub.pt`, `total_grand.pt`, `trocr_reg_small/`, `trocr_name_small/`.
`python scripts/package_models.py` zips them. `python scripts/eval_e2e.py --split heldout --variant all3 --roster <class list csv with reg_no,name>` reproduces the table.
