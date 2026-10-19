# OVERNIGHT STATE (branch `overnight`, started 2026-10-19 23:50 local)
Owner asleep. Work only on branch `overnight`. Do NOT push, delete, or change system settings. Keep `main` and models/ originals untouched
(new models get new names; backups in data/overnight/backup_models/).

## Goal
More sheets CONFIRMED (not "to check"/"fix") without increasing confirmed-WRONG. Main lever: read the written total
(Sub-Total cell or Grand Total box) and cross-check the question marks against it.

## Rules
- Guardrail: confirmed-wrong on held-out sheets must not exceed the baseline. Otherwise do not promote the change.
- Honest eval = the 39 held-out (val+test) sheets using OOF predictions; train final models on all 140 afterwards.
- Check usage (get_usage) before each stage; stop starting new work at ~90% of the 5h window, write this file, and wait for the heartbeat cron.
- Heartbeat: a recurring cron wakes the session; on wake read this file and continue the next unchecked stage.

## Stages (tick when done)
- [ ] 0 setup (branch, keep-awake, state, heartbeat)
- [ ] 1 baseline numbers on held-out 39 sheets (data/overnight/baseline.json)
- [ ] 2 label where totals are written (data/overnight/total_loc.csv: sheet,loc[S|G|B|N|X],note)
- [ ] 3 sub-total cell reader (scripts/train_total.py -> models/total_s.pt), CV
- [ ] 4 wire total into pipeline (confirm / repair / choose), score
- [ ] 5 grand-total box reader if time
- [ ] 6 compression-robust retrain, ensemble
- [ ] 7 tune acceptance rule by CV, guardrail check
- [ ] 8 OVERNIGHT_REPORT.md

## Log
- 23:58 stage0 done (branch overnight, keep_awake pid running, heartbeat cron 365159bc). stage1: baseline held-out orig: confirmed 5 ok/0 wrong; check 8/1; fix 1/22. q85: conf 3/1wrong. q60: 9 unaligned.
- 00:30 stage2 labelling done: data/overnight/total_loc.csv (S/G/B/N/X)
- 00:45 USER RULE: do not overfit. Tune only on train split / CV; the 39 held-out sheets (val+test) are scored, not tuned on (look at them once per candidate system, never pick thresholds from them). Final models use the settings fixed in CV. Report train-vs-heldout gaps.
- 00:45 stage2 result: S 51, G 47, B 15, X 15, N 4 (readable total in sub cell: 66 sheets, in grand box: 62). Wrote grade_eval/totals.py (TotalReader). Next: scripts/build_total_data.py, scripts/train_total.py (eval model on train split -> models/total_eval_*.pt; final on all -> models/total_*.pt).
- 00:25 stage3: total reader v1 (decode-based) is WEAK OOF (confident-wrong 6/23 at tau .9; circled 22 read as 2). Switching to likelihood-ratio scoring of candidate sums (totals.string_loglik). Retraining v2 (repeat-digit synth, saves logits) -> data/overnight/total_logits_*_eval.npz. Pipeline has _cross_check_total (decode-based, rule A/B/C) but is NOT yet validated; do not promote.
- 01:05 v2 totals LR calibration (train OOF): wrong numbers reach as much evidence as true ones (lam 8: true 33/74, some wrong 29/74) -> NOT safe. v3 training started (N_SYN=1200 + clutter), log train_total_eval3.log. Meanwhile: realistic roster eval + exact-unique-reg rule + alignment robustness.
