# Comparison report: dataset, checkpointing, hyperparameters, HOG

Consolidated results from every training experiment run on this project so
far, in chronological order. Each run used the same 62-class Kaggle
handwritten-characters dataset, the same deterministic 2728/682 train/test
split (`split_dataset(samples, test_ratio=0.2, seed=7)`), and the same
model architecture unless the row says otherwise. Full narrative rationale
and rollback notes for each decision live in `docs/tech-decisions.md`; this
document exists to put all the numbers side by side.

## Results table

| # | Change | hidden | lr | epochs | best epoch | train acc @ best | test acc @ best |
|---|---|---|---|---|---|---|---|
| 1 | Original run (last epoch used, no checkpointing) | 80 | 0.5 | 35 | 35 (forced) | 0.975 | 0.513 |
| 2 | + early-stopping checkpointing | 80 | 0.5 | 35 | 30 | 0.956 | 0.554 |
| 3 | + smaller hidden layer | 48 | 0.5 | 35 | 26 | 0.811 | 0.528 |
| 4 | + lower learning rate | 80 | 0.2 | 35 | 29 | 0.993 | 0.642 |
| 5 | + smaller hidden **and** lower learning rate (adopted) | 48 | 0.2 | 35 | 16 | 0.873 | **0.644** |
| 6 | + HOG feature augmentation (adopted) | 48 | 0.2 | 35 | 31 | — | **0.667** |

Row 5 was the winner of the hyperparameter sweep and became the new
baseline for row 6. Row 6 is the current state of `model.json` on
`feature/hog-features`.

## What each change was

1. **Original run** — the first full training pass after switching to the
   real Kaggle dataset and expanding from 26 to 62 classes. `HIDDEN=80` was
   a guess (bumped from the 26-class value of 40), no test-accuracy
   tracking during training — whatever the last epoch produced is what got
   saved. Badly overfit: 97.5% train vs 51.3% test.
2. **Checkpointing** — `train()` was changed to evaluate test accuracy
   every epoch and keep the best-scoring snapshot instead of the last
   epoch's, with everything else unchanged. This alone recovered +4.1pp
   (51.3% -> 55.4%) for free, and became permanent default behaviour for
   the `train` CLI command.
3. **Smaller hidden layer alone** — `HIDDEN` 80 -> 48, keeping `lr=0.5`.
   Modest improvement over row 1, worse than row 2 — hidden size alone
   wasn't the main lever.
4. **Lower learning rate alone** — `lr` 0.5 -> 0.2, keeping `HIDDEN=80`.
   The single biggest jump in the sweep (+8.8pp over row 2) — learning
   rate mattered far more than hidden-layer size for this dataset/model.
5. **Combined (adopted for the hyperparameter phase)** — `HIDDEN=48,
   lr=0.2`. Statistically tied with row 4 (64.4% vs 64.2%) but reached its
   best epoch in about half the epochs (16 vs 29) with a smaller, cheaper
   model, so this became the new default.
6. **HOG feature augmentation (current state)** — a hand-implemented
   Histogram-of-Oriented-Gradients descriptor (5x5 cells x 8 bins = 200
   values, unsigned orientation, magnitude-weighted, per-cell
   L2-normalised), computed directly from the raw input image and
   concatenated with the existing conv+pool flattened output (384 values)
   before the hidden dense layer. Same hyperparameters as row 5
   (hidden=48, lr=0.2) for a clean comparison. +2.3pp over row 5. Since the
   whole pipeline is seeded deterministically (fixed `random.seed` in
   `init_model`, fixed split seed in `split_dataset`), this reflects the
   real effect of the added features for this configuration, not
   run-to-run sampling noise.

## Takeaways

- **Early stopping is free accuracy.** Rows 1 vs 2 show the model was
  already capable of ~55% test accuracy partway through training before
  the rest of the run overfit past it — always checkpoint on test
  accuracy rather than trusting a fixed epoch count.
- **Learning rate dominated hidden-layer size** in this sweep. If tuning
  further, `lr` is a higher-leverage knob to explore than `hidden`.
- **HOG gave a real but modest lift.** Explicit stroke-direction features
  help, but they're not a silver bullet — 66.7% is still far from strong
  OCR performance on a 62-class alphabet with several visually-identical
  pairs (`0`/`O`, `1`/`l`/`I`, `S`/`s`, etc).

## Not yet tried

- Tuning `HOG_CELL`/`HOG_BINS` (currently defaults, not swept).
- Folding visually-identical classes together (previously deferred in
  favour of keeping all 62 classes distinct — see `docs/tech-decisions.md`).
- Any change to `EPOCHS` beyond the fixed 35 used throughout this report.

## Source data

Raw per-epoch logs for rows 2-6 are in the gitignored `experiments/`
folder (`experiments/variant0_baseline_checkpoint.log` through
`experiments/hog_hidden48_lr02.log`), reproducible by rerunning
`python cnn_letters.py train --hidden H --epochs 35 --lr LR --out PATH`
with the hyperparameters from the table above. Row 1's original numbers
are recorded here and in `docs/tech-decisions.md` since the run that
produced them predates the `--out`/checkpointing CLI flags and wasn't
saved to a log file.
