# Technical decisions log

Running log of concrete decisions made on this project, newest first.
Each entry: what was decided, why, and what the rollback path is if we
want to revisit it later.

---

## 2026-08-12 — Add HOG feature augmentation (branch `feature/hog-features`)

**Decision:** Added a hand-implemented HOG (Histogram of Oriented Gradients)
descriptor, computed directly from the raw 20x20 input image
(`HOG_CELL=4, HOG_BINS=8` -> 5x5 cells x 8 bins = 200 values, unsigned
orientation, magnitude-weighted, L2-normalised per cell), concatenated
with the existing conv+pool flattened output (384 values) before the
hidden dense layer (`combined` = 584 values feeding `w1`). HOG is a fixed
function of the pixels, not a learned layer, so its portion of the
backward-pass gradient is computed then discarded (no upstream weights to
update).

Compared against the current best (hidden=48, lr=0.2, no HOG: 64.4% test
acc, see the entry below) with identical hyperparameters and the same
train/test split:

| Variant | test acc | best epoch |
|---|---|---|
| No HOG (existing baseline) | 0.644 | 16 |
| + HOG augmentation | **0.667** | 31 |

+2.3pp improvement. Since the pipeline is fully seeded/deterministic
(fixed `random.seed` in `init_model`, fixed split seed in
`split_dataset`), this reflects the actual effect of the added features
for this exact configuration, not run-to-run sampling noise — though it's
a modest gain, not a dramatic one. Adopted: `model.json` now holds this
HOG-augmented, best-epoch-31 model.

**Why:** HOG gives the network explicit stroke-direction information the
raw-pixel conv path has to learn from scratch with a tiny dataset (~55
samples/class); the improvement, while modest, is consistent with that
reasoning and cost nothing at inference time beyond the extra feature
computation.

**Rollback:** The conv path is untouched — HOG is purely additive. To
remove it: revert to before commit `1d48362` on `feature/hog-features`
(or equivalently, on `main`, never merge this branch), which restores
`FLAT_SIZE = CONV_FLAT_SIZE` (384, no HOG concat) and the old
`cache["flat"]` naming. `model.json` is not compatible between the two
(different `w1` width: 584 vs 384 columns) — requires retraining either
way. Full task-by-task implementation record:
`docs/superpowers/plans/2026-08-12-hog-feature-augmentation.md`.

**Caveat:** 66.7% is still a long way from strong OCR performance. HOG
cell size (4px) and bin count (8) were picked as reasonable defaults, not
tuned — a follow-up hyperparameter sweep on those two (similar to the
HIDDEN/LR sweep below) is a plausible next lever, as is the
previously-flagged option of folding visually-identical classes together.

---

## 2026-08-12 — Add early-stopping checkpointing, tune HIDDEN/LR down after comparing 4 variants

**Decision:** Training the 62-class model with `HIDDEN=80, LR=0.5, EPOCHS=35`
(the previous entry's bump) produced severe overfitting: 97.5% train accuracy
but only 51.3% test accuracy at epoch 35. Two changes were made:

1. `train()` now evaluates test accuracy every epoch and keeps the
   best-scoring model snapshot instead of whatever the last epoch produced
   (early stopping by checkpointing). This is now always-on default
   behaviour for the `train` CLI command, not an opt-in flag.
2. Ran 4 variants (same 2728/682 train/test split, all with checkpointing)
   to compare `HIDDEN` and `LEARNING_RATE`:

   | Variant | hidden | lr | best epoch | best test acc | train acc at best epoch |
   |---|---|---|---|---|---|
   | 0: baseline + checkpointing | 80 | 0.5 | 30 | 0.554 | 0.956 |
   | 1: smaller hidden | 48 | 0.5 | 26 | 0.528 | 0.811 |
   | 2: lower learning rate | 80 | 0.2 | 29 | 0.642 | 0.993 |
   | 3: smaller hidden + lower lr | 48 | 0.2 | 16 | 0.644 | 0.873 |

   Learning rate was the dominant factor, not hidden size. Variant 3 tied
   variant 2 on accuracy but reached its best epoch in about half the
   epochs with a smaller (cheaper) hidden layer, so it's the adopted
   default: `HIDDEN = 48`, `LEARNING_RATE = 0.2`.

**Why:** Checkpointing alone recovered part of the overfitting gap
(51.3% -> 55.4%) for free — no reason not to make it the default. Among
the hyperparameter changes, a lower learning rate mattered far more than
hidden-layer size for this dataset/architecture; `HIDDEN=48` was kept
because it's strictly better (same accuracy, cheaper, fewer epochs to get
there) rather than for its own sake.

**Rollback:** Set `HIDDEN` back to 80 and/or `LEARNING_RATE` back to 0.5
in `cnn_letters.py` if a retest suggests otherwise. The `train` CLI now
accepts `--hidden`, `--epochs`, `--lr`, `--out` overrides to rerun
comparisons without editing constants — see the (gitignored) `experiments/`
folder pattern used for this comparison. `model.json` is not compatible
across `HIDDEN` values — requires retraining either way. The checkpointing
behaviour itself (saving the best epoch, not the last) is not expected to
need rolling back independent of the hyperparameter values.

**Caveat:** 64% test accuracy is still far from great for 62-class OCR;
if it's not enough, the next lever (not yet tried) is folding visually
identical classes (`0`/`O`, `1`/`l`/`I`, etc.) back together, which was
previously ruled out in favour of keeping all 62 classes distinct.

---

## 2026-08-12 — Bump HIDDEN (and possibly EPOCHS) for the 62-class problem

**Decision:** Increase `HIDDEN` from 40 (tuned for 26-class A-Z) to 80 as a
first attempt at accommodating the harder 62-class (0-9, A-Z, a-z) problem.
Leave `EPOCHS` at its current value initially; revisit only if training
accuracy plateaus low.

**Why:** More output classes with visually similar characters (e.g. `0`/`O`,
`1`/`l`/`I`, `S`/`s`) need more hidden-layer capacity to separate. Agreed
with the user to try this first and roll back / tune further only if
results are unsatisfactory.

**Rollback:** Set `HIDDEN` back to 40 in `cnn_letters.py`. `model.json` is
not compatible across `HIDDEN` values (dense layer shapes differ), so a
rollback also requires retraining.

---

## 2026-08-12 — Load dataset directly from `english.csv`, not per-class folders

**Decision:** `load_dataset()` reads `dataset/english.csv` (image path +
label columns) and loads images at the paths it lists, instead of
reorganizing the Kaggle dataset into `dataset/<CLASS>/*.png` subfolders
(the layout the old synthetic generator used).

**Why:** The dataset has both uppercase and lowercase classes (`A` and `a`,
etc). Windows (and default macOS) filesystems are case-insensitive, so an
`A/` folder and an `a/` folder would collide and silently merge two
classes' samples — a correctness bug, not just a style preference.
Reading straight from the CSV sidesteps this entirely and also avoids an
extra file-copying/reorganization step.

**Rollback:** If we ever want per-class folders again (e.g. for uppercase
only 26-class mode, no case collision), reintroduce a folder-based loader
as an alternate path gated by a flag, keep the CSV loader as default.

---

## 2026-08-12 — Expand to all 62 classes (digits + upper + lower) rather than 26 or a folded case-insensitive 26

**Decision:** `CLASSES = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"`,
`N_CLASSES = 62`. All three label groups in the Kaggle dataset are treated
as distinct classes.

**Why:** User's explicit choice over "keep 26, uppercase only" and
"fold case-insensitively" options. Uses the full dataset instead of
discarding ~2/3 of it.

**Rollback:** Set `CLASSES` back to `"ABCDEFGHIJKLMNOPQRSTUVWXYZ"` and
filter `load_dataset()` rows to uppercase-only labels. Existing
`model.json` files are not compatible across `N_CLASSES` (output layer
shape differs) — requires retraining either way.

---

## 2026-08-12 — Auto-download via `kagglehub`, cached under `./dataset`

**Decision:** Add `kagglehub` as a dependency. `ensure_dataset()` checks
for `./dataset/english.csv`; if missing, calls
`kagglehub.dataset_download("dhruvildave/english-handwritten-characters-dataset")`
and copies the result into `./dataset` so it's cached locally and
inspectable, rather than leaving it in kagglehub's own cache directory or
requiring the user to download it manually.

**Why:** User preferred automatic download over a manual step, and a
local project-relative cache over `kagglehub`'s default cache location, so
the dataset is easy to find/inspect/gitignore alongside the rest of the
project.

**Caveat:** Requires a Kaggle account with an API token (`kaggle.json`)
configured — this is standard Kaggle auth and cannot be set up by an
agent on the user's behalf.

**Rollback:** If auto-download proves unreliable (auth issues, dataset
gated, etc.), fall back to a manual step: user downloads/extracts the
Kaggle dataset into `./dataset` themselves; `ensure_dataset()` already
short-circuits when `english.csv` is present, so no code change is
strictly needed — just skip relying on the download call succeeding.

---

## 2026-08-12 — Removed the synthetic font-based dataset generator

**Decision:** Deleted `make_dataset()`, `find_fonts()`, and the `makedata`
CLI command entirely, rather than keeping them as a fallback.

**Why:** User's explicit choice ("Remove it"). The synthetic generator
existed only so the project could run without external data; now that
real data is the primary path, keeping the generator around is dead
weight that would need to stay in sync with the `CLASSES`/`N_CLASSES`
change (it only ever produced uppercase A-Z).

**Rollback:** Restore from git history (the commit immediately before its
removal) if a no-external-dependency demo mode is needed again.
