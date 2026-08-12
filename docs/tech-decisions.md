# Technical decisions log

Running log of concrete decisions made on this project, newest first.
Each entry: what was decided, why, and what the rollback path is if we
want to revisit it later.

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
