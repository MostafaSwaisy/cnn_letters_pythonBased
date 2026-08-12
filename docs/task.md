# Task: Replace synthetic letter generation with the Kaggle handwritten-characters dataset

## Background

`cnn_letters.py` is a from-scratch (no numpy/tensorflow) CNN that classifies
handwritten/printed letters. Originally it trained on a *synthetic* dataset:
`make_dataset()` rendered each of A-Z with a handful of local system fonts
(`makedata` CLI command), so the project would run without any external
data dependency.

## Goal

Replace the synthetic generator with real handwritten samples from the
Kaggle dataset:
https://www.kaggle.com/datasets/dhruvildave/english-handwritten-characters-dataset

This dataset contains scanned handwriting samples for digits (0-9),
uppercase letters (A-Z), and lowercase letters (a-z) — 62 classes, listed
in `english.csv` (columns: `image`, `label`) alongside an `Img/` folder of
PNG files.

## Scope

- Auto-download the dataset via `kagglehub` on first use, cached locally
  under `./dataset`.
- Load samples directly from `english.csv` rather than reorganizing into
  per-class folders (case-insensitive filesystems like Windows would merge
  an `A/` folder with an `a/` folder — a real correctness bug, not a
  style choice).
- Expand the model from 26 classes (`A-Z`) to 62 classes (`0-9A-Za-z`).
- Remove the now-unused synthetic generator (`make_dataset`, `find_fonts`,
  the `makedata` CLI command).
- Bump `HIDDEN` (and possibly `EPOCHS`) since 62-way classification is a
  harder problem than 26-way. Tune later if results are weak.
- Everything else (conv/pool/dense forward+backward, segmentation-based
  word reading, model persistence) is unchanged.

## Out of scope

- Architectural changes to the CNN itself (more filters, different pooling,
  etc.) — only the data pipeline and class count change here.
- CI/CD, packaging, or distribution concerns.

See `tech-decisions.md` in this folder for the running log of concrete
decisions made while implementing this, in case we need to revisit or
roll one back.
