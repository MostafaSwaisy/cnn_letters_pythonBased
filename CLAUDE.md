# CLAUDE.md

Guidance for Claude Code (or any agent) working in this repository.

## Project objective

A from-scratch CNN, in pure Python, that classifies handwritten/printed
characters and can read a whole word out of an image (segment + classify
each character).

Reference/inspiration for the overall approach (conv -> pool -> dense ->
classify, then segment a word into letters): "Handwriting Recognition with
CNN: A Beginner's Guide"
(https://medium.com/@wricha.singh01/handwriting-recognition-with-cnn-a-beginners-guide-eba5984d7b8d).
Follow the spirit of that template (a small conv+pool+dense classifier
feeding a segmentation-based word reader) rather than copying it
mechanically — this project's hard constraint is doing the maths by hand,
which that article's TensorFlow-based code does not need to.

## Hard constraints (non-negotiable)

- **No ML/numeric libraries.** No numpy, no TensorFlow, no PyTorch, no
  scikit-learn — nothing that does the maths for you.
- **Pillow is the only external dependency**, and only for reading pixel
  values out of image files (`Image.open(...).getdata()` and friends).
  `kagglehub` is a second, non-ML dependency used solely to fetch the
  training dataset — it does not participate in any maths.
- **All maths is written by hand**: convolution, max pooling, dense
  layers, sigmoid activation, backpropagation, gradient descent — all of
  it implemented from first principles in this codebase.
- **Matrices are plain Python lists of lists**: `[[float, ...], ...]`.
  No custom array/tensor classes.
- **Functions, not classes.** The whole codebase is organized as plain
  functions operating on dicts/lists (e.g. the model is a `dict` of
  weight lists). Do not introduce classes or OOP abstractions.
- **Sigmoid is the activation function** throughout (conv, hidden dense,
  output dense). Don't swap in ReLU/softmax/etc. without discussing it
  first — it would change the loss function's clean-gradient derivation
  too (see the comment in `train_sample`).

## Current architecture (cnn_letters.py)

```
input   20 x 20 grayscale, values in [0,1]  (1.0 = ink)
  |
conv    N_FILTERS kernels of 5x5, stride 1, valid  -> N x 16 x 16
sigmoid
  |
maxpool 2x2                                        -> N x 8 x 8
  |
flatten                                            -> N*64 vector
  |
dense   -> HIDDEN units, sigmoid
  |
dense   -> N_CLASSES units, sigmoid  (one-hot per class)
  |
loss    binary cross-entropy
```

Word reading (`read_text`) is separate from the classifier: it segments a
line image into letter boxes via a vertical ink-column projection, then
classifies each box independently. No RNN/CTC — segmentation replaces
that entirely, which is the intentional simplification vs. more typical
handwriting-recognition pipelines.

## Data

Training data comes from the Kaggle "English Handwritten Characters
Dataset" (`dhruvildave/english-handwritten-characters-dataset`), fetched
automatically via `kagglehub` and cached under `./dataset` (gitignored).
See `docs/task.md` for why this replaced the original synthetic
font-rendered dataset, and `docs/tech-decisions.md` for the full decision
log (dataset format, class count, hidden-layer size, etc.) — check that
file before revisiting any of those choices, and add a new entry there
whenever you make a decision worth remembering or rolling back later.

## Commands

```
python cnn_letters.py download      # fetch/cache the Kaggle dataset
python cnn_letters.py train         # train and save model.json
python cnn_letters.py test          # accuracy on the held-out split
python cnn_letters.py read examples/word.jpeg # segment a word image and print the text
```

## Working in this codebase

- Before adding a dependency, ask: does this do maths for me? If yes, it's
  out of scope — implement it by hand instead.
- Keep functions small and composable, mirroring the existing style (one
  function per architectural step: `conv_forward`, `maxpool_forward`,
  `dense_forward`, and their `*_backward` counterparts).
- When you make a non-obvious technical decision (dataset handling, class
  scope, hyperparameters, etc.), log it in `docs/tech-decisions.md` with
  the why and a rollback note, the same way existing entries are written.
