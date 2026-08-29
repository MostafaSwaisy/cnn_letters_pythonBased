# cnn_letters

A convolutional neural network for handwritten/printed character
recognition, written **from scratch in pure Python**. Every piece of the
maths — convolution, max pooling, dense layers, sigmoid activation,
backpropagation, gradient descent — is implemented by hand. No numpy, no
TensorFlow, no PyTorch, no scikit-learn.

The classifier recognises single characters (`0-9`, `A-Z`, `a-z`; 62
classes). On top of it, `read_text` reads a whole word out of an image by
segmenting the line into letter boxes (vertical ink-column projection) and
classifying each box independently — no RNN or CTC.

## Constraints

This is a learning project whose whole point is doing the maths by hand,
so the rules are strict:

- **No ML / numeric libraries.** Nothing that does the maths for you.
- **Pillow** is used only to read pixel values out of image files.
  **kagglehub** is used only to fetch the training dataset. Neither
  participates in any computation.
- **Matrices are plain Python lists of lists** (`[[float, ...], ...]`) —
  no array/tensor classes.
- **Functions, not classes.** The model is a `dict` of weight lists.
- **Sigmoid everywhere** (conv, hidden dense, output dense), with binary
  cross-entropy loss so the gradient stays alive through the saturated
  outputs of a 62-class one-hot target.

See [CLAUDE.md](CLAUDE.md) for the full rationale.

## Architecture

```
input   20 x 20 grayscale, values in [0,1]  (1.0 = ink)
  |
conv    6 kernels of 5x5, stride 1, valid           -> 6 x 16 x 16
sigmoid
  |
maxpool 2x2                                          -> 6 x 8 x 8
  |
flatten                                              -> 384 vector
  |            input image -> HOG (5x5 cells x 8 bins) -> 200 vector
  |                                                     |
  +------------------------ concat ----------------------+
  |                                                    (584 vector)
dense   -> 48 hidden units, sigmoid
  |
dense   -> 62 output units, sigmoid  (one-hot per class)
  |
loss    binary cross-entropy
```

The HOG (Histogram of Oriented Gradients) branch is a hand-implemented,
fixed function of the input pixels — unsigned gradient orientation,
magnitude-weighted, per-cell L2-normalised. It carries no weights and has
no backward pass; the HOG portion of the hidden layer's input gradient is
simply discarded during backprop.

## Setup

```
pip install -r requirements.txt
```

Fetching the dataset needs a Kaggle account with an API token configured
at `~/.kaggle/kaggle.json` — see <https://www.kaggle.com/docs/api>.

## Usage

```
python cnn_letters.py download                  # fetch/cache the Kaggle dataset -> ./dataset
python cnn_letters.py train                     # train, save best-epoch model to model.json
python cnn_letters.py test                      # accuracy on the held-out 20% split
python cnn_letters.py read examples/word.jpeg   # segment a word image and print the text
```

`train` accepts optional flags: `--hidden N`, `--epochs N`, `--lr F`,
`--out PATH`. It evaluates the held-out split every epoch and saves the
best-scoring snapshot (early stopping against overfitting), not the last
epoch's weights.

## Data

Training data is the Kaggle "English Handwritten Characters Dataset"
(`dhruvildave/english-handwritten-characters-dataset`): scanned samples
for 62 classes, listed in `english.csv` (`image`, `label` columns)
alongside an image folder. It is downloaded automatically on first use and
cached under `./dataset` (gitignored). Samples are loaded straight from
the CSV rather than reorganised into per-class folders, because
case-insensitive filesystems would silently merge the `A` and `a`
classes.

## Results

Best test accuracy so far is **66.7%** on the deterministic 2728/682
train/test split (`hidden=48`, `lr=0.2`, 35 epochs, HOG features enabled).
The full experiment log — dataset switch, early-stopping checkpointing,
hyperparameter sweep, HOG augmentation — is in
[docs/comparison-report.md](docs/comparison-report.md). 66.7% is a real
result for a hand-rolled 62-class classifier but well short of strong OCR:
several class pairs (`0`/`O`, `1`/`l`/`I`, `S`/`s`) are near-identical at
20x20.

## Repository layout

```
cnn_letters.py            the entire implementation + CLI
requirements.txt          Pillow, kagglehub
examples/word.jpeg        sample word image for `read`
docs/task.md              the dataset-migration task writeup
docs/tech-decisions.md    running decision log (hyperparameters, dataset format, ...)
docs/comparison-report.md side-by-side numbers for every training run
CLAUDE.md                 guidance for agents working in this repo
```
