# HOG Feature Augmentation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add hand-computed Histogram-of-Oriented-Gradients (HOG) features as
extra input to the hidden dense layer, alongside the existing conv+pool
output, to see whether explicit stroke-direction information improves test
accuracy beyond the current 64.4% (hidden=48, lr=0.2, see
`docs/tech-decisions.md`).

**Architecture:** `forward()` currently flattens the conv+pool output
(`CONV_FLAT_SIZE` = 384 values) and feeds it straight into the hidden dense
layer. This plan adds a new `hog_features(image)` function that computes an
unsigned-gradient HOG descriptor (5x5 cells x 8 bins = 200 values, L2
normalised per cell) directly from the raw 20x20 input image, independent of
the conv path. The two vectors are concatenated (`combined = conv_flat +
hog`, length 584) and that's what actually feeds `dense_forward` for the
hidden layer. Backward pass mirrors this: `dense_backward` returns a
gradient for the full 584-length combined input, but only the first 384
values (the conv-flatten portion) have any upstream parameters to update —
HOG is a fixed, non-learned function of the pixels, so its gradient slice is
simply discarded.

**Tech Stack:** Pure Python (per `CLAUDE.md` hard constraints — no numpy,
matrices are lists of lists, `math` stdlib only). No test framework exists
in this repo; verification is done with short inline `python -c`/heredoc
scripts and full CLI training runs, matching the pattern already used
earlier in this project (see prior smoke-tests in this session).

---

### Task 0: Create the feature branch

**Files:** none (git only)

- [ ] **Step 1: Create and switch to the branch**

```bash
git checkout -b feature/hog-features
```

- [ ] **Step 2: Verify**

```bash
git branch --show-current
```
Expected: `feature/hog-features`

---

### Task 1: Add HOG constants and rename FLAT_SIZE -> CONV_FLAT_SIZE

**Files:**
- Modify: `cnn_letters.py:72-74`

- [ ] **Step 1: Replace the constants block**

Current code at `cnn_letters.py:72-74`:

```python
CONV_OUT = IMG_SIZE - FILTER_SIZE + 1      # 16
POOL_OUT = CONV_OUT // POOL_SIZE           # 8
FLAT_SIZE = N_FILTERS * POOL_OUT * POOL_OUT
```

Replace with:

```python
CONV_OUT = IMG_SIZE - FILTER_SIZE + 1      # 16
POOL_OUT = CONV_OUT // POOL_SIZE           # 8
CONV_FLAT_SIZE = N_FILTERS * POOL_OUT * POOL_OUT   # 384

HOG_CELL = 4                                        # pixels per HOG cell side
HOG_BINS = 8                                         # unsigned orientation bins (0-180deg)
HOG_CELLS_PER_SIDE = IMG_SIZE // HOG_CELL            # 5
HOG_FEATURE_SIZE = HOG_CELLS_PER_SIDE * HOG_CELLS_PER_SIDE * HOG_BINS  # 200

FLAT_SIZE = CONV_FLAT_SIZE + HOG_FEATURE_SIZE        # 584, hidden layer's input width
```

- [ ] **Step 2: Verify no other code breaks from the rename**

```bash
python -m py_compile cnn_letters.py
```
Expected: no output (success). `FLAT_SIZE` is still defined (just bigger),
so `init_model()`'s `random_matrix(hidden, FLAT_SIZE, FLAT_SIZE)` call at
`cnn_letters.py:194` keeps working unchanged — it'll just build a wider `w1`.

- [ ] **Step 3: Commit**

```bash
git add cnn_letters.py
git commit -m "Add HOG constants, split FLAT_SIZE into CONV_FLAT_SIZE + HOG_FEATURE_SIZE"
```

---

### Task 2: Implement `hog_features()`

**Files:**
- Modify: `cnn_letters.py` — insert the new function right before `forward()` (currently at `cnn_letters.py:308`), after `dense_forward`.

- [ ] **Step 1: Add the function**

Insert before `def forward(model, image):`:

```python
def hog_features(image, cell=HOG_CELL, bins=HOG_BINS):
    """Histogram-of-oriented-gradients descriptor for one IMG_SIZE x IMG_SIZE image.

    Unsigned gradient orientation (0-180 degrees), binned per cell and
    weighted by gradient magnitude; each cell's histogram is L2-normalised
    independently so stroke thickness/contrast doesn't dominate the scale.
    Border pixels (no full neighbourhood for the central-difference
    gradient) are skipped -- their contribution to a 20x20 image is
    negligible. This is a fixed function of the pixels, not a learned
    layer, so it has no weights and needs no backward pass.
    """
    size = len(image)
    cells_per_side = size // cell
    histograms = [[0.0] * bins for _ in range(cells_per_side * cells_per_side)]
    bin_width = 180.0 / bins

    for y in range(1, size - 1):
        cell_y = y // cell
        if cell_y >= cells_per_side:
            continue
        for x in range(1, size - 1):
            cell_x = x // cell
            if cell_x >= cells_per_side:
                continue
            gx = image[y][x + 1] - image[y][x - 1]
            gy = image[y + 1][x] - image[y - 1][x]
            magnitude = math.sqrt(gx * gx + gy * gy)
            if magnitude == 0.0:
                continue
            angle = math.degrees(math.atan2(gy, gx)) % 180.0
            b = int(angle / bin_width) % bins
            histograms[cell_y * cells_per_side + cell_x][b] += magnitude

    features = []
    for hist in histograms:
        norm = math.sqrt(sum(v * v for v in hist)) + 1e-6
        features.extend(v / norm for v in hist)
    return features
```

- [ ] **Step 2: Verify shape and basic sanity with a scratch script**

```bash
python - <<'EOF'
import cnn_letters as c

blank = [[0.0] * c.IMG_SIZE for _ in range(c.IMG_SIZE)]
feats = c.hog_features(blank)
assert len(feats) == c.HOG_FEATURE_SIZE == 200, len(feats)
assert all(v == 0.0 for v in feats), "blank image should have zero gradient everywhere"

# a vertical ink stripe down the middle should light up a specific orientation bin
striped = [[0.0] * c.IMG_SIZE for _ in range(c.IMG_SIZE)]
for y in range(c.IMG_SIZE):
    striped[y][10] = 1.0
feats2 = c.hog_features(striped)
assert len(feats2) == 200
assert sum(feats2) > 0.0, "stripe image should produce nonzero HOG energy"
print("OK", sum(feats2))
EOF
```
Expected: `OK <some positive number>` with no assertion errors.

- [ ] **Step 3: Commit**

```bash
git add cnn_letters.py
git commit -m "Implement hog_features(): unsigned-gradient HOG descriptor"
```

---

### Task 3: Wire HOG into the forward pass

**Files:**
- Modify: `cnn_letters.py:308-323` (the `forward()` function)

- [ ] **Step 1: Replace `forward()`**

Current code:

```python
def forward(model, image):
    """Full forward pass. Returns the prediction plus every cached value."""
    conv_maps = conv_forward(image, model["filters"], model["conv_bias"])
    pooled_maps, winners = maxpool_forward(conv_maps)
    flat = flatten(pooled_maps)
    hidden = dense_forward(flat, model["w1"], model["b1"])
    output = dense_forward(hidden, model["w2"], model["b2"])
    cache = {
        "image": image,
        "conv_maps": conv_maps,
        "winners": winners,
        "flat": flat,
        "hidden": hidden,
        "output": output,
    }
    return output, cache
```

Replace with:

```python
def forward(model, image):
    """Full forward pass. Returns the prediction plus every cached value."""
    conv_maps = conv_forward(image, model["filters"], model["conv_bias"])
    pooled_maps, winners = maxpool_forward(conv_maps)
    conv_flat = flatten(pooled_maps)
    hog = hog_features(image)
    combined = conv_flat + hog
    hidden = dense_forward(combined, model["w1"], model["b1"])
    output = dense_forward(hidden, model["w2"], model["b2"])
    cache = {
        "image": image,
        "conv_maps": conv_maps,
        "winners": winners,
        "combined": combined,
        "hidden": hidden,
        "output": output,
    }
    return output, cache
```

Note `cache["flat"]` is renamed to `cache["combined"]` — Task 4 updates the
one place that reads it.

- [ ] **Step 2: Verify it at least runs (will still be wrong until Task 4 fixes the backward pass, but forward alone should not crash)**

```bash
python - <<'EOF'
import cnn_letters as c

model = c.init_model()
blank = [[0.0] * c.IMG_SIZE for _ in range(c.IMG_SIZE)]
output, cache = c.forward(model, blank)
assert len(output) == c.N_CLASSES
assert len(cache["combined"]) == c.FLAT_SIZE == 584
print("OK")
EOF
```
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add cnn_letters.py
git commit -m "Concatenate HOG features with conv-flatten output in forward()"
```

---

### Task 4: Fix the backward pass to split the combined gradient

**Files:**
- Modify: `cnn_letters.py` inside `train_sample()` — the block that currently reads:

```python
    # hidden layer -> flattened pooling output
    d_flat = dense_backward(
        cache["flat"], delta_hidden, model["w1"], model["b1"], learning_rate
    )

    # flatten -> pooling -> convolution
    d_pooled = unflatten(d_flat, N_FILTERS, POOL_OUT)
```

- [ ] **Step 1: Replace that block**

```python
    # hidden layer -> combined (conv-flatten ++ HOG) input
    d_combined = dense_backward(
        cache["combined"], delta_hidden, model["w1"], model["b1"], learning_rate
    )
    # HOG is a fixed function of the pixels (no upstream weights), so only
    # the conv-flatten portion of the gradient continues backward.
    d_conv_flat = d_combined[:CONV_FLAT_SIZE]

    # flatten -> pooling -> convolution
    d_pooled = unflatten(d_conv_flat, N_FILTERS, POOL_OUT)
```

- [ ] **Step 2: Compile-check**

```bash
python -m py_compile cnn_letters.py
```
Expected: no output.

- [ ] **Step 3: Verify one full train_sample() call runs end-to-end without shape errors and loss is finite**

```bash
python - <<'EOF'
import cnn_letters as c

model = c.init_model(hidden=48)
blank = [[0.0] * c.IMG_SIZE for _ in range(c.IMG_SIZE)]
loss, predicted = c.train_sample(model, blank, 0, learning_rate=0.2)
assert loss == loss, "loss is NaN"  # NaN != NaN
assert 0 <= predicted < c.N_CLASSES
print("OK loss=", loss)
EOF
```
Expected: `OK loss= <some finite number>`, no traceback.

- [ ] **Step 4: Commit**

```bash
git add cnn_letters.py
git commit -m "Split combined-input gradient in train_sample(); HOG portion is discarded"
```

---

### Task 5: Smoke-test a short real training run

**Files:** none (verification only, mirrors the smoke tests already used earlier in this project)

- [ ] **Step 1: Run 2 epochs on a small slice of the real dataset**

```bash
python - <<'EOF'
import cnn_letters as c

samples = c.load_dataset()
train_set, test_set = c.split_dataset(samples[:200])
model = c.init_model(hidden=48)
c.train(train_set, model=model, epochs=2, learning_rate=0.2,
        test_samples=test_set, checkpoint_path=None, verbose=True)
print("done")
EOF
```
Expected: two `epoch ... train acc ... test acc ...` lines with non-degenerate
(neither 0.0 nor NaN) accuracy, followed by `best epoch ...` and `done`,
no traceback.

- [ ] **Step 2: Update the module docstring's architecture diagram**

In `cnn_letters.py:17-31`, the `Architecture:` block. Replace:

```
    flatten                                            -> N*64 vector
      |
    dense   -> HIDDEN units, sigmoid
```

with:

```
    flatten                                            -> N*64 vector
      |                                    HOG (5x5 cells x 8 bins) -> 200 vector
      |                                                  |
      +---------------------- concat ---------------------+
      |
    dense   -> HIDDEN units, sigmoid
```

- [ ] **Step 3: Update `CLAUDE.md`'s matching architecture diagram** (same
  change, in the `## Current architecture (cnn_letters.py)` section)

- [ ] **Step 4: Commit**

```bash
git add cnn_letters.py CLAUDE.md
git commit -m "Update architecture diagrams for HOG augmentation"
```

---

### Task 6: Full comparison run against the current 64.4% baseline

**Files:**
- Uses existing CLI (`cnn_letters.py`'s `train` command with `--out`,
  already supports arbitrary output paths from the earlier hyperparameter
  work) — no code changes.

- [ ] **Step 1: Run full training with HOG enabled, same hyperparameters as the current best (hidden=48, lr=0.2, epochs=35), writing to a scratch path so `model.json` isn't touched yet**

```bash
mkdir -p experiments
python cnn_letters.py train --hidden 48 --epochs 35 --lr 0.2 \
    --out experiments/hog_hidden48_lr02.json \
    > experiments/hog_hidden48_lr02.log 2>&1
```

Run this in the background (it takes several minutes, same as prior full
runs in this project) and wait for completion before continuing.

- [ ] **Step 2: Compare best test accuracy against the 64.4% baseline**

```bash
tail -5 experiments/hog_hidden48_lr02.log
```
Note the `best epoch N  test acc X.XXX` line.

- [ ] **Step 3: Decide adopt vs. rollback**

If HOG's best test acc beats 64.4% by a meaningful margin (not just noise —
compare against the ~1-2pp run-to-run variance already seen in
`docs/tech-decisions.md`'s 4-variant table): adopt it — copy
`experiments/hog_hidden48_lr02.json` over `model.json` and keep the code
changes.

If it doesn't help or hurts: keep the code (it's a legitimate documented
experiment either way) but do NOT overwrite `model.json`, and note in the
decision log that HOG augmentation was tried and didn't help, so a future
reader doesn't repeat the experiment blind. Whether to then revert the
code changes entirely is a judgment call to bring back to the user rather
than deciding unilaterally.

---

### Task 7: Log the decision

**Files:**
- Modify: `docs/tech-decisions.md` (prepend new entry, same format as the
  existing "Add early-stopping checkpointing..." entry)

- [ ] **Step 1: Write the entry** with the actual measured numbers from
  Task 6 Step 2, following the existing entries' Decision/Why/Rollback
  structure. Include the HOG parameters used (`HOG_CELL=4, HOG_BINS=8,
  HOG_FEATURE_SIZE=200`) and the comparison number against the 64.4%
  baseline.

- [ ] **Step 2: Commit**

```bash
git add docs/tech-decisions.md
git commit -m "Log HOG feature augmentation experiment result"
```

- [ ] **Step 3: If adopted, also commit the retrained model**

```bash
git add model.json  # only if .gitignore allows it -- check first, it currently does not track model.json
```
(`model.json` is gitignored per `cnn_letters.py`'s existing `.gitignore`
entry — this step is a no-op in practice; the trained model stays local,
consistent with how the current 64.4% model is handled. Skip this step.)

- [ ] **Step 4: Push the branch (do not merge to main without the user's go-ahead)**

```bash
git push -u origin feature/hog-features
```

---

## Self-Review Notes

- **Spec coverage:** Augment-not-replace integration (user's chosen
  option) — Task 3/4. HOG computed by hand, no libraries — Task 2. Branch
  created first — Task 0. Plan written before touching code — this
  document. Comparison against the existing 64.4% baseline — Task 6.
  Decision logged per project convention — Task 7.
- **No placeholders:** every code step shows complete, runnable code; the
  "adopt vs rollback" step in Task 6 is a genuine decision point (depends
  on data not yet known) rather than a vague TODO, and it's resolved by an
  explicit rule (meaningful margin vs. documented noise floor) plus an
  explicit instruction to bring ambiguity back to the user instead of
  guessing.
- **Type/name consistency:** `CONV_FLAT_SIZE`, `HOG_FEATURE_SIZE`,
  `cache["combined"]`, `hog_features()` are named identically everywhere
  they're introduced and later referenced (Tasks 1, 2, 3, 4).
