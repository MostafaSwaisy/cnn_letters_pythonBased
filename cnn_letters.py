"""
CNN for handwritten/printed character classification -- pure Python.
External dependencies: Pillow (read pixel values), kagglehub (fetch the
training dataset).

No numpy, no tensorflow, no pytorch. All maths written by hand.
All matrices are plain lists of lists: [[float, ...], ...]
All code is organised in functions (no classes).
All activations are sigmoid.

Training data: the Kaggle "English Handwritten Characters Dataset"
(dhruvildave/english-handwritten-characters-dataset), downloaded
automatically on first use via kagglehub and cached under ./dataset.
This requires a Kaggle account with an API token configured
(~/.kaggle/kaggle.json) -- see https://www.kaggle.com/docs/api.

Architecture:
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
    dense   -> 62 units, sigmoid  (one-hot 0-9, A-Z, a-z)
      |
    loss    binary cross-entropy (keeps the gradient alive through sigmoid)

Usage:
    python3 cnn_letters.py download      # fetch/cache the Kaggle dataset
    python3 cnn_letters.py train         # train and save model.json
    python3 cnn_letters.py test          # accuracy on the held-out split
    python3 cnn_letters.py read examples/word.jpeg # segment a word image and print the text
"""

import csv
import json
import math
import os
import random
import shutil

from PIL import Image

import kagglehub


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------

IMG_SIZE = 20          # every letter is normalised to IMG_SIZE x IMG_SIZE
FILTER_SIZE = 5        # convolution kernel is FILTER_SIZE x FILTER_SIZE
N_FILTERS = 6          # number of feature maps
POOL_SIZE = 2          # max pooling window
HIDDEN = 48            # neurons in the hidden dense layer (tuned via experiments/, see docs/tech-decisions.md)
CLASSES = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
N_CLASSES = len(CLASSES)

LEARNING_RATE = 0.2    # tuned via experiments/, see docs/tech-decisions.md
EPOCHS = 35

DATA_DIR = "dataset"
DATASET_SLUG = "dhruvildave/english-handwritten-characters-dataset"
DATASET_CSV = "english.csv"
MODEL_FILE = "model.json"

CONV_OUT = IMG_SIZE - FILTER_SIZE + 1      # 16
POOL_OUT = CONV_OUT // POOL_SIZE           # 8
FLAT_SIZE = N_FILTERS * POOL_OUT * POOL_OUT


# ---------------------------------------------------------------------------
# 1. basic maths helpers
# ---------------------------------------------------------------------------

def sigmoid(x):
    """Logistic function, clamped so exp() never overflows."""
    if x < -60.0:
        return 0.0
    if x > 60.0:
        return 1.0
    return 1.0 / (1.0 + math.exp(-x))


def sigmoid_prime_from_output(a):
    """Derivative of sigmoid expressed with its own output: s'(z) = a * (1 - a)."""
    return a * (1.0 - a)


def zeros_matrix(rows, cols):
    return [[0.0 for _ in range(cols)] for _ in range(rows)]


def random_matrix(rows, cols, fan_in):
    """Xavier-style uniform init: keeps the weighted sums inside sigmoid's useful range."""
    limit = math.sqrt(6.0 / fan_in)
    return [[random.uniform(-limit, limit) for _ in range(cols)] for _ in range(rows)]


def argmax(vector):
    best_i = 0
    best_v = vector[0]
    for i in range(1, len(vector)):
        if vector[i] > best_v:
            best_v = vector[i]
            best_i = i
    return best_i


# ---------------------------------------------------------------------------
# 2. image loading (the only place Pillow is used)
# ---------------------------------------------------------------------------

def pil_to_matrix(pil_image, size=IMG_SIZE, invert=True):
    """Convert a PIL image to a list-of-lists of floats in [0,1].

    invert=True turns 'dark ink on white paper' into 'high value = ink',
    which is what the network expects.
    """
    img = pil_image.convert("L").resize((size, size))
    pixels = list(img.getdata())
    matrix = []
    for row in range(size):
        line = []
        for col in range(size):
            value = pixels[row * size + col] / 255.0
            if invert:
                value = 1.0 - value
            line.append(value)
        matrix.append(line)
    return matrix


def load_image(path, size=IMG_SIZE, invert=True):
    return pil_to_matrix(Image.open(path), size, invert)


def one_hot(index, length=N_CLASSES):
    vector = [0.0] * length
    vector[index] = 1.0
    return vector


def ensure_dataset(folder=DATA_DIR):
    """Make sure folder/english.csv exists, downloading it via kagglehub if not.

    Kept CSV-based (image path + label per row) rather than reorganised into
    one-subfolder-per-class: this dataset has both 'A' and 'a' classes, and
    case-insensitive filesystems (Windows, default macOS) would silently
    merge an 'A/' folder with an 'a/' folder.
    """
    csv_path = os.path.join(folder, DATASET_CSV)
    if os.path.isfile(csv_path):
        return
    cache_path = kagglehub.dataset_download(DATASET_SLUG)
    os.makedirs(folder, exist_ok=True)
    shutil.copytree(cache_path, folder, dirs_exist_ok=True)
    if not os.path.isfile(csv_path):
        raise RuntimeError("kagglehub download did not produce %s" % csv_path)


def load_dataset(folder=DATA_DIR):
    """Load every (image, label) pair listed in folder/english.csv."""
    ensure_dataset(folder)
    samples = []
    csv_path = os.path.join(folder, DATASET_CSV)
    with open(csv_path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            label_index = CLASSES.index(row["label"])
            image_path = os.path.join(folder, row["image"])
            image = load_image(image_path)
            samples.append((image, label_index))
    return samples


# ---------------------------------------------------------------------------
# 3. model creation / persistence
# ---------------------------------------------------------------------------

def init_model(seed=42, hidden=HIDDEN):
    """Returns the model as a plain dict of lists -- easy to save as JSON."""
    random.seed(seed)
    filters = []
    for _ in range(N_FILTERS):
        filters.append(random_matrix(FILTER_SIZE, FILTER_SIZE, FILTER_SIZE * FILTER_SIZE))
    return {
        "filters": filters,
        "conv_bias": [0.0] * N_FILTERS,
        "w1": random_matrix(hidden, FLAT_SIZE, FLAT_SIZE),
        "b1": [0.0] * hidden,
        "w2": random_matrix(N_CLASSES, hidden, hidden),
        "b2": [0.0] * N_CLASSES,
    }


def save_model(model, path=MODEL_FILE):
    with open(path, "w") as handle:
        json.dump(model, handle)


def load_model(path=MODEL_FILE):
    with open(path) as handle:
        return json.load(handle)


# ---------------------------------------------------------------------------
# 4. forward pass
# ---------------------------------------------------------------------------

def conv_forward(image, filters, conv_bias):
    """Valid convolution + sigmoid. Returns a list of feature maps."""
    size = len(image)
    k = len(filters[0])
    out_size = size - k + 1
    feature_maps = []
    for f in range(len(filters)):
        kernel = filters[f]
        bias = conv_bias[f]
        fmap = []
        for i in range(out_size):
            row = []
            for j in range(out_size):
                total = bias
                for a in range(k):
                    image_row = image[i + a]
                    kernel_row = kernel[a]
                    for b in range(k):
                        total += image_row[j + b] * kernel_row[b]
                row.append(sigmoid(total))
            fmap.append(row)
        feature_maps.append(fmap)
    return feature_maps


def maxpool_forward(feature_maps, pool=POOL_SIZE):
    """Max pooling. Also returns the winning coordinates, needed for backprop."""
    pooled_maps = []
    winners = []
    for fmap in feature_maps:
        size = len(fmap)
        out_size = size // pool
        pooled = []
        positions = []
        for i in range(out_size):
            pooled_row = []
            pos_row = []
            for j in range(out_size):
                best_value = -1e30
                best_i = i * pool
                best_j = j * pool
                for a in range(pool):
                    for b in range(pool):
                        value = fmap[i * pool + a][j * pool + b]
                        if value > best_value:
                            best_value = value
                            best_i = i * pool + a
                            best_j = j * pool + b
                pooled_row.append(best_value)
                pos_row.append((best_i, best_j))
            pooled.append(pooled_row)
            positions.append(pos_row)
        pooled_maps.append(pooled)
        winners.append(positions)
    return pooled_maps, winners


def flatten(maps):
    vector = []
    for fmap in maps:
        for row in fmap:
            for value in row:
                vector.append(value)
    return vector


def unflatten(vector, n_maps, size):
    maps = []
    index = 0
    for _ in range(n_maps):
        fmap = []
        for _ in range(size):
            row = []
            for _ in range(size):
                row.append(vector[index])
                index += 1
            fmap.append(row)
        maps.append(fmap)
    return maps


def dense_forward(inputs, weights, biases):
    """weights[out][in] -- classic list of lists."""
    outputs = []
    for i in range(len(weights)):
        total = biases[i]
        row = weights[i]
        for j in range(len(inputs)):
            total += row[j] * inputs[j]
        outputs.append(sigmoid(total))
    return outputs


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


# ---------------------------------------------------------------------------
# 5. backward pass  (this is where the real maths lives)
# ---------------------------------------------------------------------------

def dense_backward(inputs, delta, weights, biases, learning_rate):
    """delta = dLoss/dz of THIS layer.

    Returns dLoss/d(inputs) so the previous layer can continue.
    Note the gradient wrt inputs is accumulated BEFORE the weight is updated,
    otherwise we would be back-propagating through already-modified weights.
    """
    d_inputs = [0.0] * len(inputs)
    for i in range(len(weights)):
        di = delta[i]
        row = weights[i]
        for j in range(len(inputs)):
            d_inputs[j] += row[j] * di
            row[j] -= learning_rate * di * inputs[j]
        biases[i] -= learning_rate * di
    return d_inputs


def maxpool_backward(d_pooled_flatmaps, winners, n_maps, conv_size):
    """Gradient flows only to the pixel that won the max in each window."""
    d_conv = [zeros_matrix(conv_size, conv_size) for _ in range(n_maps)]
    for f in range(n_maps):
        pooled = d_pooled_flatmaps[f]
        for i in range(len(pooled)):
            for j in range(len(pooled[i])):
                wi, wj = winners[f][i][j]
                d_conv[f][wi][wj] += pooled[i][j]
    return d_conv


def conv_backward(image, conv_maps, d_conv_activation, filters, conv_bias, learning_rate):
    """Gradient wrt each kernel weight:
           dL/dW[a][b] = sum over output positions ( delta_z[i][j] * image[i+a][j+b] )
       where delta_z = dL/da * a * (1 - a)   (sigmoid derivative).
    """
    k = len(filters[0])
    out_size = len(conv_maps[0])
    for f in range(len(filters)):
        activation = conv_maps[f]
        d_activation = d_conv_activation[f]

        delta_z = []
        for i in range(out_size):
            row = []
            for j in range(out_size):
                a_val = activation[i][j]
                row.append(d_activation[i][j] * sigmoid_prime_from_output(a_val))
            delta_z.append(row)

        kernel = filters[f]
        for a in range(k):
            for b in range(k):
                gradient = 0.0
                for i in range(out_size):
                    delta_row = delta_z[i]
                    image_row = image[i + a]
                    for j in range(out_size):
                        gradient += delta_row[j] * image_row[j + b]
                kernel[a][b] -= learning_rate * gradient

        bias_gradient = 0.0
        for i in range(out_size):
            for j in range(out_size):
                bias_gradient += delta_z[i][j]
        conv_bias[f] -= learning_rate * bias_gradient


def train_sample(model, image, label_index, learning_rate=LEARNING_RATE):
    """One forward + backward + weight update. Returns (loss, predicted index)."""
    output, cache = forward(model, image)
    target = one_hot(label_index)

    # Binary cross-entropy on top of the sigmoid outputs.
    # The sigmoid derivative cancels against the loss derivative, leaving the
    # famously clean  delta = (a - y).  With plain squared error the factor
    # a*(1-a) makes the gradient vanish whenever the output saturates near 0,
    # and a 26-class one-hot target saturates almost every output.
    loss = 0.0
    delta_output = []
    for i in range(N_CLASSES):
        a = min(max(output[i], 1e-9), 1.0 - 1e-9)
        y = target[i]
        loss -= y * math.log(a) + (1.0 - y) * math.log(1.0 - a)
        delta_output.append(output[i] - y)

    # output layer -> hidden layer
    d_hidden_activation = dense_backward(
        cache["hidden"], delta_output, model["w2"], model["b2"], learning_rate
    )
    delta_hidden = [
        d_hidden_activation[i] * sigmoid_prime_from_output(cache["hidden"][i])
        for i in range(len(cache["hidden"]))
    ]

    # hidden layer -> flattened pooling output
    d_flat = dense_backward(
        cache["flat"], delta_hidden, model["w1"], model["b1"], learning_rate
    )

    # flatten -> pooling -> convolution
    d_pooled = unflatten(d_flat, N_FILTERS, POOL_OUT)
    d_conv_activation = maxpool_backward(d_pooled, cache["winners"], N_FILTERS, CONV_OUT)
    conv_backward(
        image, cache["conv_maps"], d_conv_activation,
        model["filters"], model["conv_bias"], learning_rate,
    )
    return loss, argmax(output)


def train(samples, model=None, epochs=EPOCHS, learning_rate=LEARNING_RATE, verbose=True,
          test_samples=None, checkpoint_path=None):
    """Train for `epochs` passes over `samples`.

    If `test_samples` is given, test accuracy is measured every epoch and
    the best-scoring model snapshot is tracked (early stopping against
    overfitting the training set). That snapshot is saved to
    `checkpoint_path` (if given) once training finishes; either way the
    function still returns the LAST epoch's model, matching the old
    no-test-tracking behaviour.
    """
    if model is None:
        model = init_model()
    order = list(range(len(samples)))
    best_test_acc = -1.0
    best_epoch = None
    best_snapshot = None
    for epoch in range(epochs):
        random.shuffle(order)
        total_loss = 0.0
        correct = 0
        for position in order:
            image, label = samples[position]
            loss, predicted = train_sample(model, image, label, learning_rate)
            total_loss += loss
            if predicted == label:
                correct += 1
        train_acc = correct / len(samples)
        line = "epoch %2d/%d  loss %.4f  train acc %.3f" % (epoch + 1, epochs, total_loss / len(samples), train_acc)
        if test_samples is not None:
            test_acc = evaluate(model, test_samples)
            line += "  test acc %.3f" % test_acc
            if test_acc > best_test_acc:
                best_test_acc = test_acc
                best_epoch = epoch + 1
                best_snapshot = json.loads(json.dumps(model))  # deep copy, model is plain JSON-able lists/dicts
        if verbose:
            print(line)
    if test_samples is not None:
        if verbose:
            print("best epoch %d  test acc %.3f" % (best_epoch, best_test_acc))
        if checkpoint_path:
            save_model(best_snapshot, checkpoint_path)
    return model


def evaluate(model, samples):
    correct = 0
    for image, label in samples:
        output, _ = forward(model, image)
        if argmax(output) == label:
            correct += 1
    return correct / len(samples) if samples else 0.0


def predict_letter(model, image):
    output, _ = forward(model, image)
    index = argmax(output)
    return CLASSES[index], output[index]


# ---------------------------------------------------------------------------
# 6. reading a whole word: segmentation by vertical projection
#    (this is what replaces the BiLSTM + CTC of the article)
# ---------------------------------------------------------------------------

def column_ink_profile(pil_image, threshold=0.5):
    """For every column, count how many pixels contain ink."""
    grey = pil_image.convert("L")
    width, height = grey.size
    pixels = list(grey.getdata())
    profile = []
    for x in range(width):
        count = 0
        for y in range(height):
            if 1.0 - pixels[y * width + x] / 255.0 > threshold:
                count += 1
        profile.append(count)
    return profile, width, height


def find_letter_boxes(profile, width, min_width=2):
    """Split the profile into runs of non-empty columns = candidate letters."""
    boxes = []
    start = None
    for x in range(width):
        if profile[x] > 0 and start is None:
            start = x
        elif profile[x] == 0 and start is not None:
            if x - start >= min_width:
                boxes.append((start, x))
            start = None
    if start is not None and width - start >= min_width:
        boxes.append((start, width))
    return boxes


def square_glyph(grey_image, box, pad=3):
    """Crop the glyph, centre it on a white square, keeping its aspect ratio.

    Training images and letters cut out of a word MUST go through this same
    function, otherwise the network sees two different distributions and
    accuracy collapses at inference time.
    """
    letter = grey_image.crop(box)
    if letter.size[0] == 0 or letter.size[1] == 0:
        return None
    side = max(letter.size) + 2 * pad
    canvas = Image.new("L", (side, side), 255)
    canvas.paste(letter, ((side - letter.size[0]) // 2, (side - letter.size[1]) // 2))
    return canvas


def crop_and_normalise(pil_image, left, right, threshold=0.5, pad=3):
    """Cut one letter out of a line image, trim its empty rows, square it."""
    grey = pil_image.convert("L")
    width, height = grey.size
    pixels = list(grey.getdata())

    top, bottom = None, None
    for y in range(height):
        has_ink = False
        for x in range(left, right):
            if 1.0 - pixels[y * width + x] / 255.0 > threshold:
                has_ink = True
                break
        if has_ink:
            if top is None:
                top = y
            bottom = y + 1
    if top is None:
        return None

    canvas = square_glyph(grey, (left, top, right, bottom), pad)
    return pil_to_matrix(canvas) if canvas is not None else None


def read_text(model, path, space_ratio=0.55):
    """Segment a single line/word image and classify every letter."""
    image = Image.open(path)
    profile, width, height = column_ink_profile(image)
    boxes = find_letter_boxes(profile, width)
    if not boxes:
        return ""

    gaps = [boxes[i + 1][0] - boxes[i][1] for i in range(len(boxes) - 1)]
    average_letter_width = sum(r - l for l, r in boxes) / len(boxes)
    space_gap = average_letter_width * space_ratio

    text = ""
    for index, (left, right) in enumerate(boxes):
        matrix = crop_and_normalise(image, left, right)
        if matrix is None:
            continue
        letter, _confidence = predict_letter(model, matrix)
        text += letter
        if index < len(gaps) and gaps[index] > space_gap:
            text += " "
    return text


def split_dataset(samples, test_ratio=0.2, seed=7):
    random.seed(seed)
    shuffled = samples[:]
    random.shuffle(shuffled)
    cut = int(len(shuffled) * (1.0 - test_ratio))
    return shuffled[:cut], shuffled[cut:]


# ---------------------------------------------------------------------------
# 7. command line entry point
# ---------------------------------------------------------------------------

def parse_flags(argv, spec):
    """Tiny --flag value parser. spec maps flag name -> converter function."""
    values = {}
    i = 0
    while i < len(argv):
        name = argv[i].lstrip("-")
        if name in spec:
            values[name] = spec[name](argv[i + 1])
            i += 2
        else:
            i += 1
    return values


def main():
    import sys
    command = sys.argv[1] if len(sys.argv) > 1 else "help"

    if command == "download":
        ensure_dataset()
        print("dataset ready in", DATA_DIR)

    elif command == "train":
        flags = parse_flags(sys.argv[2:], {
            "hidden": int, "epochs": int, "lr": float, "out": str,
        })
        hidden = flags.get("hidden", HIDDEN)
        epochs = flags.get("epochs", EPOCHS)
        learning_rate = flags.get("lr", LEARNING_RATE)
        out_path = flags.get("out", MODEL_FILE)

        samples = load_dataset()
        if not samples:
            print("No data. Run: python3 cnn_letters.py download")
            return
        train_set, test_set = split_dataset(samples)
        print("train %d samples, test %d samples  (hidden=%d epochs=%d lr=%.3f)"
              % (len(train_set), len(test_set), hidden, epochs, learning_rate))
        model = init_model(hidden=hidden)
        train(train_set, model=model, epochs=epochs, learning_rate=learning_rate,
              test_samples=test_set, checkpoint_path=out_path)
        print("best-epoch model saved to", out_path)

    elif command == "test":
        model = load_model()
        samples = load_dataset()
        _train_set, test_set = split_dataset(samples)
        print("test accuracy %.3f" % evaluate(model, test_set))

    elif command == "read":
        model = load_model()
        print(read_text(model, sys.argv[2]))

    else:
        print(__doc__)


if __name__ == "__main__":
    main()
