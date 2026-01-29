# Flower Image Classifier (Udacity AIPND)

Command-line (CLI) app to **train** an image classifier on the **Udacity Flowers** dataset and **predict** a flower name from a new image using **transfer learning**.

**TL;DR (try this first):**

- Train a checkpoint in ~1–5 minutes: `python train.py flowers --epochs 2`
- Predict a label from an image: `python predict.py flowers/test/3/image_06634.jpg save_directory/checkpoint.pth --top_k 5`
- Uses a pretrained backbone (default: **ResNet-50**) + a new classifier head.

---

## Demo (what you should see)

After training, run:

```bash
python predict.py flowers/test/3/image_06634.jpg save_directory/checkpoint.pth --top_k 5
```

Example output (format may vary slightly):

```text
Path to image: flowers/test/3/image_06634.jpg
Path to checkpoint: save_directory/checkpoint.pth
Number of top K classes: 5
Path to category names file: cat_to_name.json
GPU: False

Prediction (name): cape flower
Probability: 0.10262521356344223

Top classes (names): ['cape flower', 'cyclamen', 'lotus lotus', 'magnolia', 'columbine']
Top probabilites: [0.10262521356344223, 0.07787298411130905, 0.05228663235902786, 0.048569660633802414, 0.0458136685192585]
```

---

## What’s in this repo

- `train.py` — trains a classifier and saves a checkpoint
- `predict.py` — loads a checkpoint and predicts top-*k* classes for an input image
- `helper.py` — training / preprocessing / checkpoint helpers
- `get_input_args.py` — CLI argument definitions
- `cat_to_name.json` — mapping from class id → flower name
- `assets/` — screenshots / example images (optional)
- `notebooks/` — project notebook (reference / exploration)

**Not included:** the dataset folder `flowers/` (it is ignored by git).

---

## Approach

This project uses **transfer learning**:

1. Load a pretrained convolutional neural network (CNN) backbone (default: **ResNet-50**).
2. Replace the final classification layer with a new **fully-connected classifier head** for 102 flower classes.
3. Freeze (or mostly freeze) backbone parameters and train the classifier head on the flowers dataset.
4. Save a checkpoint so you can run fast predictions later without retraining.

---

## Quickstart (CPU) — clone → run in 5 minutes

### 1) Create an environment + install

#### Option A (recommended): pip + virtual environment

```bash
python -m venv .venv

# Windows (Git Bash)
source .venv/Scripts/activate

# macOS / Linux
# source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

#### Option B: Conda

```bash
conda create -n flower_image_classifier python=3.11 -y
conda activate flower_image_classifier
pip install --upgrade pip
pip install -r requirements.txt
```

### 2) Dataset layout (expected)

Place the **Udacity Flowers** dataset in the repo root:

```text
flowers/
  train/
  valid/
  test/
```

### 3) Train (creates a checkpoint)

Minimal training run (1 epoch):

```bash
python train.py flowers --epochs 2
```

Defaults:

- checkpoint folder: `save_directory/`
- checkpoint file: `save_directory/checkpoint.pth`
- architecture: `resnet50`

A “more realistic” training example:

```bash
python train.py flowers   --arch resnet50   --learning_rate 0.003   --hidden_units 512 256   --dropout 0.2   --epochs 3
```

### 4) Predict (top-5)

```bash
python predict.py flowers/test/3/image_06634.jpg save_directory/checkpoint.pth --top_k 5
```

---

## GPU usage (optional)

If you have a CUDA-capable GPU and a compatible PyTorch install, add `--gpu`:

```bash
python train.py flowers --epochs 3 --gpu
python predict.py flowers/test/3/image_06634.jpg save_directory/checkpoint.pth --top_k 5 --gpu
```

If no GPU is available, the code runs on CPU.

---

## Results

| Setting | Value |
| --- | --- |
| Backbone | ResNet-50 (default) |
| Epochs | 5 |
| Learning rate | 0.0005 |
| Hidden units | 512 |
| Dropout | 0.2 |
| Validation accuracy | 0.893 |
| Test accuracy (optional) | 0.878 |

---

## CLI reference

### Training help

```bash
python train.py -h
```

Common arguments:

- `data_dir` (positional): dataset folder (e.g. `flowers`)
- `--save_dir`: folder where checkpoints are saved (default: `save_directory/`)
- `--arch`: pretrained architecture (default: `resnet50`)
- `--learning_rate`
- `--hidden_units` (space-separated list, e.g. `--hidden_units 512 256`)
- `--dropout`
- `--epochs`
- `--gpu`

### Prediction help

```bash
python predict.py -h
```

Common arguments:

- `path_to_image` (positional)
- `path_to_checkpoint` (positional)
- `--top_k`
- `--category_names` (default: `cat_to_name.json`)
- `--gpu`

---

## License

This project is for learning/portfolio purposes. Add a license file if you plan to reuse/distribute it widely.
