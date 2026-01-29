# Flower Image Classifier (Udacity AIPND)

Command-line (CLI) app to train an image classifier on the **Udacity Flowers** dataset and run predictions on new images.

This repo is designed to be **runnable on a fresh machine**: clone → install → train → predict.

---

## What’s in this repo

- `train.py` — trains a classifier and saves a checkpoint
- `predict.py` — loads a checkpoint and predicts top-*k* classes for an input image
- `helper.py` — training / preprocessing / checkpoint helpers
- `get_input_args.py` — CLI argument definitions
- `cat_to_name.json` — mapping from class id → flower name
- `assets/` — example images (optional)

> The dataset (`flowers/`) is **not included** and is ignored by git.

---

## Setup

### Option A (recommended): pip + virtual environment

```bash
python -m venv .venv

# Windows (Git Bash)
source .venv/Scripts/activate

# macOS / Linux
# source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

### Option B: Conda

```bash
conda create -n flower_image_classifier python=3.10 -y
conda activate flower_image_classifier
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Dataset layout (expected)

Download the **Udacity Flowers dataset** and place it in the repo root as:

```bash
flowers/
  train/
  valid/
  test/
```

Example (paths used by the CLI):

- training data: `flowers/train/`
- validation data: `flowers/valid/`
- test data: `flowers/test/`

> `flowers/` is ignored by `.gitignore`, so it will not be committed.

---

## Quickstart (CPU)

### 1) Train (creates a checkpoint)

This command trains for 1 epoch and saves a checkpoint into the default folder:

```bash
python train.py flowers --epochs 1
```

Default checkpoint output folder:

- `save_directory/`

Typical checkpoint file created:

- `save_directory/checkpoint.pth`

You can change where checkpoints go with `--save_dir`, for example:

```bash
python train.py flowers --epochs 3 --save_dir checkpoints/
```

### 2) Predict (top-5 classes)

Use the checkpoint produced by training:

```bash
python predict.py flowers/test/3/image_06634.jpg save_directory/checkpoint.pth --top_k 5
```

The output prints:

- predicted flower name + probability
- top-*k* class names + probabilities

---

## GPU usage (optional)

If you have a CUDA-capable GPU and a compatible PyTorch install, add `--gpu`:

```bash
python train.py flowers --epochs 3 --gpu
python predict.py <image_path> save_directory/checkpoint.pth --top_k 5 --gpu
```

If no GPU is available, the code will run on CPU.

---

## CLI reference

### Training

```bash
python train.py -h
```

Key arguments:

- `data_dir` (positional): dataset folder (e.g. `flowers`)
- `--save_dir`: folder where checkpoints are saved (default: `save_directory/`)
- `--arch`: pretrained architecture (default: `resnet50`)
- `--learning_rate`
- `--hidden_units` (space-separated list, e.g. `--hidden_units 512 256`)
- `--dropout`
- `--epochs`
- `--gpu`

### Prediction

```bash
python predict.py -h
```

Key arguments:

- `path_to_image` (positional)
- `path_to_checkpoint` (positional)
- `--top_k` (default shown by `-h`)
- `--category_names` (default: `cat_to_name.json`)
- `--gpu`

---

## Notes

- **Checkpoints are not committed.** They are ignored by `.gitignore` (including `*.pth` and the `save_directory/` output folder).
- **Image preprocessing:** images are converted to RGB during preprocessing so `.png` images with transparency (RGBA) work reliably.
- You may see torchvision warnings about `pretrained` being deprecated. These are harmless and do not affect runtime.

---

## Troubleshooting

### “Parent directory save_directory does not exist”

Create the folder or use a different `--save_dir`. The code should also auto-create it when saving:

```bash
mkdir save_directory
```

### “Weights only load failed” (PyTorch 2.6+)

Newer PyTorch versions changed default loading behavior. The code should load trusted checkpoints correctly. If you hit this, ensure you are using the latest code in this repo.

---

## License

This project is for learning/portfolio purposes. Add a license file if you plan to reuse/distribute it widely.
