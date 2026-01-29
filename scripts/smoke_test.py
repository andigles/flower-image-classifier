import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
FLOWERS_DIR = REPO_ROOT / "flowers"
DEFAULT_IMAGE = REPO_ROOT / "flowers" / "test" / "3" / "image_06634.jpg"
CHECKPOINT = REPO_ROOT / "save_directory" / "checkpoint.pth"

def run(cmd: list[str]) -> None:
    print("\n$", " ".join(cmd))
    completed = subprocess.run(cmd, cwd=REPO_ROOT)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)

def main() -> None:
    # Always verify the CLIs are wired
    run([sys.executable, "train.py", "-h"])
    run([sys.executable, "predict.py", "-h"])

    # If dataset exists locally, do a tiny real run
    if FLOWERS_DIR.exists():
        # Train a minimal checkpoint (1 epoch)
        run([sys.executable, "train.py", "flowers", "--epochs", "1"])
        # Predict using a known test image path
        img = str(DEFAULT_IMAGE) if DEFAULT_IMAGE.exists() else str(next(FLOWERS_DIR.rglob("*.jpg")))
        run([sys.executable, "predict.py", img, str(CHECKPOINT), "--top_k", "3"])
        print("\nSmoke test: OK (train + predict)")
    else:
        print("\nSmoke test: OK (CLI help only — dataset not present)")

if __name__ == "__main__":
    main()
