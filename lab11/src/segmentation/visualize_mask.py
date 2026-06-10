from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).parent.parent.parent

DATA_PATH = PROJECT_ROOT / "data"

SEGMENTATION_DIR = DATA_PATH / "segmentation"
IMAGE_DIR = SEGMENTATION_DIR / "images"
MASK_DIR = SEGMENTATION_DIR / "masks"

IMAGE_PATH = IMAGE_DIR / "scene_0000.png"
MASK_PATH = MASK_DIR / "scene_0000.png"

REPORTS_DIR = PROJECT_ROOT / "reports"
SEGMENTATION_OUTPUT_DIR = REPORTS_DIR / "segmentation_examples"
OUTPUT_PATH = SEGMENTATION_OUTPUT_DIR / "dataset_example.png"

PALETTE = {
    0: (80, 80, 80),  # background
    1: (40, 140, 40),  # vegetation
    2: (40, 80, 180),  # water
    3: (180, 180, 180),  # urban
}


def mask_to_rgb(mask):
    height, width = mask.shape
    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    for class_id, color in PALETTE.items():
        rgb[mask == class_id] = color
    return rgb


def main():
    if not IMAGE_PATH.exists():
        print(f"Error: image not found: {IMAGE_PATH}")
        print("Run generate_synthetic_dataset.py first.")
        raise SystemExit(1)
    if not MASK_PATH.exists():
        print(f"Error: mask not found: {MASK_PATH}")
        print("Run generate_synthetic_dataset.py first.")
        raise SystemExit(1)
    image = Image.open(IMAGE_PATH).convert("RGB")
    mask = np.array(Image.open(MASK_PATH), dtype=np.uint8)
    mask_rgb = mask_to_rgb(mask)
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title("Synthetic EO Image")
    plt.axis("off")
    plt.subplot(1, 2, 2)
    plt.imshow(mask_rgb)
    plt.title("Colorized Segmentation Mask")
    plt.axis("off")
    plt.tight_layout()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
