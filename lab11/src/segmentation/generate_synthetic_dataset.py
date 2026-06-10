import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).parent.parent.parent

DATA_PATH = PROJECT_ROOT / "data"

SEGMENTATION_DIR = DATA_PATH / "segmentation"
IMAGE_DIR = SEGMENTATION_DIR / "images"
MASK_DIR = SEGMENTATION_DIR / "masks"

IMAGE_SIZE = 128
NUM_IMAGES = 200
RANDOM_SEED = 42

CLASS_COLORS = {
    0: (80, 80, 80),  # background
    1: (40, 140, 40),  # vegetation
    2: (40, 80, 180),  # water
    3: (180, 180, 180),  # urban
}


def create_scene(index):
    image = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), CLASS_COLORS[0])
    mask = Image.new("L", (IMAGE_SIZE, IMAGE_SIZE), 0)

    image_draw = ImageDraw.Draw(image)
    mask_draw = ImageDraw.Draw(mask)
    # vegetation region
    x1 = random.randint(0, 40)
    y1 = random.randint(0, 40)
    x2 = random.randint(70, 127)
    y2 = random.randint(70, 127)
    image_draw.ellipse([x1, y1, x2, y2], fill=CLASS_COLORS[1])
    mask_draw.ellipse([x1, y1, x2, y2], fill=1)
    # water region
    river_x = random.randint(40, 80)
    image_draw.rectangle([river_x, 0, river_x + 15, IMAGE_SIZE], fill=CLASS_COLORS[2])
    mask_draw.rectangle([river_x, 0, river_x + 15, IMAGE_SIZE], fill=2)
    # urban blocks
    for _ in range(5):
        ux = random.randint(0, IMAGE_SIZE - 20)
        uy = random.randint(0, IMAGE_SIZE - 20)
        image_draw.rectangle([ux, uy, ux + 12, uy + 12], fill=CLASS_COLORS[3])
        mask_draw.rectangle([ux, uy, ux + 12, uy + 12], fill=3)
    # add slight RGB noise
    image_array = np.array(image).astype(np.int16)
    noise = np.random.randint(-15, 16, image_array.shape)
    image_array = np.clip(image_array + noise, 0, 255).astype(np.uint8)
    image = Image.fromarray(image_array)
    image.save(IMAGE_DIR / f"scene_{index:04d}.png")
    mask.save(MASK_DIR / f"scene_{index:04d}.png")


def main():
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    MASK_DIR.mkdir(parents=True, exist_ok=True)
    for index in range(NUM_IMAGES):
        create_scene(index)
    print("=== Synthetic Segmentation Dataset ===")
    print(f"Generated images: {NUM_IMAGES}")
    print(f"Image folder: {IMAGE_DIR}")
    print(f"Mask folder: {MASK_DIR}")


if __name__ == "__main__":
    main()
