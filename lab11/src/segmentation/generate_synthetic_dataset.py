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
    # base image and mask
    image = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), CLASS_COLORS[0])
    mask = Image.new("L", (IMAGE_SIZE, IMAGE_SIZE), 0)

    image_draw = ImageDraw.Draw(image)
    mask_draw = ImageDraw.Draw(mask)

    # --- vegetation: make it more irregular by composing several overlapping ellipses/polygons
    veg_center_x = random.randint(20, IMAGE_SIZE - 20)
    veg_center_y = random.randint(20, IMAGE_SIZE - 20)
    veg_pieces = random.randint(3, 6)
    veg_coords = []
    for i in range(veg_pieces):
        rx = random.randint(10, 40)
        ry = random.randint(8, 36)
        ox = veg_center_x + random.randint(-20, 20)
        oy = veg_center_y + random.randint(-20, 20)
        bbox = [ox - rx, oy - ry, ox + rx, oy + ry]
        image_draw.ellipse(bbox, fill=CLASS_COLORS[1])
        mask_draw.ellipse(bbox, fill=1)
        veg_coords.append(bbox)

    # add small irregular protrusions using polygons
    for _ in range(random.randint(1, 3)):
        poly = [
            (random.randint(0, IMAGE_SIZE), random.randint(0, IMAGE_SIZE))
            for _ in range(random.randint(3, 6))
        ]
        image_draw.polygon(poly, fill=CLASS_COLORS[1])
        mask_draw.polygon(poly, fill=1)

    # --- water: thin, meandering river made of several narrow segments
    river_width = random.randint(3, 8)  # thinner than before
    x = random.randint(10, 30)
    for _ in range(8):
        # small horizontal or diagonal segment
        next_x = x + random.randint(8, 18)
        y_top = random.randint(0, IMAGE_SIZE - 1)
        y_bottom = min(IMAGE_SIZE - 1, y_top + river_width + random.randint(-1, 1))
        image_draw.rectangle([x, y_top, next_x, y_bottom], fill=CLASS_COLORS[2])
        mask_draw.rectangle([x, y_top, next_x, y_bottom], fill=2)
        x = next_x - random.randint(0, 6)

    # --- urban: more, smaller blocks and some overlap with other classes
    num_blocks = random.randint(8, 18)
    for _ in range(num_blocks):
        w = random.randint(4, 9)  # smaller blocks
        h = random.randint(4, 9)
        ux = random.randint(0, IMAGE_SIZE - w - 1)
        uy = random.randint(0, IMAGE_SIZE - h - 1)
        # sometimes draw before vegetation to create occlusion
        if random.random() < 0.4:
            image_draw.rectangle([ux, uy, ux + w, uy + h], fill=CLASS_COLORS[3])
            mask_draw.rectangle([ux, uy, ux + w, uy + h], fill=3)
        else:
            # draw on top (overlap)
            image_draw.rectangle([ux, uy, ux + w, uy + h], fill=CLASS_COLORS[3])
            mask_draw.rectangle([ux, uy, ux + w, uy + h], fill=3)

    # add stronger RGB noise and occasional salt-and-pepper to make classification harder
    image_array = np.array(image).astype(np.int16)
    # Gaussian-like noise via normal distribution (clipped)
    noise = np.random.normal(loc=0.0, scale=22.0, size=image_array.shape).astype(np.int16)
    image_array = image_array + noise
    # salt-and-pepper
    sp_count = int(0.002 * IMAGE_SIZE * IMAGE_SIZE)
    for _ in range(sp_count):
        sx = random.randint(0, IMAGE_SIZE - 1)
        sy = random.randint(0, IMAGE_SIZE - 1)
        if random.random() < 0.5:
            image_array[sy, sx] = [0, 0, 0]
        else:
            image_array[sy, sx] = [255, 255, 255]

    image_array = np.clip(image_array, 0, 255).astype(np.uint8)
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
