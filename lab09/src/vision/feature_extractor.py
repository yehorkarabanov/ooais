from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_PATH = PROJECT_ROOT / "data"
RAW_ROOT = DATA_PATH / "raw"
PROCESSED_ROOT = DATA_PATH / "processed"


def extract_features(image):
    image = image.convert("RGB")
    image = image.resize((64, 64))
    array = np.array(image) / 255.0
    return array.flatten()
