from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_PATH = PROJECT_ROOT / "data"
RAW_ROOT = DATA_PATH / "raw"
PROCESSED_ROOT = DATA_PATH / "processed"
DATASET_DIR = PROCESSED_ROOT

MODELS_PATH = PROJECT_ROOT / "models"
MODEL_PATH = MODELS_PATH / "image_model.joblib"

output_path = DATA_PATH / "inference_samples/noise.jpg"
output_path.parent.mkdir(parents=True, exist_ok=True)
array = np.random.randint(0, 256, size=(64, 64, 3), dtype=np.uint8)
image = Image.fromarray(array)
image.save(output_path)
print(f"Saved noise image: {output_path}")
