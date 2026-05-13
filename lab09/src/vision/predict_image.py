from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
from feature_extractor import extract_features
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


def load_model():
    if not MODEL_PATH.exists():
        print(f"Error: model file not found: {MODEL_PATH}")
        raise SystemExit(1)
    model = joblib.load(MODEL_PATH)
    print("Model loaded.")
    return model


def predict_image(model, image_path):
    path = Path(image_path)
    if not path.exists():
        print(f"Error: file not found: {image_path}")
        return
    with Image.open(path) as image:
        features = extract_features(image)
        image_for_plot = image.copy()

    prediction = model.predict([features])[0]
    print("=== Prediction ===")
    print(f"Image: {image_path}")
    print(f"Predicted class: {prediction}")

    plt.imshow(image_for_plot)
    plt.title(f"Prediction: {prediction}")
    plt.axis("off")
    plt.show()


def main():
    model = load_model()
    image_path = DATASET_DIR / "test/forest/forest_0000.jpg"
    image_path = RAW_ROOT / "eurosat/2750/Highway/Highway_1.jpg"
    predict_image(model, image_path)


if __name__ == "__main__":
    main()
