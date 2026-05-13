from pathlib import Path

import joblib
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


def load_image_split(split_dir):
    X = []
    y = []
    class_dirs = sorted([path for path in split_dir.iterdir() if path.is_dir()])
    for class_dir in class_dirs:
        class_name = class_dir.name
        image_files = sorted(
            [
                path
                for path in class_dir.iterdir()
                if path.suffix.lower() in [".jpg", ".jpeg", ".png"]
            ]
        )
        for image_path in image_files:
            with Image.open(image_path) as image:
                features = extract_features(image)
            X.append(features)
            y.append(class_name)
    X = np.array(X)
    y = np.array(y)
    return X, y


def load_training_and_test_data():
    train_dir = DATASET_DIR / "train"
    test_dir = DATASET_DIR / "test"
    X_train, y_train = load_image_split(train_dir)
    X_test, y_test = load_image_split(test_dir)
    print("=== Image ML Dataset ===")
    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_test shape: {y_test.shape}")
    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train):
    print("=== Training Image Classifier ===")
    print(f"Training samples: {len(X_train)}")
    print(f"Number of features per image: {X_train.shape[1]}")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    print("Model trained.")
    return model

def evaluate_model(model, X_test, y_test):
    print("=== Model Evaluation ===")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Number of test samples: {len(X_test)}")
    print(f"Accuracy: {accuracy:.4f}")
    return accuracy

def save_model(model):
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print("=== Saving Model ===")
    print(f"Saved model: {MODEL_PATH}")

def main():
    X_train, X_test, y_train, y_test = load_training_and_test_data()

    model = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)

    save_model(model)



if __name__ == "__main__":
    main()
