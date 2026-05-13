import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from feature_extractor import extract_features
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_PATH = PROJECT_ROOT / "data"
RAW_ROOT = DATA_PATH / "raw"
PROCESSED_ROOT = DATA_PATH / "processed"
DATASET_DIR = PROCESSED_ROOT

MODELS_PATH = PROJECT_ROOT / "models"
MODEL_PATH = MODELS_PATH / "image_model.joblib"


def load_image_split(split_dir):
    """Load images from a split directory and extract features."""
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
    """Load training and test data."""
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


def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    """Train multiple models and measure their performance."""
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=3),
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "SVM": SVC(),
    }

    results = []

    print("\n=== Training and Evaluating Models ===")
    for model_name, model in models.items():
        print(f"\nTraining {model_name}...")

        # Measure training time
        start_time = time.time()
        model.fit(X_train, y_train)
        end_time = time.time()
        training_time = end_time - start_time

        # Evaluate model
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        # Store results
        results.append(
            {
                "model_name": model_name,
                "accuracy": accuracy,
                "training_time": training_time,
                "model": model,
            }
        )

        print(f"Model: {model_name}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Training time: {training_time:.4f} s")

    return results


def plot_accuracy_vs_training_time(results):
    """Create a plot showing accuracy vs training time."""
    model_names = [r["model_name"] for r in results]
    accuracies = [r["accuracy"] for r in results]
    training_times = [r["training_time"] for r in results]

    plt.figure(figsize=(10, 6))
    plt.scatter(
        training_times, accuracies, s=200, alpha=0.6, c="blue", edgecolors="black"
    )

    # Annotate each point with model name
    for i, model_name in enumerate(model_names):
        plt.annotate(
            model_name,
            (training_times[i], accuracies[i]),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
        )

    plt.xlabel("Training Time (seconds)", fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    plt.title("Model Accuracy vs Training Time", fontsize=14, fontweight="bold")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def predict_with_all_models(results, image_path):
    """Predict with all trained models and visualize results."""
    path = Path(image_path)
    if not path.exists():
        print(f"Error: file not found: {image_path}")
        return

    # Load and extract features from image
    with Image.open(path) as image:
        features = extract_features(image)
        image_for_plot = image.convert("RGB")

    # Get predictions from all models
    predictions = {}
    for result in results:
        model_name = result["model_name"]
        model = result["model"]
        prediction = model.predict([features])[0]
        predictions[model_name] = prediction

    # Create title with all predictions
    title_parts = [f"{name}: {predictions[name]}" for name in predictions.keys()]
    title = " | ".join(title_parts)

    # Display image with predictions
    plt.figure(figsize=(12, 6))
    plt.imshow(image_for_plot)
    plt.title(title, fontsize=12, wrap=True)
    plt.axis("off")
    plt.tight_layout()
    plt.show()

    # Print predictions
    print("\n=== Multi-Model Predictions ===")
    print(f"Image: {image_path}")
    for model_name, prediction in predictions.items():
        print(f"{model_name}: {prediction}")


def main():
    """Main function to run all tasks."""
    # Task 11.1 & 11.2 & 11.3: Train models and measure performance
    X_train, X_test, y_train, y_test = load_training_and_test_data()
    results = train_and_evaluate_models(X_train, X_test, y_train, y_test)

    # Task 11.4: Create accuracy vs training time plot
    print("\n=== Creating Accuracy vs Training Time Plot ===")
    plot_accuracy_vs_training_time(results)

    # Task 11.5: Visualize predictions from all models
    print("\n=== Multi-Model Image Prediction ===")
    test_image_path = DATASET_DIR / "test/forest/forest_0000.jpg"
    if not test_image_path.exists():
        # Use alternative test image if available
        print(
            f"Test image not found at {test_image_path}, searching for alternatives..."
        )
        test_dirs = list((DATASET_DIR / "test").glob("*/"))
        if test_dirs:
            first_class_dir = test_dirs[0]
            test_images = list(first_class_dir.glob("*.jpg"))
            if test_images:
                test_image_path = test_images[0]
                print(f"Using: {test_image_path}")

    predict_with_all_models(results, test_image_path)


if __name__ == "__main__":
    main()
