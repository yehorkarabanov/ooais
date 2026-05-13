from pathlib import Path
import time
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms, models
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from src.vision.image_dataset import EuroSATDataset
from pathlib import Path
import torch
from torch import nn
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_PATH = PROJECT_ROOT / "data"
RAW_ROOT = DATA_PATH / "raw"
PROCESSED_ROOT = DATA_PATH / "processed"
DATASET_DIR = PROCESSED_ROOT

MODELS_PATH = PROJECT_ROOT / "models"
MODEL_PATH = MODELS_PATH / "image_model.joblib"

TRAIN_DIR = PROCESSED_ROOT / "train"
TEST_DIR = PROCESSED_ROOT / "test"


MODEL_PATH = MODELS_PATH / "resnet18_transfer.pt"
CLASS_NAMES_PATH = MODELS_PATH / "resnet18_classes.txt"
REPORT_PATHS = PROJECT_ROOT / "reports"
REPORT_PATH = REPORT_PATHS / "fine_tuning_report.txt"
CONFUSION_MATRIX_PATH = REPORT_PATHS / "fine_tuning_confusion_matrix.png"
OUTPUT_PATH = REPORT_PATHS / "transfer_prediction_gallery.png"

IMAGES_PER_CLASS = 3


def load_class_names():
    if not CLASS_NAMES_PATH.exists():
        print(f"Error: class file not found: {CLASS_NAMES_PATH}")
        print("Train the transfer model first.")
        raise SystemExit(1)
    with open(CLASS_NAMES_PATH, "r") as f:
        class_names = [line.strip() for line in f.readlines() if line.strip()]
    return class_names


def load_model(class_names):
    if not MODEL_PATH.exists():
        print(f"Error: model file not found: {MODEL_PATH}")
        print("Train the transfer model first.")
        raise SystemExit(1)
    model = models.resnet18(weights=None)
    input_features = model.fc.in_features
    model.fc = nn.Linear(input_features, len(class_names))
    state_dict = torch.load(MODEL_PATH, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model


def select_test_images(class_names):
    selected_images = []
    for class_name in class_names:
        class_dir = TEST_DIR / class_name
        image_paths = sorted(
            [
                path
                for path in class_dir.iterdir()
                if path.suffix.lower() in [".jpg", ".jpeg", ".png"]
            ]
        )
        for image_path in image_paths[:IMAGES_PER_CLASS]:
            selected_images.append((image_path, class_name))
    return selected_images


def predict_image(model, class_names, image_path, transform):
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        image_tensor = transform(image)
        image_tensor = image_tensor.unsqueeze(0)
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted_index = torch.max(probabilities, dim=1)
    predicted_class = class_names[predicted_index.item()]
    return predicted_class, confidence.item()


def create_gallery(model, class_names):
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    selected_images = select_test_images(class_names)
    rows = len(class_names)
    columns = IMAGES_PER_CLASS
    figure, axes = plt.subplots(rows, columns, figsize=(10, 8))
    for axis, (image_path, true_class) in zip(axes.flatten(), selected_images):
        predicted_class, confidence = predict_image(
            model, class_names, image_path, transform
        )
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            axis.imshow(image)
            axis.axis("off")
            axis.set_title(
                f"True: {true_class}\nPred: {predicted_class}\nConf: {confidence:.2f}"
            )
    plt.tight_layout()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_PATH)
    plt.close()
    print(f"Saved prediction gallery: {OUTPUT_PATH}")


def main():
    class_names = load_class_names()
    model = load_model(class_names)
    create_gallery(model, class_names)


if __name__ == "__main__":
    main()
