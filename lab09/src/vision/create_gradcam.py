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
from pathlib import Path
import numpy as np
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
OUTPUT_PATH = REPORT_PATHS / "gradcam_example.png"
IMAGE_PATH = DATASET_DIR / "test/forest/forest_0000.jpg"


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


def load_image(image_path):
    if not image_path.exists():
        print(f"Error: image not found: {image_path}")
        raise SystemExit(1)
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    display_transform = transforms.Compose([transforms.Resize((224, 224))])
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        display_image = display_transform(image)
        image_tensor = transform(image)
    image_tensor = image_tensor.unsqueeze(0)
    return image_tensor, display_image
