from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from image_dataset import EuroSATDataset
from torch import nn
from cnn_model import SimpleCNN
from pathlib import Path
import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_PATH = PROJECT_ROOT / "data"
RAW_ROOT = DATA_PATH / "raw"
PROCESSED_ROOT = DATA_PATH / "processed"

TRAIN_DIR = PROCESSED_ROOT / "train"
TEST_DIR = PROCESSED_ROOT / "test"
BATCH_SIZE = 16

EPOCHS = 8
LEARNING_RATE = 0.001

MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODELS_DIR / "cnn_model.pt"
CLASS_NAMES_PATH = MODELS_DIR / "cnn_classes.txt"

REPORTS_PATH = PROJECT_ROOT / "reports"
CONFUSION_MATRIX_PATH = REPORTS_PATH / "confusion_matrix.png"


def load_class_names():
    if not CLASS_NAMES_PATH.exists():
        print(f"Error: class file not found: {CLASS_NAMES_PATH}")
        print("Train the model first.")
        raise SystemExit(1)
    with open(CLASS_NAMES_PATH, "r") as f:
        class_names = [line.strip() for line in f.readlines() if line.strip()]
    return class_names


def load_model(class_names):
    if not MODEL_PATH.exists():
        print(f"Error: model file not found: {MODEL_PATH}")
        print("Train the model first.")
        raise SystemExit(1)
    model = SimpleCNN(num_classes=len(class_names))
    state_dict = torch.load(MODEL_PATH, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model


def predict_image(model, class_names, image_path):
    path = Path(image_path)
    if not path.exists():
        print(f"Error: image not found: {path}")
        return
    transform = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor()])
    with Image.open(path) as image:
        image = image.convert("RGB")
        image_for_plot = image.copy()
        image_tensor = transform(image)
        image_tensor = image_tensor.unsqueeze(0)
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted_index = torch.max(probabilities, dim=1)
    predicted_class = class_names[predicted_index.item()]
    print("=== CNN Prediction ===")
    print(f"Image: {image_path}")
    print(f"Predicted class: {predicted_class}")
    print(f"Confidence: {confidence.item():.4f}")
    plt.imshow(image_for_plot)
    plt.title(f"Prediction: {predicted_class}\nConfidence: {confidence.item():.4f}")
    plt.axis("off")
    plt.show()


def generate_convolution_matrix(model, class_names):
    test_dataset = EuroSATDataset(root_dir=TEST_DIR, transform=transforms.ToTensor())
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    all_labels = []
    all_predictions = []
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, dim=1)
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
    cm = confusion_matrix(all_labels, all_predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.savefig(CONFUSION_MATRIX_PATH, dpi=100, bbox_inches="tight")
    plt.show()


def main():
    class_names = load_class_names()
    model = load_model(class_names)
    # image_path = TEST_DIR / "forest/forest_0000.jpg"
    # image_path = RAW_ROOT / "eurosat/2750/Highway/Highway_1.jpg"
    # image_path = DATA_PATH / "inference_samples/noise.jpg"
    # predict_image(model, class_names, image_path)
    generate_convolution_matrix(model, class_names)


if __name__ == "__main__":
    main()
