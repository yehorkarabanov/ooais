from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from pytorch_grad_cam import EigenCAM, GradCAM, HiResCAM, LayerCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from torch import nn
from torchvision import models, transforms

PROJECT_ROOT = Path(__file__).parent.parent.parent

DATA_PATH = PROJECT_ROOT / "data"
RAW_ROOT = DATA_PATH / "raw"
PROCESSED_ROOT = DATA_PATH / "processed"
DATASET_DIR = PROCESSED_ROOT

IMAGE_PATH = DATA_PATH / "inference_samples/noise.jpg"

MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODELS_DIR / "resnet18_transfer.pt"
CLASS_NAMES_PATH = MODELS_DIR / "resnet18_classes.txt"

REPORTS_PATH = PROJECT_ROOT / "reports"


def load_class_names():
    with open(CLASS_NAMES_PATH) as f:
        class_names = [line.strip() for line in f]
    return class_names


def load_model(class_names):
    model = models.resnet18(weights=None)
    input_features = model.fc.in_features
    model.fc = nn.Linear(input_features, len(class_names))
    state_dict = torch.load(MODEL_PATH, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model


def load_image(image_path):
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose(
        [transforms.Resize((224, 224)), transforms.ToTensor()]
    )
    tensor = transform(image)
    return image, tensor.unsqueeze(0)


def predict(model, image_tensor, class_names):
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, dim=1)
    predicted_class = class_names[predicted.item()]
    print(f"Prediction: {predicted_class}")
    print(f"Confidence: {confidence.item():.4f}")
    return predicted.item()


def create_heatmap(model, image_tensor):
    target_layers = [model.layer4[-1]]
    cam = GradCAM(model=model, target_layers=target_layers)
    grayscale_cam = cam(input_tensor=image_tensor)
    return grayscale_cam[0]


def visualize(image, heatmap):
    image = image.resize((224, 224))
    image_array = np.array(image).astype(np.float32) / 255.0
    visualization = show_cam_on_image(image_array, heatmap, use_rgb=True)

    output_dir = PROJECT_ROOT / "reports" / "gradcam_examples"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "example.png"

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(visualization)
    plt.title("Grad-CAM")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.show()


def create_cam_by_method(model, image_tensor, method_name):
    target_layers = [model.layer4[-1]]
    methods = {
        "GradCAM": GradCAM,
        "HiResCAM": HiResCAM,
        "EigenCAM": EigenCAM,
        "LayerCAM": LayerCAM,
    }
    cam_class = methods[method_name]
    cam = cam_class(model=model, target_layers=target_layers)
    grayscale_cam = cam(input_tensor=image_tensor)
    return grayscale_cam[0]


def visualize_multiple_cams(image, heatmaps):
    image = image.resize((224, 224))
    image_array = np.array(image).astype(np.float32) / 255.0
    plt.figure(figsize=(16, 8))
    plt.subplot(2, 3, 1)
    plt.imshow(image)
    plt.title("Original")
    plt.axis("off")
    index = 2
    for method_name, heatmap in heatmaps.items():
        visualization = show_cam_on_image(image_array, heatmap, use_rgb=True)
        plt.subplot(2, 3, index)
        plt.imshow(visualization)
        plt.title(method_name)
        plt.axis("off")
        index += 1
    plt.tight_layout()
    output_path = REPORTS_PATH / "gradcam_examples/cam_methods_comparison.png"
    plt.savefig(output_path)
    plt.show()
    print(f"Saved: {output_path}")


def main():
    images = {
        "river": [
            PROCESSED_ROOT / "test/river/river_0030.jpg",
            PROCESSED_ROOT / "test/river/river_0035.jpg",
            PROCESSED_ROOT / "test/river/river_0055.jpg",
        ],
        "forest": [
            PROCESSED_ROOT / "test/forest/forest_0030.jpg",
            PROCESSED_ROOT / "test/forest/forest_0035.jpg",
            PROCESSED_ROOT / "test/forest/forest_0055.jpg",
        ],
        "residential": [
            PROCESSED_ROOT / "test/residential/residential_0030.jpg",
            PROCESSED_ROOT / "test/residential/residential_0035.jpg",
            PROCESSED_ROOT / "test/residential/residential_0055.jpg",
        ],
    }
    for type, imgs in images.items():
        for img in imgs:
            class_names = load_class_names()
            model = load_model(class_names)
            image_path = img
            image, tensor = load_image(image_path)
            predict(model, tensor, class_names)
            heatmap = create_heatmap(model, tensor)
            visualize(image, heatmap)


if __name__ == "__main__":
    main()
