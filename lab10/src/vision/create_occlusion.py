from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from create_gradcam import (
    load_class_names,
    load_image,
    load_model,
    predict,
)

PROJECT_ROOT = Path(__file__).parent.parent.parent

DATA_PATH = PROJECT_ROOT / "data"
RAW_ROOT = DATA_PATH / "raw"
PROCESSED_ROOT = DATA_PATH / "processed"
DATASET_DIR = PROCESSED_ROOT

IMAGE_PATH = DATA_PATH / "processed/test/river/river_0000.jpg"

MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODELS_DIR / "resnet18_transfer.pt"
CLASS_NAMES_PATH = MODELS_DIR / "resnet18_classes.txt"

REPORTS_PATH = PROJECT_ROOT / "reports"


def create_occlusion_map(
    model, image_tensor, predicted_class, patch_size=32, stride=16
):
    _, _, H, W = image_tensor.shape
    sensitivity = np.zeros((H, W))
    with torch.no_grad():
        original_output = model(image_tensor)
        original_probability = torch.softmax(original_output, dim=1)[
            0, predicted_class
        ].item()
    for y in range(0, H - patch_size, stride):
        for x in range(0, W - patch_size, stride):
            occluded = image_tensor.clone()
            occluded[:, :, y : y + patch_size, x : x + patch_size] = 0
            with torch.no_grad():
                output = model(occluded)
                probability = torch.softmax(output, dim=1)[0, predicted_class].item()
            drop = original_probability - probability
            sensitivity[y : y + patch_size, x : x + patch_size] += drop
    sensitivity -= sensitivity.min()
    sensitivity /= sensitivity.max() + 1e-8
    return sensitivity


def visualize_occlusion(image, sensitivity):
    image = image.resize((224, 224))
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title("Original")
    plt.axis("off")
    plt.subplot(1, 2, 2)
    plt.imshow(sensitivity, cmap="jet")
    plt.title("Occlusion Sensitivity")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(REPORTS_PATH / "gradcam_examples/occlusion.png")
    plt.show()


def main():
    class_names = load_class_names()
    model = load_model(class_names)
    image_path = IMAGE_PATH
    image, tensor = load_image(image_path)
    predicted_class = predict(model, tensor, class_names)
    sensitivity = create_occlusion_map(model, tensor, predicted_class)
    visualize_occlusion(image, sensitivity)


if __name__ == "__main__":
    main()
