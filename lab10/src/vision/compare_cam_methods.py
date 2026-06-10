from pathlib import Path

from create_gradcam import (
    create_cam_by_method,
    load_class_names,
    load_image,
    load_model,
    predict,
    visualize_multiple_cams,
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


def main():
    class_names = load_class_names()
    model = load_model(class_names)
    image_path = IMAGE_PATH
    image, tensor = load_image(image_path)
    predict(model, tensor, class_names)
    methods = ["GradCAM", "HiResCAM", "EigenCAM", "LayerCAM"]
    heatmaps = {}
    for method_name in methods:
        heatmaps[method_name] = create_cam_by_method(model, tensor, method_name)
    visualize_multiple_cams(image, heatmaps)


if __name__ == "__main__":
    main()
