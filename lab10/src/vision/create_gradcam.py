import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter
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


def pil_to_tensor(image_pil):
    """Convert a PIL image to a model input tensor (batch dim added)."""
    transform = transforms.Compose(
        [transforms.Resize((224, 224)), transforms.ToTensor()]
    )
    return transform(image_pil).unsqueeze(0)


def predict(model, image_tensor, class_names):
    """Return predicted index, class name and confidence."""
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, dim=1)
    predicted_class = class_names[predicted.item()]
    print(f"Prediction: {predicted_class} (confidence={confidence.item():.4f})")
    return predicted.item(), predicted_class, confidence.item()


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


# --- Sensitivity analysis utilities -----------------------------------------
def add_gaussian_noise(pil_img, std=0.06):
    arr = np.array(pil_img).astype(np.float32) / 255.0
    noise = np.random.normal(0, std, arr.shape).astype(np.float32)
    arr = np.clip(arr + noise, 0.0, 1.0)
    arr = (arr * 255).astype(np.uint8)
    return Image.fromarray(arr)


def threshold_cam(cam, thresh=0.5):
    return (cam >= thresh).astype(np.uint8)


def pearson_corr(a, b):
    # compute Pearson correlation between flattened arrays; return nan-safe float
    a_f = a.flatten()
    b_f = b.flatten()
    if np.all(a_f == a_f[0]) or np.all(b_f == b_f[0]):
        return float("nan")
    return float(np.corrcoef(a_f, b_f)[0, 1])


def jaccard_index(bin_a, bin_b):
    inter = np.logical_and(bin_a, bin_b).sum()
    union = np.logical_or(bin_a, bin_b).sum()
    if union == 0:
        return 1.0 if inter == 0 else 0.0
    return float(inter / union)


def save_side_by_side(
    original_img, original_cam, transformed_img, transformed_cam, out_path, titles=None
):
    """Save a 2x2 figure: original image, original overlay, transformed image, transformed overlay."""
    original_img = original_img.resize((224, 224))
    transformed_img = transformed_img.resize((224, 224))
    orig_arr = np.array(original_img).astype(np.float32) / 255.0
    trans_arr = np.array(transformed_img).astype(np.float32) / 255.0

    orig_vis = show_cam_on_image(orig_arr, original_cam, use_rgb=True)
    trans_vis = show_cam_on_image(trans_arr, transformed_cam, use_rgb=True)
    diff = transformed_cam - original_cam
    diff_norm = (diff - diff.min()) / (diff.max() - diff.min() + 1e-8)

    plt.figure(figsize=(10, 8))
    plt.subplot(2, 2, 1)
    plt.imshow(original_img)
    plt.title("Original Image" if not titles else titles[0])
    plt.axis("off")

    plt.subplot(2, 2, 2)
    plt.imshow(orig_vis)
    plt.title("Original Grad-CAM" if not titles else titles[1])
    plt.axis("off")

    plt.subplot(2, 2, 3)
    plt.imshow(transformed_img)
    plt.title("Transformed Image" if not titles else titles[2])
    plt.axis("off")

    plt.subplot(2, 2, 4)
    plt.imshow(show_cam_on_image(trans_arr, transformed_cam, use_rgb=True))
    plt.title("Transformed Grad-CAM" if not titles else titles[3])
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def run_sensitivity_for_image(model, class_names, image_path, output_root):
    """Run transforms, predict, compute Grad-CAMs and metrics, save visualizations and summary CSV."""
    output_root.mkdir(parents=True, exist_ok=True)

    with Image.open(image_path) as im:
        original = im.convert("RGB")

    # transforms to try
    transforms_list = [
        ("original", lambda img: img.copy()),
        ("hflip", lambda img: img.transpose(Image.FLIP_LEFT_RIGHT)),
        ("rot90", lambda img: img.rotate(90, expand=True)),
        ("gaussian_blur", lambda img: img.filter(ImageFilter.GaussianBlur(radius=3))),
        ("bright_up", lambda img: ImageEnhance.Brightness(img).enhance(1.5)),
        ("bright_down", lambda img: ImageEnhance.Brightness(img).enhance(0.6)),
        ("noise", lambda img: add_gaussian_noise(img, std=0.06)),
    ]

    orig_tensor = pil_to_tensor(original)
    orig_idx, orig_class, orig_conf = predict(model, orig_tensor, class_names)
    orig_cam = create_heatmap(model, orig_tensor)
    orig_bin = threshold_cam(orig_cam, 0.5)

    rows = []
    for name, fn in transforms_list:
        t_img = fn(original)
        t_tensor = pil_to_tensor(t_img)

        idx, pred_class, conf = predict(model, t_tensor, class_names)
        t_cam = create_heatmap(model, t_tensor)

        corr = pearson_corr(orig_cam, t_cam)
        jacc = jaccard_index(orig_bin, threshold_cam(t_cam, 0.5))

        out_img_path = output_root / f"{image_path.stem}_{name}.png"
        save_side_by_side(original, orig_cam, t_img, t_cam, out_img_path)

        rows.append(
            {
                "image": str(image_path),
                "transform": name,
                "predicted_class": pred_class,
                "confidence": conf,
                "pearson_corr_with_original_cam": corr,
                "jaccard_threshold_0.5": jacc,
                "output_image": str(out_img_path),
            }
        )

    # save CSV
    csv_path = output_root / f"{image_path.stem}_summary.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
        fieldnames = list(rows[0].keys())
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print(f"Saved sensitivity outputs to: {output_root}")
    return csv_path


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

    class_names = load_class_names()
    model = load_model(class_names)

    for _type, imgs in images.items():
        for img in imgs:
            image_path = img
            image, tensor = load_image(image_path)
            predict(model, tensor, class_names)
            heatmap = create_heatmap(model, tensor)
            visualize(image, heatmap)

    sensitivity_output_root = REPORTS_PATH / "gradcam_sensitivity"
    sensitivity_output_root.mkdir(parents=True, exist_ok=True)

    # Run sensitivity analysis for the first image of each class to limit runtime
    for cls, imgs in images.items():
        first_image = imgs[0]
        print(f"Running sensitivity analysis for {first_image}")
        run_sensitivity_for_image(
            model, class_names, first_image, sensitivity_output_root
        )


if __name__ == "__main__":
    main()
