import random
from pathlib import Path

import torch
from segmentation_dataset import SyntheticSegmentationDataset
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from unet_model import SmallUNet

PROJECT_ROOT = Path(__file__).parent.parent.parent

DATA_PATH = PROJECT_ROOT / "data"

SEGMENTATION_DIR = DATA_PATH / "segmentation"
IMAGE_DIR = SEGMENTATION_DIR / "images"
MASK_DIR = SEGMENTATION_DIR / "masks"

IMAGE_PATH = IMAGE_DIR / "scene_0000.png"
MASK_PATH = MASK_DIR / "scene_0000.png"

REPORTS_DIR = PROJECT_ROOT / "reports"
SEGMENTATION_OUTPUT_DIR = REPORTS_DIR / "segmentation_examples"
OUTPUT_PATH = SEGMENTATION_OUTPUT_DIR / "dataset_example.png"

MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODELS_DIR / "small_unet.pt"
REPORT_PATH = REPORTS_DIR / "segmentation_report.txt"

BATCH_SIZE = 8
EPOCHS = 10
LEARNING_RATE = 0.001
NUM_CLASSES = 4
RANDOM_SEED = 42
CLASS_NAMES = {
    0: "background",
    1: "vegetation",
    2: "water",
    3: "urban",
}


def create_dataloaders():
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = SyntheticSegmentationDataset(
        image_dir=IMAGE_DIR, mask_dir=MASK_DIR, transform=transform
    )
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(
        dataset,
        [train_size, test_size],
        generator=torch.Generator().manual_seed(RANDOM_SEED),
    )
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    print("=== Segmentation DataLoaders ===")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Testing samples: {len(test_dataset)}")
    images, masks = next(iter(train_loader))
    print(f"Batch image shape: {images.shape}")
    print(f"Batch mask shape: {masks.shape}")
    return train_loader, test_loader


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def train_model(model, train_loader, device):
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0.0
        for images, masks in train_loader:
            images = images.to(device)
            masks = masks.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_function(outputs, masks)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        average_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{EPOCHS}, Loss: {average_loss:.4f}")


def evaluate_model(model, test_loader, device):
    model.eval()
    correct_pixels = 0
    total_pixels = 0
    class_correct_pixels = dict.fromkeys(range(NUM_CLASSES), 0)
    class_total_pixels = dict.fromkeys(range(NUM_CLASSES), 0)
    with torch.no_grad():
        for images, masks in test_loader:
            images = images.to(device)
            masks = masks.to(device)
            outputs = model(images)
            predictions = torch.argmax(outputs, dim=1)
            correct_pixels += (predictions == masks).sum().item()
            total_pixels += masks.numel()

            for class_id in range(NUM_CLASSES):
                class_mask = masks == class_id
                class_total_pixels[class_id] += class_mask.sum().item()
                class_correct_pixels[class_id] += (
                    (predictions == masks) & class_mask
                ).sum().item()

    accuracy = correct_pixels / total_pixels
    class_accuracies = {}
    for class_id in range(NUM_CLASSES):
        total_for_class = class_total_pixels[class_id]
        if total_for_class == 0:
            class_accuracies[class_id] = None
        else:
            class_accuracies[class_id] = (
                class_correct_pixels[class_id] / total_for_class
            )
    print("=== Segmentation Evaluation ===")
    print(f"Pixel accuracy: {accuracy:.4f}")
    for class_id in range(NUM_CLASSES):
        class_name = CLASS_NAMES.get(class_id, f"class_{class_id}")
        class_accuracy = class_accuracies[class_id]
        if class_accuracy is None:
            print(f"{class_name.capitalize()} accuracy: N/A (no pixels in test set)")
        else:
            print(f"{class_name.capitalize()} accuracy: {class_accuracy:.4f}")
    return accuracy, class_accuracies


def save_model(model):
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Saved model: {MODEL_PATH}")


def save_report(accuracy, class_accuracies):
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w") as f:
        f.write("SEMANTIC SEGMENTATION REPORT\n")
        f.write("============================\n\n")
        f.write("Model: Small U-Net\n")
        f.write("Dataset: synthetic EO segmentation dataset\n")
        f.write(f"Classes: {NUM_CLASSES}\n")
        f.write(f"Epochs: {EPOCHS}\n")
        f.write(f"Learning rate: {LEARNING_RATE}\n")
        f.write(f"Pixel accuracy: {accuracy:.4f}\n\n")
        f.write("Class-wise pixel accuracy:\n")
        for class_id in range(NUM_CLASSES):
            class_name = CLASS_NAMES.get(class_id, f"class_{class_id}")
            class_accuracy = class_accuracies[class_id]
            if class_accuracy is None:
                f.write(f"- {class_name}: N/A (no pixels in test set)\n")
            else:
                f.write(f"- {class_name}: {class_accuracy:.4f}\n")
        f.write("\n")
        f.write("Interpretation:\n")
        f.write(
            "The model was trained to assign a land-cover class "
            "to every pixel in the image.\n"
        )
    print(f"Saved report: {REPORT_PATH}")


def main():
    random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    train_loader, test_loader = create_dataloaders()
    device = get_device()
    print(f"Using device: {device}")
    model = SmallUNet(num_classes=NUM_CLASSES)
    model = model.to(device)
    train_model(model, train_loader, device)
    accuracy, class_accuracies = evaluate_model(model, test_loader, device)
    save_model(model)
    save_report(accuracy, class_accuracies)


if __name__ == "__main__":
    main()
