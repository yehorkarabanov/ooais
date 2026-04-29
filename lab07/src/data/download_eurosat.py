from pathlib import Path

from torchvision.datasets import EuroSAT

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_PATH = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_PATH / "raw"


def main():
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    dataset = EuroSAT(root=str(RAW_DATA_DIR), download=True)
    print(f"Downloaded dataset with {len(dataset)} images")
    print(f"Classes: {dataset.classes}")


if __name__ == "__main__":
    main()
