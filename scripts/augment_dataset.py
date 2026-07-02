from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from augmentation import augmenter

if __name__ == "__main__":
    print("Bangladesh ANPR OCR Dataset Augmentation")