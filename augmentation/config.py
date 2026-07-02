"""
Configuration for the Bangladesh ANPR OCR augmentation pipeline.
"""

from pathlib import Path

# ==========================================================
# Project Paths
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TRAIN_IMAGE_DIR = PROJECT_ROOT / "data" / "train" / "images"
TRAIN_CSV = PROJECT_ROOT / "data" / "annotations" / "train.csv"

OUTPUT_DIR = PROJECT_ROOT / "data" / "train_augmented"
OUTPUT_IMAGE_DIR = OUTPUT_DIR / "images"
OUTPUT_CSV = OUTPUT_DIR / "labels.csv"

REPORT_DIR = PROJECT_ROOT / "results"
REPORT_CSV = REPORT_DIR / "augmentation_report.csv"

# ==========================================================
# Dataset Configuration
# ==========================================================

# Original + 3 augmentations = 4 images per sample
OUTPUT_MULTIPLIER = 4

RANDOM_SEED = 42

IMAGE_EXTENSION = ".jpg"

# ==========================================================
# Lighting Augmentation
# ==========================================================

BRIGHTNESS_RANGE = (0.8, 1.2)
CONTRAST_RANGE = (0.8, 1.2)

GAMMA_RANGE = (0.8, 1.2)

JPEG_QUALITY = (60, 95)

# ==========================================================
# Camera Quality
# ==========================================================

GAUSSIAN_NOISE_STD = (5, 20)

GAUSSIAN_BLUR_KERNEL = (3, 5)

MOTION_BLUR_KERNEL = 5

# ==========================================================
# Geometry
# ==========================================================

ROTATION_LIMIT = 3

PERSPECTIVE_SHIFT = 0.03

SCALE_RANGE = (0.95, 1.05)

TRANSLATION_PERCENT = 0.02

# ==========================================================
# Adaptive Augmentation
# ==========================================================

LOW_RESOLUTION_PERCENTILE = 25

# ==========================================================
# Logging
# ==========================================================

SAVE_REPORT = True

VERBOSE = True