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

# One original + three augmented versions
OUTPUT_MULTIPLIER = 4

# Random seed for reproducibility
RANDOM_SEED = 42

# Image extension for augmented images
IMAGE_EXTENSION = ".jpg"

# ==========================================================
# Lighting Augmentation
# ==========================================================

BRIGHTNESS_LIMIT = 0.20
CONTRAST_LIMIT = 0.20

GAMMA_LIMIT = (80, 120)

CLAHE_CLIP_LIMIT = 2.0

JPEG_QUALITY = (60, 95)

# ==========================================================
# Camera Quality Augmentation
# ==========================================================

GAUSSIAN_NOISE_STD = (10, 40)

GAUSSIAN_BLUR = (3, 5)

MOTION_BLUR = 5

# ==========================================================
# Geometry Augmentation
# ==========================================================

ROTATION_LIMIT = 3

PERSPECTIVE_SCALE = (0.02, 0.05)

SCALE_LIMIT = 0.05

TRANSLATE_LIMIT = 0.02

# ==========================================================
# Probabilities
# ==========================================================

P_BRIGHTNESS = 0.8
P_GAMMA = 0.5
P_CLAHE = 0.3
P_JPEG = 0.5

P_NOISE = 0.6
P_GAUSSIAN_BLUR = 0.5
P_MOTION_BLUR = 0.4

P_ROTATION = 0.7
P_PERSPECTIVE = 0.5
P_AFFINE = 0.5

# ==========================================================
# Adaptive Augmentation
# ==========================================================

# Images with area below this percentile
# will receive lighter augmentations.
LOW_RESOLUTION_PERCENTILE = 25

# ==========================================================
# Logging
# ==========================================================

SAVE_REPORT = True

VERBOSE = True