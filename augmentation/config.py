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

# Original + Light + Camera + Geometry
OUTPUT_MULTIPLIER = 4

# Reproducibility
RANDOM_SEED = 42

# Image Extension
IMAGE_EXTENSION = ".jpg"

# ==========================================================
# Lighting Augmentation
# ==========================================================

# Contrast multiplier
CONTRAST_RANGE = (0.2, 2.0)

# Brightness multiplier
BRIGHTNESS_RANGE = (0.2, 2.0)

# Gamma correction
GAMMA_RANGE = (0.2, 2.2)

# JPEG Compression Quality
JPEG_QUALITY = (25, 95)

# CLAHE
CLAHE_CLIP_LIMIT = 2.5
CLAHE_TILE_GRID_SIZE = (8, 8)

# ==========================================================
# Camera Quality
# ==========================================================

# Gaussian Noise (standard deviation)
GAUSSIAN_NOISE_STD = (8, 25)

# Gaussian Blur kernel sizes
GAUSSIAN_BLUR_KERNEL = (3, 5, 7)

# Motion Blur kernel size
MOTION_BLUR_KERNEL = 7

# ==========================================================
# Geometry
# ==========================================================

# Rotation (degrees)
ROTATION_LIMIT = 5

# Perspective distortion
PERSPECTIVE_SHIFT = 0.05

# Scaling
SCALE_RANGE = (0.90, 1.10)

# Translation (% of image)
TRANSLATION_PERCENT = 0.03

# ==========================================================
# Adaptive Augmentation
# ==========================================================

# Images below this percentile receive lighter augmentation
LOW_RESOLUTION_PERCENTILE = 25

# ==========================================================
# Pipeline Behaviour
# ==========================================================

# Lighting pipeline
LIGHTING_MIN_OPERATIONS = 2
LIGHTING_MAX_OPERATIONS = 3

# Camera pipeline
CAMERA_MIN_OPERATIONS = 1
CAMERA_MAX_OPERATIONS = 2

# Geometry pipeline
GEOMETRY_MIN_OPERATIONS = 1
GEOMETRY_MAX_OPERATIONS = 2

# ==========================================================
# Logging
# ==========================================================

SAVE_REPORT = True

VERBOSE = True