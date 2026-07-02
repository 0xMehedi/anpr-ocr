"""
Augmentation pipelines for Bangladesh ANPR OCR.
"""

import cv2
import albumentations as A

from augmentation.config import (
    # Lighting
    BRIGHTNESS_LIMIT,
    CONTRAST_LIMIT,
    GAMMA_LIMIT,
    CLAHE_CLIP_LIMIT,
    JPEG_QUALITY,

    # Camera
    GAUSSIAN_NOISE_STD,
    GAUSSIAN_BLUR,
    MOTION_BLUR,

    # Geometry
    ROTATION_LIMIT,
    PERSPECTIVE_SCALE,
    SCALE_LIMIT,
    TRANSLATE_LIMIT,

    # Probabilities
    P_BRIGHTNESS,
    P_GAMMA,
    P_CLAHE,
    P_JPEG,

    P_NOISE,
    P_GAUSSIAN_BLUR,
    P_MOTION_BLUR,

    P_ROTATION,
    P_PERSPECTIVE,
    P_AFFINE
)

# ==========================================================
# Lighting Pipeline
# ==========================================================

lighting_pipeline = A.Compose([
    A.RandomBrightnessContrast(
        brightness_limit=BRIGHTNESS_LIMIT,
        contrast_limit=CONTRAST_LIMIT,
        p=P_BRIGHTNESS,
    ),

    A.RandomGamma(
        gamma_limit=GAMMA_LIMIT,
        p=P_GAMMA,
    ),

    A.CLAHE(
        clip_limit=CLAHE_CLIP_LIMIT,
        tile_grid_size=(8, 8),
        p=P_CLAHE,
    ),

    A.ImageCompression(
        quality_range=JPEG_QUALITY,
        p=P_JPEG,
    ),
])

# ==========================================================
# Camera Quality Pipeline
# ==========================================================

camera_pipeline = A.Compose([

    A.GaussNoise(
        std_range=GAUSSIAN_NOISE_STD,
        p=P_NOISE,
    ),

    A.GaussianBlur(
        blur_limit=GAUSSIAN_BLUR,
        p=P_GAUSSIAN_BLUR,
    ),

    A.MotionBlur(
        blur_limit=MOTION_BLUR,
        p=P_MOTION_BLUR,
    ),
])

# ==========================================================
# Geometry Pipeline
# ==========================================================

geometry_pipeline = A.Compose([

    A.Rotate(
        limit=ROTATION_LIMIT,
        border_mode=cv2.BORDER_REPLICATE,
        p=P_ROTATION,
    ),

    A.Perspective(
        scale=PERSPECTIVE_SCALE,
        keep_size=True,
        p=P_PERSPECTIVE,
    ),

    A.Affine(
        scale=(1 - SCALE_LIMIT, 1 + SCALE_LIMIT),
        translate_percent=(-TRANSLATE_LIMIT, TRANSLATE_LIMIT),
        rotate=0,
        shear=0,
        fit_output=False,
        mode=cv2.BORDER_REPLICATE,
        p=P_AFFINE,
    ),
])

# ==========================================================
# Helper Functions
# ==========================================================

PIPELINES = {
    "light": lighting_pipeline,
    "camera": camera_pipeline,
    "geo": geometry_pipeline,
}


def get_pipeline(name: str):
    """
    Returns a single augmentation pipeline.

    Parameters
    ----------
    name : str
        light | camera | geo

    Returns
    -------
    albumentations.Compose
    """

    if name not in PIPELINES:
        raise ValueError(f"Unknown pipeline: {name}")

    return PIPELINES[name]


def get_all_pipelines():
    """
    Returns all augmentation pipelines.
    """

    return PIPELINES