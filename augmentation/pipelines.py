"""
Augmentation pipelines.

Each pipeline receives an image and returns
an augmented image.
"""

from augmentation.transforms import (
    adjust_brightness_contrast,
    gamma_correction,
    jpeg_compression,
    gaussian_noise,
    gaussian_blur,
    motion_blur,
    rotate,
    perspective_transform,
    affine_transform,
)


# ==========================================================
# Lighting Pipeline
# ==========================================================

def lighting_pipeline(image):
    """
    Simulates different lighting conditions.
    """

    image = adjust_brightness_contrast(image)
    image = gamma_correction(image)
    image = jpeg_compression(image)

    return image


# ==========================================================
# Camera Quality Pipeline
# ==========================================================

def camera_pipeline(image):
    """
    Simulates low-quality cameras.
    """

    image = gaussian_noise(image)
    image = gaussian_blur(image)
    image = motion_blur(image)

    return image


# ==========================================================
# Geometry Pipeline
# ==========================================================

def geometry_pipeline(image):
    """
    Simulates different viewing angles.
    """

    image = rotate(image)
    image = perspective_transform(image)
    image = affine_transform(image)

    return image


# ==========================================================
# Pipeline Dictionary
# ==========================================================

PIPELINES = {
    "light": lighting_pipeline,
    "camera": camera_pipeline,
    "geo": geometry_pipeline,
}


def get_pipeline(name):
    """
    Returns a pipeline by name.
    """

    if name not in PIPELINES:
        raise ValueError(f"Unknown pipeline: {name}")

    return PIPELINES[name]


def get_all_pipelines():
    """
    Returns all available pipelines.
    """

    return PIPELINES