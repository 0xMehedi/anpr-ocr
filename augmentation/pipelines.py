"""
Professional augmentation pipelines.

Instead of applying every transform, each pipeline
randomly selects a subset of realistic transforms.
"""

import random

from augmentation.transforms import (
    adjust_brightness_contrast,
    gamma_correction,
    clahe,
    jpeg_compression,
    gaussian_noise,
    gaussian_blur,
    motion_blur,
    rotate,
    perspective_transform,
    affine_transform,
)


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
# Helper
# ==========================================================

def apply_random(image, transforms, min_ops, max_ops):

    n = random.randint(min_ops, max_ops)

    selected = random.sample(transforms, n)

    for transform in selected:
        image = transform(image)

    return image

# ==========================================================
# Lighting
# ==========================================================

def lighting_pipeline(image):

    transforms = [

    adjust_brightness_contrast,

    gamma_correction,

    clahe,

    jpeg_compression,
    
    ]

    return apply_random(
        image,
        transforms,
        min_ops=2,
        max_ops=3
    )

# ==========================================================
# Camera
# ==========================================================

def camera_pipeline(image):

    transforms = [

        gaussian_noise,

        gaussian_blur,

        motion_blur

    ]

    return apply_random(
        image,
        transforms,
        min_ops=1,
        max_ops=2
    )

# ==========================================================
# Geometry
# ==========================================================

def geometry_pipeline(image):

    transforms = [

        rotate,

        perspective_transform,

        affine_transform

    ]

    return apply_random(
        image,
        transforms,
        min_ops=1,
        max_ops=2
    )

# ==========================================================
# Registry
# ==========================================================

PIPELINES = {

    "light": lighting_pipeline,

    "camera": camera_pipeline,

    "geo": geometry_pipeline,

}

def get_pipeline(name):

    return PIPELINES[name]


def get_all_pipelines():

    return PIPELINES