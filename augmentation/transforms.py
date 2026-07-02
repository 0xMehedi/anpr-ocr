"""
Image transformation functions using OpenCV and NumPy.

Each function takes an OpenCV image (NumPy array)
and returns the transformed image.
"""

import random
import cv2
import numpy as np

from augmentation.config import (
    BRIGHTNESS_RANGE,
    CONTRAST_RANGE,
    GAMMA_RANGE,
    JPEG_QUALITY,
    CLAHE_CLIP_LIMIT,
    CLAHE_TILE_GRID_SIZE,
    GAUSSIAN_NOISE_STD,
    GAUSSIAN_BLUR_KERNEL,
    MOTION_BLUR_KERNEL,
    ROTATION_LIMIT,
    PERSPECTIVE_SHIFT,
    SCALE_RANGE,
    TRANSLATION_PERCENT,
)






from augmentation.config import (
    BRIGHTNESS_RANGE,
    CONTRAST_RANGE,
    GAMMA_RANGE,
    JPEG_QUALITY,
    GAUSSIAN_NOISE_STD,
    GAUSSIAN_BLUR_KERNEL,
    MOTION_BLUR_KERNEL,
    ROTATION_LIMIT,
    PERSPECTIVE_SHIFT,
    SCALE_RANGE,
    TRANSLATION_PERCENT,
)

# ==========================================================
# Lighting
# ==========================================================

def adjust_brightness_contrast(image):

    alpha = random.uniform(*CONTRAST_RANGE)

    brightness = random.uniform(*BRIGHTNESS_RANGE)

    beta = int((brightness - 1.0) * 100)

    return cv2.convertScaleAbs(
        image,
        alpha=alpha,
        beta=beta
    )


def gamma_correction(image):
    gamma = random.uniform(*GAMMA_RANGE)

    inv_gamma = 1.0 / gamma

    table = np.array([
        ((i / 255.0) ** inv_gamma) * 255
        for i in np.arange(256)
    ]).astype("uint8")

    return cv2.LUT(image, table)

def clahe(image):
    """
    Apply Contrast Limited Adaptive Histogram Equalization (CLAHE).
    """

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    l, a, b = cv2.split(lab)

    clahe_filter = cv2.createCLAHE(
        clipLimit=CLAHE_CLIP_LIMIT,
        tileGridSize=CLAHE_TILE_GRID_SIZE
    )

    l = clahe_filter.apply(l)

    merged = cv2.merge((l, a, b))

    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

def jpeg_compression(image):

    quality = random.randint(*JPEG_QUALITY)

    encode_param = [
        int(cv2.IMWRITE_JPEG_QUALITY),
        quality
    ]

    success, encoded = cv2.imencode(".jpg", image, encode_param)

    if not success:
        return image

    return cv2.imdecode(encoded, cv2.IMREAD_COLOR)

# ==========================================================
# Camera Quality
# ==========================================================

def gaussian_noise(image):

    std = random.uniform(*GAUSSIAN_NOISE_STD)

    noise = np.random.normal(
        0,
        std,
        image.shape
    ).astype(np.float32)

    noisy = image.astype(np.float32) + noise

    noisy = np.clip(noisy, 0, 255)

    return noisy.astype(np.uint8)


def gaussian_blur(image):

    kernel = random.choice(GAUSSIAN_BLUR_KERNEL)

    if kernel % 2 == 0:
        kernel += 1

    return cv2.GaussianBlur(
        image,
        (kernel, kernel),
        0
    )


def motion_blur(image):

    kernel_size = MOTION_BLUR_KERNEL

    kernel = np.zeros((kernel_size, kernel_size))

    kernel[kernel_size // 2, :] = np.ones(kernel_size)

    kernel /= kernel_size

    return cv2.filter2D(
        image,
        -1,
        kernel
    )

# ==========================================================
# Geometry
# ==========================================================

def rotate(image):

    h, w = image.shape[:2]

    angle = random.uniform(
        -ROTATION_LIMIT,
        ROTATION_LIMIT
    )

    matrix = cv2.getRotationMatrix2D(
        (w // 2, h // 2),
        angle,
        1.0
    )

    return cv2.warpAffine(
        image,
        matrix,
        (w, h),
        borderMode=cv2.BORDER_REPLICATE
    )


def perspective_transform(image):

    h, w = image.shape[:2]

    shift = PERSPECTIVE_SHIFT

    dx = int(w * shift)
    dy = int(h * shift)

    src = np.float32([
        [0, 0],
        [w, 0],
        [0, h],
        [w, h]
    ])

    dst = np.float32([
        [random.randint(0, dx), random.randint(0, dy)],
        [w-random.randint(0, dx), random.randint(0, dy)],
        [random.randint(0, dx), h-random.randint(0, dy)],
        [w-random.randint(0, dx), h-random.randint(0, dy)]
    ])

    matrix = cv2.getPerspectiveTransform(src, dst)

    return cv2.warpPerspective(
        image,
        matrix,
        (w, h),
        borderMode=cv2.BORDER_REPLICATE
    )


def affine_transform(image):

    h, w = image.shape[:2]

    scale = random.uniform(*SCALE_RANGE)

    tx = random.uniform(
        -TRANSLATION_PERCENT,
        TRANSLATION_PERCENT
    ) * w

    ty = random.uniform(
        -TRANSLATION_PERCENT,
        TRANSLATION_PERCENT
    ) * h

    matrix = np.float32([
        [scale, 0, tx],
        [0, scale, ty]
    ])

    return cv2.warpAffine(
        image,
        matrix,
        (w, h),
        borderMode=cv2.BORDER_REPLICATE
    )