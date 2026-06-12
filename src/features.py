from __future__ import annotations

import cv2
import numpy as np
from skimage import color
from skimage.feature import hog
from tqdm import tqdm


def _to_uint8_rgb(image: np.ndarray) -> np.ndarray:
    arr = image
    if arr.max() <= 1.0:
        arr = arr * 255.0
    return np.clip(arr, 0, 255).astype(np.uint8)


def make_flat_gray_features(images: np.ndarray) -> np.ndarray:
    features = []
    for image in tqdm(images, desc="Extract gray pixel features"):
        gray = color.rgb2gray(_to_uint8_rgb(image))
        features.append(gray.reshape(-1))
    return np.asarray(features, dtype=np.float32)


def _color_histogram(image: np.ndarray, bins: int = 16) -> np.ndarray:
    rgb = _to_uint8_rgb(image)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    feats = []
    for arr in (rgb, hsv):
        for channel in range(3):
            hist, _ = np.histogram(arr[:, :, channel], bins=bins, range=(0, 256), density=True)
            feats.append(hist.astype(np.float32))
    return np.concatenate(feats)


def _hog_feature(image: np.ndarray) -> np.ndarray:
    gray = color.rgb2gray(_to_uint8_rgb(image))
    return hog(
        gray,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        feature_vector=True,
    ).astype(np.float32)


def make_hog_color_features(images: np.ndarray) -> np.ndarray:
    features = []
    for image in tqdm(images, desc="Extract HOG/color features"):
        features.append(np.concatenate([_hog_feature(image), _color_histogram(image)]))
    return np.asarray(features, dtype=np.float32)
