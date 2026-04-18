"""Preprocessing helpers.

Important: classical CV isolation should generally run on the original image (or resize-only),
not on a globally normalized image, because normalization can change edge behavior and colors.
"""
import cv2
import numpy as np
from pathlib import Path
from typing import Union


def preprocess(image: np.ndarray, max_side: int = 1024) -> np.ndarray:
    """
    Resize-only preprocessing for CV pipeline.
    - If the longest side is larger than max_side, downscale (keeps aspect ratio).
    - No pixel normalization is applied here.
    """
    h, w = image.shape[:2]
    scale = max_side / max(h, w)
    if scale < 1:
        new_w, new_h = int(w * scale), int(h * scale)
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return image


def load_image(path: Union[str, Path]) -> np.ndarray:
    """Load image from path; returns BGR numpy array (OpenCV convention). Returns None if read fails."""
    return cv2.imread(str(path))
