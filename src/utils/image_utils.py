"""Image I/O and processing utilities."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def load_image(path: str | Path) -> Image.Image:
    """Load an image from disk."""
    return Image.open(str(path)).convert("RGB")


def save_image(image: Image.Image | np.ndarray, path: str | Path) -> None:
    """Save an image to disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(image, np.ndarray):
        image = numpy_to_pil(image)
    image.save(str(path))


def numpy_to_pil(array: np.ndarray) -> Image.Image:
    """Convert numpy array to PIL Image."""
    if array.dtype in (np.float32, np.float64):
        array = (array * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(array)


def pil_to_numpy(image: Image.Image) -> np.ndarray:
    """Convert PIL Image to numpy float32 array in [0, 1]."""
    return np.array(image).astype(np.float32) / 255.0


def depth_to_controlnet(depth: np.ndarray) -> Image.Image:
    """Convert a depth map (float, any range) to ControlNet-compatible format.

    ControlNet depth convention: near=white (255), far=black (0).
    """
    d_min, d_max = depth.min(), depth.max()
    if d_max - d_min < 1e-6:
        normalized = np.zeros_like(depth, dtype=np.uint8)
    else:
        # Invert: near = high value
        normalized = ((d_max - depth) / (d_max - d_min) * 255).astype(np.uint8)
    return Image.fromarray(normalized, mode="L")


def compute_silhouette_iou(image_a: Image.Image, image_b: Image.Image) -> float:
    """Compute IoU between silhouettes of two images (non-white pixels)."""
    a = np.array(image_a.convert("L")) < 250
    b = np.array(image_b.convert("L")) < 250
    intersection = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 1.0
    return float(intersection / union)
