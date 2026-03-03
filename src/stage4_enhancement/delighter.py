"""Texture delighting: remove baked lighting to produce albedo-only textures.

Uses IC-Light or classical intrinsic image decomposition as fallback.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from PIL import Image

from src.config import Stage4Config
from src.vram_manager import VRAMManager

logger = logging.getLogger(__name__)


def delight_textures(
    multiview_images: list[list[Image.Image]],
    config: Stage4Config,
    vram: VRAMManager,
) -> list[list[Image.Image]]:
    """Remove baked lighting from all texture views for all objects.

    Args:
        multiview_images: For each object, list of multi-view texture images.
        config: Stage 4 configuration.
        vram: VRAM manager.

    Returns:
        Same structure, but with lighting removed (albedo only).
    """
    backend = config.delight.backend

    if backend == "ic_light":
        return _delight_ic_light(multiview_images, config, vram)
    elif backend == "intrinsic":
        return _delight_intrinsic(multiview_images)
    else:
        logger.warning(f"Unknown delight backend '{backend}', returning original images")
        return multiview_images


def _delight_ic_light(
    all_views: list[list[Image.Image]],
    config: Stage4Config,
    vram: VRAMManager,
) -> list[list[Image.Image]]:
    """Remove lighting using IC-Light (diffusion-based relighting)."""
    try:
        # IC-Light is typically loaded as a custom diffusion pipeline
        def _load_ic_light():
            import torch
            from diffusers import StableDiffusionPipeline

            pipe = StableDiffusionPipeline.from_pretrained(
                "lllyasviel/ic-light-fbc-normals",
                torch_dtype=torch.float16,
                safety_checker=None,
            )
            pipe.to("cuda")
            return pipe

        all_delighted = []

        with vram.load_model("ic_light", _load_ic_light) as pipe:
            for i, views in enumerate(all_views):
                delighted_views = []
                for j, view in enumerate(views):
                    try:
                        result = pipe(
                            prompt="albedo, flat lighting, no shadows, no highlights",
                            image=view,
                            num_inference_steps=config.delight.num_inference_steps,
                        )
                        delighted_views.append(result.images[0])
                    except Exception as e:
                        logger.warning(f"IC-Light failed for object {i} view {j}: {e}")
                        delighted_views.append(_simple_delight(view))

                all_delighted.append(delighted_views)
                logger.info(f"Object {i}: {len(delighted_views)} views delighted")

        return all_delighted

    except Exception as e:
        logger.warning(f"IC-Light not available ({e}), using simple delight")
        return [[_simple_delight(v) for v in views] for views in all_views]


def _delight_intrinsic(
    all_views: list[list[Image.Image]],
) -> list[list[Image.Image]]:
    """Remove lighting using classical intrinsic image decomposition.

    This is a lightweight alternative that doesn't require ML models.
    Uses a simple approach: reduce contrast and normalize illumination.
    """
    all_delighted = []
    for views in all_views:
        delighted = [_simple_delight(v) for v in views]
        all_delighted.append(delighted)
    return all_delighted


def _simple_delight(image: Image.Image) -> Image.Image:
    """Simple delighting by reducing luminance contrast.

    Approximates albedo by:
    1. Converting to LAB color space
    2. Compressing the L channel toward its median
    3. Converting back to RGB
    """
    arr = np.array(image).astype(np.float32) / 255.0

    # Convert RGB to a simple luminance estimate
    luminance = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
    median_lum = np.median(luminance)

    # Compress luminance toward median (reduce shadows and highlights)
    compression = 0.5  # How much to compress (0 = no change, 1 = flat)
    target_lum = luminance * (1 - compression) + median_lum * compression

    # Scale RGB by luminance ratio
    ratio = np.where(
        luminance > 1e-4,
        target_lum / luminance,
        1.0,
    )
    result = arr * ratio[:, :, np.newaxis]
    result = np.clip(result, 0, 1)

    return Image.fromarray((result * 255).astype(np.uint8))
