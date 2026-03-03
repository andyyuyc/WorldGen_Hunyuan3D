"""ControlNet depth-conditioned reference image generation."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from PIL import Image

from src.config import Stage1Config
from src.utils.image_utils import depth_to_controlnet
from src.vram_manager import VRAMManager

logger = logging.getLogger(__name__)


def generate_reference_image(
    depth_map: np.ndarray,
    style: str,
    scene_prompt: str,
    config: Stage1Config,
    vram: VRAMManager,
) -> Image.Image:
    """Generate a styled reference image from a depth map using ControlNet.

    Args:
        depth_map: Float32 depth map from blockout rendering.
        style: Visual style string (e.g., "medieval fantasy").
        scene_prompt: Original user prompt for the scene.
        config: Stage 1 configuration.
        vram: VRAM manager for model lifecycle.

    Returns:
        RGB PIL Image of the reference scene.
    """
    ref_config = config.reference_image

    # Build the generation prompt
    prompt = _build_prompt(style, scene_prompt)
    negative_prompt = (
        "blurry, low quality, distorted, text, watermark, logo, "
        "out of frame, deformed, ugly, duplicate"
    )

    # Convert depth to ControlNet format
    depth_image = depth_to_controlnet(depth_map)
    # Resize to target resolution
    depth_image = depth_image.resize(
        (ref_config.resolution, ref_config.resolution), Image.BILINEAR
    )
    # Convert to RGB (ControlNet expects 3-channel)
    depth_rgb = Image.merge("RGB", [depth_image] * 3)

    logger.info(f"Generating reference image: {ref_config.resolution}x{ref_config.resolution}")
    logger.info(f"Prompt: {prompt[:100]}...")

    def _load_pipeline():
        import torch
        from diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline

        controlnet = ControlNetModel.from_pretrained(
            ref_config.controlnet_model,
            torch_dtype=torch.float16,
        )
        pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
            ref_config.base_model,
            controlnet=controlnet,
            torch_dtype=torch.float16,
        )
        pipe.to("cuda")
        # Enable memory optimizations
        pipe.enable_model_cpu_offload()
        return pipe

    with vram.load_model("sdxl_controlnet_depth", _load_pipeline) as pipe:
        result = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=depth_rgb,
            num_inference_steps=ref_config.num_inference_steps,
            controlnet_conditioning_scale=ref_config.controlnet_conditioning_scale,
            width=ref_config.resolution,
            height=ref_config.resolution,
        )
        image = result.images[0]

    logger.info("Reference image generated successfully")
    return image


def _build_prompt(style: str, scene_prompt: str) -> str:
    """Construct the generation prompt from style and scene description."""
    parts = []

    # Style prefix
    if style:
        parts.append(f"{style} style")

    # Scene description
    parts.append(f"aerial isometric view of {scene_prompt}")

    # Quality suffixes
    parts.append(
        "highly detailed, game art, professional concept art, "
        "sharp focus, vibrant colors, 8k quality"
    )

    return ", ".join(parts)
