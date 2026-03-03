"""Per-object image enhancement using SDXL img2img.

Takes coarse object renders and enhances them with style-consistent detail,
guided by VLM-generated descriptions.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import trimesh
from PIL import Image

from src.config import Stage4Config
from src.utils.image_utils import compute_silhouette_iou, numpy_to_pil
from src.utils.render_utils import render_object_front
from src.vram_manager import VRAMManager

logger = logging.getLogger(__name__)


def enhance_object_images(
    object_meshes: list[trimesh.Trimesh],
    descriptions: list[dict[str, Any]],
    config: Stage4Config,
    vram: VRAMManager,
) -> list[Image.Image]:
    """Enhance rendered images of all objects using SDXL img2img.

    Batch loads the model once, processes all objects, then unloads.

    Args:
        object_meshes: Individual object meshes.
        descriptions: VLM descriptions for each object.
        config: Stage 4 configuration.
        vram: VRAM manager.

    Returns:
        List of enhanced PIL images, one per object.
    """
    logger.info(f"Enhancing images for {len(object_meshes)} objects")

    # Render coarse views of all objects first (CPU operation)
    coarse_images = []
    for mesh in object_meshes:
        color, _depth = render_object_front(mesh, resolution=512)
        coarse_images.append(numpy_to_pil(color))

    # Load SDXL img2img pipeline
    def _load_sdxl_img2img():
        import torch
        from diffusers import StableDiffusionXLImg2ImgPipeline

        pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float16,
            variant="fp16",
        )
        pipe.to("cuda")
        pipe.enable_model_cpu_offload()
        return pipe

    enhanced_images = []

    with vram.load_model("sdxl_img2img", _load_sdxl_img2img) as pipe:
        for i, (coarse_img, desc) in enumerate(zip(coarse_images, descriptions)):
            prompt = desc.get("texture_prompt", "detailed 3D object texture")
            negative_prompt = (
                "blurry, low quality, distorted, text, watermark, "
                "deformed, ugly, noisy"
            )

            enhanced = _enhance_with_verification(
                pipe, coarse_img, prompt, negative_prompt, config
            )
            enhanced_images.append(enhanced)
            logger.info(f"Object {i} ({desc.get('name', '?')}): image enhanced")

    return enhanced_images


def _enhance_with_verification(
    pipe: Any,
    coarse_image: Image.Image,
    prompt: str,
    negative_prompt: str,
    config: Stage4Config,
) -> Image.Image:
    """Enhance an image with IoU-based silhouette verification.

    If the enhanced image's silhouette deviates too much from the original,
    retry with lower denoising strength.
    """
    enh_config = config.enhancement
    strength = enh_config.strength

    for attempt in range(enh_config.max_retries):
        result = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=coarse_image,
            strength=strength,
            num_inference_steps=30,
        )
        enhanced = result.images[0]

        # Verify silhouette IoU
        iou = compute_silhouette_iou(coarse_image, enhanced)

        if iou >= enh_config.iou_threshold:
            return enhanced

        logger.warning(
            f"IoU {iou:.3f} < {enh_config.iou_threshold} "
            f"(attempt {attempt + 1}), reducing strength"
        )
        strength *= 0.7  # Reduce strength to preserve more structure

    # Return last attempt even if IoU is below threshold
    return enhanced
