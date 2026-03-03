"""Multi-view consistent texture generation.

Generates textures from multiple viewpoints around each object, using either
MVPaint (preferred) or a manual ControlNet-based pipeline.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import trimesh
from PIL import Image

from src.config import Stage4Config
from src.utils.image_utils import depth_to_controlnet, numpy_to_pil
from src.utils.render_utils import render_multiview_depths
from src.vram_manager import VRAMManager

logger = logging.getLogger(__name__)


def generate_textures(
    object_meshes: list[trimesh.Trimesh],
    enhanced_images: list[Image.Image],
    descriptions: list[dict[str, Any]],
    config: Stage4Config,
    vram: VRAMManager,
) -> list[list[Image.Image]]:
    """Generate multi-view texture images for all objects.

    Args:
        object_meshes: Individual object meshes.
        enhanced_images: Enhanced front-view images from image_enhancer.
        descriptions: VLM descriptions for prompts.
        config: Stage 4 configuration.
        vram: VRAM manager.

    Returns:
        List of lists: for each object, a list of multi-view texture images.
        Order: [front, side_0, ..., side_N, top]
    """
    backend = config.texture.backend

    if backend == "mvpaint":
        return _generate_mvpaint(object_meshes, descriptions, config, vram)
    else:
        return _generate_controlnet(
            object_meshes, enhanced_images, descriptions, config, vram
        )


def _generate_mvpaint(
    meshes: list[trimesh.Trimesh],
    descriptions: list[dict[str, Any]],
    config: Stage4Config,
    vram: VRAMManager,
) -> list[list[Image.Image]]:
    """Generate textures using MVPaint (synchronized multi-view diffusion).

    MVPaint processes multiple views simultaneously for consistency.
    Falls back to ControlNet if MVPaint is not installed.
    """
    try:
        from mvpaint import MVPaintPipeline
    except ImportError:
        logger.warning("MVPaint not available, falling back to ControlNet pipeline")
        return _generate_controlnet_fallback(meshes, descriptions, config, vram)

    def _load_mvpaint():
        import torch
        pipe = MVPaintPipeline.from_pretrained("3DTopia/MVPaint")
        pipe.to("cuda")
        return pipe

    all_views = []

    with vram.load_model("mvpaint", _load_mvpaint) as pipe:
        for i, (mesh, desc) in enumerate(zip(meshes, descriptions)):
            prompt = desc.get("texture_prompt", "detailed texture")

            try:
                result = pipe(
                    mesh=mesh,
                    prompt=prompt,
                    num_views=config.texture.num_side_views + 2,
                )
                views = result.images
                all_views.append(views)
                logger.info(f"Object {i}: {len(views)} texture views generated (MVPaint)")
            except Exception as e:
                logger.warning(f"MVPaint failed for object {i}: {e}")
                # Generate placeholder views
                all_views.append(_placeholder_views(mesh, config))

    return all_views


def _generate_controlnet(
    meshes: list[trimesh.Trimesh],
    enhanced_images: list[Image.Image],
    descriptions: list[dict[str, Any]],
    config: Stage4Config,
    vram: VRAMManager,
) -> list[list[Image.Image]]:
    """Generate textures using ControlNet depth conditioning per view.

    Less consistent than MVPaint but more widely available.
    """

    def _load_controlnet():
        import torch
        from diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline

        controlnet = ControlNetModel.from_pretrained(
            "diffusers/controlnet-depth-sdxl-1.0",
            torch_dtype=torch.float16,
        )
        pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            controlnet=controlnet,
            torch_dtype=torch.float16,
        )
        pipe.to("cuda")
        pipe.enable_model_cpu_offload()
        return pipe

    all_views = []

    with vram.load_model("sdxl_controlnet_texture", _load_controlnet) as pipe:
        for i, (mesh, front_img, desc) in enumerate(
            zip(meshes, enhanced_images, descriptions)
        ):
            prompt = desc.get("texture_prompt", "detailed texture")
            negative_prompt = "blurry, low quality, distorted, watermark"

            # Render depth maps from all viewpoints
            num_side = config.texture.num_side_views
            depth_maps = render_multiview_depths(mesh, num_side_views=num_side)

            views = []
            for j, depth in enumerate(depth_maps):
                depth_img = depth_to_controlnet(depth)
                depth_rgb = Image.merge("RGB", [depth_img.resize((512, 512))] * 3)

                result = pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    image=depth_rgb,
                    num_inference_steps=25,
                    controlnet_conditioning_scale=0.7,
                    width=512,
                    height=512,
                )
                views.append(result.images[0])

            all_views.append(views)
            logger.info(f"Object {i}: {len(views)} texture views generated (ControlNet)")

    return all_views


def _generate_controlnet_fallback(
    meshes: list[trimesh.Trimesh],
    descriptions: list[dict[str, Any]],
    config: Stage4Config,
    vram: VRAMManager,
) -> list[list[Image.Image]]:
    """Fallback when MVPaint is not available - uses simple rendered views."""
    all_views = []
    for mesh in meshes:
        all_views.append(_placeholder_views(mesh, config))
    return all_views


def _placeholder_views(
    mesh: trimesh.Trimesh,
    config: Stage4Config,
) -> list[Image.Image]:
    """Generate placeholder texture views from direct mesh renders."""
    from src.utils.render_utils import render_object_front

    num_views = config.texture.num_side_views + 2
    views = []
    for _ in range(num_views):
        color, _ = render_object_front(mesh, resolution=512)
        views.append(numpy_to_pil(color))
    return views
