"""Initial mesh texturing for Stage II output.

If the 3D generation model (e.g., TRELLIS.2) already produces textured meshes,
this module serves as a fallback for models that produce untextured geometry.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image

from src.vram_manager import VRAMManager

logger = logging.getLogger(__name__)


def apply_initial_texture(
    mesh_path: str,
    reference_image_path: str,
    vram: VRAMManager,
    output_dir: Path,
) -> str:
    """Apply initial texturing to an untextured mesh using reference image projection.

    This is a fallback for when the 3D generation backend doesn't produce
    textured output. Uses simple top-down projection from the reference image.

    Args:
        mesh_path: Path to the untextured mesh.
        reference_image_path: Path to the reference image from Stage I.
        vram: VRAM manager.
        output_dir: Output directory.

    Returns:
        Path to the textured mesh.
    """
    mesh = trimesh.load(mesh_path, force="mesh")

    # Check if mesh already has texture/vertex colors
    if mesh.visual.kind == "texture" or (
        hasattr(mesh.visual, "vertex_colors")
        and mesh.visual.vertex_colors is not None
        and len(mesh.visual.vertex_colors) > 0
    ):
        logger.info("Mesh already has texture, skipping initial texturing")
        return mesh_path

    logger.info("Applying top-down projection texture from reference image")

    ref_image = Image.open(reference_image_path).convert("RGB")
    ref_array = np.array(ref_image)

    # Simple top-down UV projection
    bounds = mesh.bounds
    vertices = mesh.vertices

    # Map XZ coordinates to UV
    x_min, z_min = bounds[0][0], bounds[0][2]
    x_range = bounds[1][0] - bounds[0][0]
    z_range = bounds[1][2] - bounds[0][2]

    if x_range < 1e-6 or z_range < 1e-6:
        logger.warning("Mesh has zero XZ extent, cannot apply projection texture")
        return mesh_path

    u = (vertices[:, 0] - x_min) / x_range
    v = (vertices[:, 2] - z_min) / z_range

    # Sample colors from reference image at UV coordinates
    h, w = ref_array.shape[:2]
    px = np.clip((u * (w - 1)).astype(int), 0, w - 1)
    py = np.clip((v * (h - 1)).astype(int), 0, h - 1)

    vertex_colors = ref_array[py, px]

    # Add alpha channel
    alpha = np.full((len(vertex_colors), 1), 255, dtype=np.uint8)
    vertex_colors = np.hstack([vertex_colors, alpha])

    mesh.visual.vertex_colors = vertex_colors

    output_path = output_dir / "scene_mesh_textured.glb"
    mesh.export(str(output_path))
    logger.info(f"Textured mesh saved to {output_path}")

    return str(output_path)
