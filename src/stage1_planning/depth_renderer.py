"""Isometric depth map rendering from blockout mesh."""

from __future__ import annotations

import logging

import numpy as np
import trimesh

from src.config import Stage1Config
from src.utils.render_utils import render_depth_isometric

logger = logging.getLogger(__name__)


def render_depth(
    blockout_mesh: trimesh.Trimesh,
    config: Stage1Config,
) -> np.ndarray:
    """Render an isometric depth map of the blockout mesh.

    The depth map is used as input to ControlNet for reference image generation.

    Args:
        blockout_mesh: Combined terrain + proxy boxes mesh.
        config: Stage 1 configuration.

    Returns:
        Depth map as float32 numpy array (H, W), values in meters.
    """
    resolution = config.reference_image.resolution

    logger.info(f"Rendering depth map at {resolution}x{resolution}")

    depth = render_depth_isometric(
        blockout_mesh,
        resolution=resolution,
        elevation_angle=45.0,
        azimuth_angle=45.0,
    )

    # Add small random perturbation to proxy box depths to break rectilinear appearance
    # (as described in the WorldGen paper)
    rng = np.random.RandomState(config.noise_seed + 200)
    noise = rng.normal(0, 0.02, size=depth.shape).astype(np.float32)
    # Only add noise where depth is valid (not background)
    valid_mask = depth > 0
    depth[valid_mask] += noise[valid_mask]

    logger.info(
        f"Depth map rendered: range [{depth[valid_mask].min():.2f}, "
        f"{depth[valid_mask].max():.2f}]"
    )

    return depth
