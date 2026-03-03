"""Assemble blockout mesh from terrain and asset proxy boxes."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import trimesh

from src.utils.mesh_utils import create_box_proxy

logger = logging.getLogger(__name__)


def assemble_blockout(
    terrain_mesh: trimesh.Trimesh,
    placements: list[dict[str, Any]],
) -> trimesh.Trimesh:
    """Combine terrain mesh with proxy bounding boxes for all placed assets.

    Args:
        terrain_mesh: The terrain triangle mesh.
        placements: List of asset placement dicts (from asset_placement).

    Returns:
        Combined blockout mesh (terrain + all proxy boxes).
    """
    meshes = [terrain_mesh]

    for p in placements:
        position = np.array(p["position"])
        size = np.array(p["size"])
        rotation_y = float(p.get("rotation_y", 0.0))

        box = create_box_proxy(position, size, rotation_y)
        meshes.append(box)

    blockout = trimesh.util.concatenate(meshes)

    logger.info(
        f"Blockout assembled: {len(placements)} proxy boxes + terrain = "
        f"{len(blockout.vertices)} vertices, {len(blockout.faces)} faces"
    )

    return blockout
