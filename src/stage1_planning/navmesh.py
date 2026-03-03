"""Navigation mesh extraction from blockout geometry.

Uses PyRecastDetour on Windows, with a fallback to a simplified heightmap-based
navmesh if the Recast bindings are unavailable.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import trimesh

from src.config import NavmeshConfig

logger = logging.getLogger(__name__)


def extract_navmesh(
    blockout_mesh: trimesh.Trimesh,
    config: NavmeshConfig,
) -> trimesh.Trimesh:
    """Extract a navigation mesh from the blockout geometry.

    Attempts to use PyRecastDetour first, falls back to a simplified
    heightmap-based approach if unavailable.

    Args:
        blockout_mesh: The combined terrain + proxy boxes mesh.
        config: Navmesh configuration parameters.

    Returns:
        Navigation mesh (triangle mesh of walkable surfaces).
    """
    try:
        return _extract_with_recast(blockout_mesh, config)
    except ImportError:
        logger.warning(
            "PyRecastDetour not available, using simplified navmesh extraction"
        )
        return _extract_simplified(blockout_mesh, config)


def _extract_with_recast(
    mesh: trimesh.Trimesh,
    config: NavmeshConfig,
) -> trimesh.Trimesh:
    """Extract navmesh using PyRecastDetour (Recast Navigation bindings)."""
    from pyrecastdetour import RecastNavigation

    nav = RecastNavigation()

    # Set build parameters
    nav.set_cell_size(config.cell_size)
    nav.set_cell_height(config.cell_height)
    nav.set_agent_height(config.agent_height)
    nav.set_agent_radius(config.agent_radius)
    nav.set_agent_max_climb(config.agent_max_climb)
    nav.set_agent_max_slope(config.agent_max_slope)

    # Build navmesh from vertices and faces
    vertices = mesh.vertices.astype(np.float32).flatten()
    faces = mesh.faces.astype(np.int32).flatten()
    nav.build(vertices, len(mesh.vertices), faces, len(mesh.faces))

    # Extract navmesh polygons as triangle mesh
    nav_verts, nav_faces = nav.get_navmesh()
    nav_verts = np.array(nav_verts, dtype=np.float64).reshape(-1, 3)
    nav_faces = np.array(nav_faces, dtype=np.int64).reshape(-1, 3)

    navmesh = trimesh.Trimesh(vertices=nav_verts, faces=nav_faces)
    logger.info(
        f"Recast navmesh: {len(navmesh.vertices)} vertices, {len(navmesh.faces)} faces"
    )
    return navmesh


def _extract_simplified(
    mesh: trimesh.Trimesh,
    config: NavmeshConfig,
) -> trimesh.Trimesh:
    """Simplified navmesh: extract faces whose normals are mostly upward
    and within agent slope tolerance.

    This is a basic fallback when Recast is not available.
    """
    normals = mesh.face_normals
    up = np.array([0.0, 1.0, 0.0])
    cos_max_slope = np.cos(np.radians(config.agent_max_slope))

    # Select walkable faces: normal dot up > cos(max_slope)
    dots = np.dot(normals, up)
    walkable_mask = dots > cos_max_slope

    if not walkable_mask.any():
        logger.warning("No walkable faces found, returning empty navmesh")
        return trimesh.Trimesh()

    walkable_faces = mesh.faces[walkable_mask]

    # Build new mesh from walkable faces only
    # Remap vertex indices
    unique_verts = np.unique(walkable_faces.flatten())
    vert_map = {old: new for new, old in enumerate(unique_verts)}
    new_faces = np.vectorize(vert_map.get)(walkable_faces)
    new_vertices = mesh.vertices[unique_verts]

    navmesh = trimesh.Trimesh(vertices=new_vertices, faces=new_faces)

    # Remove small disconnected patches (noise)
    components = navmesh.split(only_watertight=False)
    if components:
        # Keep only components with significant area
        total_area = sum(c.area for c in components)
        min_area = total_area * 0.01  # At least 1% of total
        significant = [c for c in components if c.area > min_area]
        if significant:
            navmesh = trimesh.util.concatenate(significant)

    logger.info(
        f"Simplified navmesh: {len(navmesh.vertices)} vertices, "
        f"{len(navmesh.faces)} faces "
        f"({walkable_mask.sum()}/{len(mesh.faces)} faces walkable)"
    )
    return navmesh
