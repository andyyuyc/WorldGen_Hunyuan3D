"""Mesh refinement: geometric cleanup and optional regeneration-based detail."""

from __future__ import annotations

import logging

import numpy as np
import trimesh

from src.config import Stage4Config

logger = logging.getLogger(__name__)


def refine_mesh(
    mesh: trimesh.Trimesh,
    config: Stage4Config,
) -> trimesh.Trimesh:
    """Refine a coarse object mesh with geometric cleanup.

    Steps:
        1. Remove degenerate faces
        2. Fill small holes
        3. Laplacian smoothing
        4. Optional subdivision for low-poly meshes

    Args:
        mesh: Coarse object mesh.
        config: Stage 4 configuration.

    Returns:
        Refined mesh.
    """
    original_faces = len(mesh.faces)

    # Step 1: Remove degenerate faces (zero-area triangles)
    mesh.remove_degenerate_faces()

    # Step 2: Remove duplicate faces
    mesh.remove_duplicate_faces()

    # Step 3: Fix normals
    mesh.fix_normals()

    # Step 4: Fill small holes
    try:
        mesh.fill_holes()
    except Exception:
        pass  # Some meshes can't have holes filled

    # Step 5: Laplacian smoothing (gentle, 2 iterations)
    try:
        trimesh.smoothing.filter_laplacian(mesh, iterations=2, lamb=0.5)
    except Exception:
        pass  # Smoothing can fail on non-manifold meshes

    # Step 6: Subdivide if very low poly
    if len(mesh.faces) < 200:
        try:
            mesh = mesh.subdivide()
            logger.info(
                f"Subdivided low-poly mesh: {original_faces} -> {len(mesh.faces)} faces"
            )
        except Exception:
            pass

    logger.info(
        f"Mesh refined: {original_faces} -> {len(mesh.faces)} faces"
    )

    return mesh
