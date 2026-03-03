"""UV unwrapping using xatlas (multi-threaded)."""

from __future__ import annotations

import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import trimesh

from src.config import Stage4Config

logger = logging.getLogger(__name__)


def unwrap_uvs(
    object_meshes: list[trimesh.Trimesh],
    config: Stage4Config,
) -> list[trimesh.Trimesh]:
    """UV unwrap all object meshes using xatlas (parallel)."""
    try:
        import xatlas
    except ImportError:
        logger.warning("xatlas not available, using trimesh UV unwrap")
        return _unwrap_trimesh(object_meshes)

    n = len(object_meshes)
    n_workers = min(n, os.cpu_count() or 4)
    logger.info(f"UV unwrapping {n} objects with {n_workers} threads...")
    t0 = time.time()

    def _unwrap_one(args):
        i, mesh = args
        try:
            # Decimate large meshes before UV unwrapping
            max_faces = 50000
            if len(mesh.faces) > max_faces:
                logger.info(
                    f"  Object {i}: decimating {len(mesh.faces):,} -> {max_faces:,} faces"
                )
                mesh = mesh.simplify_quadric_decimation(max_faces)

            vertices = mesh.vertices.astype(np.float32)
            faces = mesh.faces.astype(np.uint32)

            vmapping, indices, uvs = xatlas.parametrize(vertices, faces)

            new_vertices = vertices[vmapping]
            new_mesh = trimesh.Trimesh(vertices=new_vertices, faces=indices)
            new_mesh.visual = trimesh.visual.TextureVisuals(uv=uvs)

            logger.info(
                f"  Object {i}/{n}: UV done "
                f"({len(mesh.vertices):,} -> {len(new_vertices):,} verts)"
            )
            return i, new_mesh

        except Exception as e:
            logger.warning(f"  Object {i}/{n}: xatlas failed ({e}), keeping original")
            return i, mesh

    tasks = list(enumerate(object_meshes))

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        results = list(pool.map(_unwrap_one, tasks))

    results.sort(key=lambda r: r[0])
    uv_meshes = [r[1] for r in results]

    logger.info(f"UV unwrapping done in {time.time() - t0:.1f}s")
    return uv_meshes


def _unwrap_trimesh(meshes: list[trimesh.Trimesh]) -> list[trimesh.Trimesh]:
    """Fallback UV unwrapping using trimesh's built-in methods."""
    result = []
    for mesh in meshes:
        if not hasattr(mesh.visual, "uv") or mesh.visual.uv is None:
            bounds = mesh.bounds
            vertices = mesh.vertices
            x_range = bounds[1][0] - bounds[0][0]
            z_range = bounds[1][2] - bounds[0][2]

            if x_range < 1e-6:
                x_range = 1.0
            if z_range < 1e-6:
                z_range = 1.0

            u = (vertices[:, 0] - bounds[0][0]) / x_range
            v = (vertices[:, 2] - bounds[0][2]) / z_range
            uvs = np.column_stack([u, v]).astype(np.float64)

            mesh.visual = trimesh.visual.TextureVisuals(uv=uvs)

        result.append(mesh)
    return result
