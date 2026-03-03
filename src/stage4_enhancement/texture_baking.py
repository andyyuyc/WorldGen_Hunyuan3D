"""Texture baking: back-project multi-view images onto UV-unwrapped meshes."""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import trimesh
from PIL import Image

from src.config import Stage4Config

logger = logging.getLogger(__name__)


def bake_textures(
    uv_meshes: list[trimesh.Trimesh],
    delighted_views: list[list[Image.Image]],
    config: Stage4Config,
    output_dir: Path,
) -> list[dict[str, Any]]:
    """Bake multi-view texture images onto UV-unwrapped meshes (parallel)."""
    tex_resolution = config.texture.texture_resolution
    n = len(uv_meshes)
    logger.info(f"Baking textures for {n} objects (parallel)...")

    def _process_one(args):
        i, mesh, views = args
        t0 = time.time()
        obj_dir = output_dir / f"object_{i:03d}"
        obj_dir.mkdir(parents=True, exist_ok=True)

        texture = _bake_from_views(mesh, views, tex_resolution)
        texture = _inpaint_gaps(texture)

        tex_path = obj_dir / "texture_basecolor.png"
        Image.fromarray(texture).save(str(tex_path))

        mesh_with_tex = _apply_texture_to_mesh(mesh, texture)
        glb_path = obj_dir / "final.glb"
        mesh_with_tex.export(str(glb_path))

        elapsed = time.time() - t0
        logger.info(f"  Object {i}/{n}: baked in {elapsed:.1f}s ({len(mesh.faces):,} faces)")

        return {
            "index": i,
            "glb_path": str(glb_path),
            "texture_path": str(tex_path),
            "vertex_count": len(mesh.vertices),
            "face_count": len(mesh.faces),
        }

    tasks = [(i, m, v) for i, (m, v) in enumerate(zip(uv_meshes, delighted_views))]

    # Use threads - numpy/xatlas release the GIL
    import os
    n_workers = min(n, os.cpu_count() or 4)
    logger.info(f"  Using {n_workers} threads")

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        results = list(pool.map(_process_one, tasks))

    results.sort(key=lambda r: r["index"])
    return results


def _bake_from_views(
    mesh: trimesh.Trimesh,
    views: list[Image.Image],
    resolution: int,
) -> np.ndarray:
    """Back-project multi-view images onto UV space (fully vectorized)."""
    texture = np.zeros((resolution, resolution, 3), dtype=np.float64)
    weight_map = np.zeros((resolution, resolution), dtype=np.float64)

    if not hasattr(mesh.visual, "uv") or mesh.visual.uv is None:
        if views:
            return np.array(views[0].resize((resolution, resolution)))
        return np.full((resolution, resolution, 3), 128, dtype=np.uint8)

    uvs = mesh.visual.uv
    faces = mesh.faces
    face_normals = mesh.face_normals

    num_views = len(views)
    num_side = num_views - 2
    view_dirs = []

    view_dirs.append(np.array([0, 0, -1]))
    for k in range(num_side):
        angle = (k + 1) * (2 * np.pi / (num_side + 1))
        view_dirs.append(np.array([-np.sin(angle), 0, -np.cos(angle)]))
    view_dirs.append(np.array([0, -1, 0]))
    view_dirs = [d / np.linalg.norm(d) for d in view_dirs]

    view_arrays = [np.array(v.resize((512, 512))).astype(np.float64) for v in views]

    # Vectorized best-view computation
    neg_view_dirs = np.array([-vd for vd in view_dirs])
    dots = face_normals @ neg_view_dirs.T
    best_view_indices = dots.argmax(axis=1)
    best_dots = dots[np.arange(len(faces)), best_view_indices]
    visible = best_dots > 0.1

    # UV centroids for all faces
    face_uvs_all = uvs[faces]  # (num_faces, 3, 2)
    uv_centroids = face_uvs_all.mean(axis=1)  # (num_faces, 2)

    img_h, img_w = view_arrays[0].shape[:2]

    # Fully vectorized: use np.add.at to accumulate at UV centroid pixels
    for vi in range(num_views):
        mask = visible & (best_view_indices == vi)
        if not mask.any():
            continue

        fi = np.where(mask)[0]
        w = best_dots[fi]

        # UV centroid -> texture pixel coords
        cu = np.clip((uv_centroids[fi, 0] * (resolution - 1)).astype(int), 0, resolution - 1)
        cv = np.clip((uv_centroids[fi, 1] * (resolution - 1)).astype(int), 0, resolution - 1)

        # Sample colors from view image
        sx = np.clip((uv_centroids[fi, 0] * (img_w - 1)).astype(int), 0, img_w - 1)
        sy = np.clip(((1 - uv_centroids[fi, 1]) * (img_h - 1)).astype(int), 0, img_h - 1)
        colors = view_arrays[vi][sy, sx]  # (n, 3)

        # Accumulate weighted colors at UV pixels (no Python loop!)
        weighted_colors = colors * w[:, None]
        np.add.at(texture, (cv, cu), weighted_colors)
        np.add.at(weight_map, (cv, cu), w)

    # Normalize
    valid = weight_map > 0
    for c in range(3):
        texture[:, :, c][valid] /= weight_map[valid]

    return texture.clip(0, 255).astype(np.uint8)


def _inpaint_gaps(texture: np.ndarray) -> np.ndarray:
    """Fill unfilled (black) regions in the texture using OpenCV inpainting."""
    try:
        import cv2
        gray = cv2.cvtColor(texture, cv2.COLOR_RGB2GRAY)
        mask = (gray < 5).astype(np.uint8) * 255

        if mask.sum() == 0:
            return texture

        result = cv2.inpaint(texture, mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA)
        return result
    except ImportError:
        mask = texture.sum(axis=2) < 15
        if mask.any():
            avg_color = texture[~mask].mean(axis=0) if (~mask).any() else [128, 128, 128]
            texture[mask] = avg_color
        return texture


def _apply_texture_to_mesh(
    mesh: trimesh.Trimesh,
    texture: np.ndarray,
) -> trimesh.Trimesh:
    """Apply a baked texture atlas to a mesh."""
    material = trimesh.visual.material.PBRMaterial(
        baseColorTexture=Image.fromarray(texture),
        metallicFactor=0.0,
        roughnessFactor=0.7,
    )

    if hasattr(mesh.visual, "uv") and mesh.visual.uv is not None:
        mesh.visual = trimesh.visual.TextureVisuals(
            uv=mesh.visual.uv,
            material=material,
        )

    return mesh
