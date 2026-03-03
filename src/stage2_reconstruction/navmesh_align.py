"""Navmesh-aware alignment of generated 3D scene mesh.

Since we lack Meta's learned navmesh conditioning, this module performs
geometric post-processing to align the generated mesh with the original
blockout's scale, ground plane, and walkable regions.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import trimesh

from src.config import Stage2Config

logger = logging.getLogger(__name__)


def align_to_navmesh(
    scene_mesh_path: str,
    navmesh_path: str,
    blockout_path: str,
    config: Stage2Config,
    output_dir: Path,
) -> str:
    """Align the generated scene mesh to the original blockout geometry.

    Steps:
        1. Scale alignment: match bounding box extents to blockout
        2. Ground plane enforcement: align bottom of mesh to navmesh height
        3. Walkability verification via raycasting

    Args:
        scene_mesh_path: Path to the generated mesh (from image_to_3d).
        navmesh_path: Path to the navmesh from Stage I.
        blockout_path: Path to the blockout mesh from Stage I.
        config: Stage 2 configuration.
        output_dir: Directory to save aligned mesh.

    Returns:
        Path to the aligned mesh file.
    """
    scene_mesh = trimesh.load(scene_mesh_path, force="mesh")
    navmesh = trimesh.load(navmesh_path, force="mesh")
    blockout = trimesh.load(blockout_path, force="mesh")

    output_path = output_dir / "scene_mesh_aligned.glb"

    logger.info(
        f"Aligning scene mesh ({len(scene_mesh.vertices)} verts) "
        f"to blockout ({len(blockout.vertices)} verts)"
    )

    # Step 1: Scale alignment
    scene_mesh = _align_scale(scene_mesh, blockout, config.alignment.scale_tolerance)

    # Step 2: Ground plane enforcement
    scene_mesh = _align_ground_plane(scene_mesh, navmesh)

    # Step 3: Center alignment
    scene_mesh = _align_center(scene_mesh, blockout)

    # Step 4: Walkability report
    _walkability_report(scene_mesh, navmesh, config, output_dir)

    # Save
    scene_mesh.export(str(output_path))
    logger.info(f"Aligned mesh saved to {output_path}")

    return str(output_path)


def _align_scale(
    scene_mesh: trimesh.Trimesh,
    blockout: trimesh.Trimesh,
    tolerance: float,
) -> trimesh.Trimesh:
    """Scale scene mesh to match blockout bounding box extents."""
    scene_extents = scene_mesh.bounding_box.extents
    blockout_extents = blockout.bounding_box.extents

    # Compute uniform scale factor (use the largest axis ratio)
    ratios = blockout_extents / np.maximum(scene_extents, 1e-6)
    scale_factor = np.median(ratios)  # Median is more robust than mean

    logger.info(
        f"Scale alignment: scene={scene_extents}, blockout={blockout_extents}, "
        f"scale_factor={scale_factor:.3f}"
    )

    scene_mesh.apply_scale(scale_factor)
    return scene_mesh


def _align_ground_plane(
    scene_mesh: trimesh.Trimesh,
    navmesh: trimesh.Trimesh,
) -> trimesh.Trimesh:
    """Align the scene mesh's ground plane to the navmesh."""
    # Use the minimum Y coordinate as ground reference
    scene_ground = scene_mesh.vertices[:, 1].min()
    navmesh_ground = navmesh.vertices[:, 1].min()

    offset_y = navmesh_ground - scene_ground
    scene_mesh.apply_translation([0, offset_y, 0])

    logger.info(f"Ground plane offset: {offset_y:.3f}m")
    return scene_mesh


def _align_center(
    scene_mesh: trimesh.Trimesh,
    blockout: trimesh.Trimesh,
) -> trimesh.Trimesh:
    """Align the horizontal center of scene mesh to blockout centroid."""
    scene_center = scene_mesh.centroid
    blockout_center = blockout.centroid

    # Only align X and Z (horizontal), Y is handled by ground alignment
    offset = blockout_center - scene_center
    offset[1] = 0  # Don't move vertically

    scene_mesh.apply_translation(offset)
    logger.info(f"Center offset: [{offset[0]:.2f}, 0, {offset[2]:.2f}]m")
    return scene_mesh


def _walkability_report(
    scene_mesh: trimesh.Trimesh,
    navmesh: trimesh.Trimesh,
    config: Stage2Config,
    output_dir: Path,
) -> None:
    """Generate a walkability report comparing scene mesh to navmesh."""
    import json

    # Sample points on navmesh
    num_samples = min(1000, len(navmesh.vertices))
    sample_indices = np.random.choice(len(navmesh.vertices), num_samples, replace=False)
    sample_points = navmesh.vertices[sample_indices]

    # Raycast downward from above each navmesh point onto scene mesh
    ray_origins = sample_points.copy()
    ray_origins[:, 1] += 50.0  # Start rays from above

    ray_directions = np.zeros_like(ray_origins)
    ray_directions[:, 1] = -1.0  # Cast downward

    # Use trimesh ray casting
    try:
        locations, index_ray, index_tri = scene_mesh.ray.intersects_location(
            ray_origins=ray_origins,
            ray_directions=ray_directions,
        )

        if len(locations) > 0:
            # For each hit, compute height deviation from navmesh point
            hit_heights = locations[:, 1]
            expected_heights = sample_points[index_ray, 1]
            deviations = np.abs(hit_heights - expected_heights)

            report = {
                "total_samples": num_samples,
                "hits": len(locations),
                "hit_ratio": len(locations) / num_samples,
                "mean_deviation_m": float(deviations.mean()),
                "max_deviation_m": float(deviations.max()),
                "rms_deviation_m": float(np.sqrt(np.mean(deviations**2))),
                "within_threshold": float(
                    np.mean(deviations < config.alignment.raycast_deviation_max)
                ),
            }
        else:
            report = {
                "total_samples": num_samples,
                "hits": 0,
                "hit_ratio": 0.0,
                "error": "No ray hits on scene mesh",
            }
    except Exception as e:
        report = {"error": str(e)}

    report_path = output_dir / "walkability_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    logger.info(f"Walkability report: {report}")
