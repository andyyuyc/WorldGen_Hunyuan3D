"""Ground plane detection from mesh components."""

from __future__ import annotations

import logging

import numpy as np
import trimesh

from src.config import Stage3Config

logger = logging.getLogger(__name__)


def detect_ground(
    components: list[trimesh.Trimesh],
    degrees: np.ndarray,
    config: Stage3Config,
) -> tuple[int, trimesh.Trimesh]:
    """Identify which component is the ground/terrain.

    Heuristics (in priority order):
    1. Highest connectivity degree (touches the most other components)
    2. Largest face area
    3. Most face normals pointing upward

    Args:
        components: List of mesh components.
        degrees: Connectivity degrees for each component.
        config: Stage 3 configuration.

    Returns:
        Tuple of (ground component index, ground mesh).
    """
    if len(components) == 0:
        raise ValueError("No components to analyze")

    if len(components) == 1:
        logger.info("Only one component, treating as ground")
        return 0, components[0]

    n = len(components)
    scores = np.zeros(n, dtype=np.float64)

    # Score 1: Connectivity degree (normalized)
    max_degree = degrees.max()
    if max_degree > 0:
        scores += (degrees / max_degree) * 3.0  # Weight: 3x

    # Score 2: Face area (normalized)
    areas = np.array([c.area for c in components])
    max_area = areas.max()
    if max_area > 0:
        scores += (areas / max_area) * 2.0  # Weight: 2x

    # Score 3: Upward-facing normals
    threshold = config.ground_normal_threshold
    for i, comp in enumerate(components):
        if len(comp.face_normals) == 0:
            continue
        up_ratio = np.mean(comp.face_normals[:, 1] > threshold)
        scores[i] += up_ratio * 2.0  # Weight: 2x

    # Score 4: XZ extent (ground tends to be the widest horizontally)
    for i, comp in enumerate(components):
        extents = comp.bounding_box.extents
        xz_extent = extents[0] * extents[2]
        y_extent = extents[1]
        if y_extent > 0:
            flatness = xz_extent / (y_extent + 1e-6)
            scores[i] += min(flatness / 100.0, 1.0)  # Weight: 1x, capped

    ground_idx = int(np.argmax(scores))
    ground_mesh = components[ground_idx]

    logger.info(
        f"Ground detected: component {ground_idx} "
        f"(score={scores[ground_idx]:.2f}, "
        f"degree={degrees[ground_idx]}, "
        f"area={areas[ground_idx]:.1f}m², "
        f"{len(ground_mesh.faces)} faces)"
    )

    return ground_idx, ground_mesh
