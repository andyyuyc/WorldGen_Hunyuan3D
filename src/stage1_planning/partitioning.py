"""Spatial partitioning for scene layout: Voronoi, BSP, and Grid methods."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.spatial import Voronoi

from src.config import Stage1Config

logger = logging.getLogger(__name__)


@dataclass
class Region:
    """A spatial region in the scene layout."""

    index: int
    polygon: np.ndarray  # Nx2 array of (x, z) vertices
    centroid: np.ndarray  # (x, z) center point
    area: float
    zone_type: str = "open"  # open, residential, forest, plaza, etc.
    metadata: dict[str, Any] = field(default_factory=dict)


def partition_space(
    partition_spec: dict[str, Any],
    config: Stage1Config,
    scene_extent: float = 50.0,
) -> list[Region]:
    """Partition the scene into spatial regions based on specification.

    Args:
        partition_spec: Partitioning parameters (method, density, regularity).
        config: Stage 1 configuration.
        scene_extent: Scene size in meters.

    Returns:
        List of Region objects covering the scene.
    """
    method = partition_spec.get("method", config.default_partition_method)
    density = float(partition_spec.get("density", 0.5))
    regularity = float(partition_spec.get("regularity", 0.5))

    half = scene_extent / 2
    bounds = (-half, -half, half, half)  # (min_x, min_z, max_x, max_z)

    # Number of regions scales with density
    num_regions = max(4, int(5 + density * 25))

    logger.info(
        f"Partitioning: method={method}, num_regions={num_regions}, "
        f"regularity={regularity}"
    )

    if method == "voronoi":
        regions = _voronoi_partition(bounds, num_regions, regularity, config)
    elif method == "bsp":
        regions = _bsp_partition(bounds, num_regions, regularity)
    elif method == "grid":
        regions = _grid_partition(bounds, num_regions, regularity)
    else:
        logger.warning(f"Unknown partition method '{method}', falling back to voronoi")
        regions = _voronoi_partition(bounds, num_regions, regularity, config)

    logger.info(f"Created {len(regions)} regions")
    return regions


def _voronoi_partition(
    bounds: tuple[float, float, float, float],
    num_regions: int,
    regularity: float,
    config: Stage1Config,
) -> list[Region]:
    """Voronoi-based partitioning with Lloyd relaxation for organic layouts."""
    min_x, min_z, max_x, max_z = bounds
    rng = np.random.RandomState(config.noise_seed)

    # Initial seed points
    points = rng.uniform(
        low=[min_x, min_z], high=[max_x, max_z], size=(num_regions, 2)
    )

    # Lloyd relaxation: more iterations = more regular
    relaxation_iters = int(regularity * config.voronoi_relaxation_iters * 2)
    for _ in range(relaxation_iters):
        vor = Voronoi(points)
        new_points = []
        for i, region_idx in enumerate(vor.point_region):
            region = vor.regions[region_idx]
            if -1 not in region and len(region) > 2:
                polygon = vor.vertices[region]
                # Clip to bounds
                polygon[:, 0] = np.clip(polygon[:, 0], min_x, max_x)
                polygon[:, 1] = np.clip(polygon[:, 1], min_z, max_z)
                centroid = polygon.mean(axis=0)
                new_points.append(centroid)
            else:
                new_points.append(points[i])
        points = np.array(new_points)

    # Final Voronoi
    vor = Voronoi(points)

    regions = []
    for i, region_idx in enumerate(vor.point_region):
        vertex_indices = vor.regions[region_idx]
        if -1 in vertex_indices or len(vertex_indices) < 3:
            continue

        polygon = vor.vertices[vertex_indices]

        # Clip polygon to bounds
        polygon[:, 0] = np.clip(polygon[:, 0], min_x, max_x)
        polygon[:, 1] = np.clip(polygon[:, 1], min_z, max_z)

        centroid = polygon.mean(axis=0)
        area = _polygon_area(polygon)

        if area > 0.1:  # Skip degenerate regions
            regions.append(
                Region(
                    index=len(regions),
                    polygon=polygon,
                    centroid=centroid,
                    area=area,
                )
            )

    return regions


def _bsp_partition(
    bounds: tuple[float, float, float, float],
    num_regions: int,
    regularity: float,
) -> list[Region]:
    """Binary Space Partitioning for structured/urban layouts."""
    rng = np.random.RandomState(42)

    # Start with the full bounds as one rectangle
    rects: list[tuple[float, float, float, float]] = [bounds]

    # Recursively split until we have enough regions
    while len(rects) < num_regions:
        # Pick the largest rectangle to split
        areas = [(r[2] - r[0]) * (r[3] - r[1]) for r in rects]
        idx = np.argmax(areas)
        rect = rects.pop(idx)

        min_x, min_z, max_x, max_z = rect
        width = max_x - min_x
        height = max_z - min_z

        if width < 3.0 and height < 3.0:
            rects.append(rect)  # Too small to split
            break

        # Split along longer axis
        if width >= height:
            # Split ratio varies with regularity (0.5 = perfect, random otherwise)
            ratio = regularity * 0.5 + (1 - regularity) * rng.uniform(0.3, 0.7)
            split = min_x + width * ratio
            rects.append((min_x, min_z, split, max_z))
            rects.append((split, min_z, max_x, max_z))
        else:
            ratio = regularity * 0.5 + (1 - regularity) * rng.uniform(0.3, 0.7)
            split = min_z + height * ratio
            rects.append((min_x, min_z, max_x, split))
            rects.append((min_x, split, max_x, max_z))

    regions = []
    for i, (x0, z0, x1, z1) in enumerate(rects):
        polygon = np.array([[x0, z0], [x1, z0], [x1, z1], [x0, z1]])
        centroid = np.array([(x0 + x1) / 2, (z0 + z1) / 2])
        area = (x1 - x0) * (z1 - z0)
        regions.append(
            Region(index=i, polygon=polygon, centroid=centroid, area=area)
        )

    return regions


def _grid_partition(
    bounds: tuple[float, float, float, float],
    num_regions: int,
    regularity: float,
) -> list[Region]:
    """Grid-based partitioning with optional jitter."""
    min_x, min_z, max_x, max_z = bounds
    rng = np.random.RandomState(42)

    # Determine grid dimensions
    grid_size = max(2, int(np.sqrt(num_regions)))
    cell_w = (max_x - min_x) / grid_size
    cell_h = (max_z - min_z) / grid_size

    # Jitter amount inversely proportional to regularity
    jitter = (1.0 - regularity) * min(cell_w, cell_h) * 0.2

    regions = []
    for i in range(grid_size):
        for j in range(grid_size):
            x0 = min_x + i * cell_w + rng.uniform(-jitter, jitter)
            z0 = min_z + j * cell_h + rng.uniform(-jitter, jitter)
            x1 = x0 + cell_w + rng.uniform(-jitter, jitter)
            z1 = z0 + cell_h + rng.uniform(-jitter, jitter)

            # Clamp to bounds
            x0 = max(min_x, x0)
            z0 = max(min_z, z0)
            x1 = min(max_x, x1)
            z1 = min(max_z, z1)

            polygon = np.array([[x0, z0], [x1, z0], [x1, z1], [x0, z1]])
            centroid = np.array([(x0 + x1) / 2, (z0 + z1) / 2])
            area = (x1 - x0) * (z1 - z0)

            if area > 0.1:
                regions.append(
                    Region(index=len(regions), polygon=polygon, centroid=centroid, area=area)
                )

    return regions


def _polygon_area(polygon: np.ndarray) -> float:
    """Compute area of a 2D polygon using the shoelace formula."""
    n = len(polygon)
    if n < 3:
        return 0.0
    x = polygon[:, 0]
    y = polygon[:, 1]
    return abs(0.5 * np.sum(x[:-1] * y[1:] - x[1:] * y[:-1]) + 0.5 * (x[-1] * y[0] - x[0] * y[-1]))
