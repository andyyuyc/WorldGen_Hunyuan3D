"""Procedural terrain generation using multi-octave Perlin/Simplex noise."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import trimesh
from opensimplex import OpenSimplex

from src.config import Stage1Config
from src.utils.mesh_utils import heightmap_to_mesh

logger = logging.getLogger(__name__)


def generate_terrain(
    terrain_spec: dict[str, Any],
    config: Stage1Config,
    scene_extent: float | None = None,
) -> tuple[np.ndarray, trimesh.Trimesh]:
    """Generate terrain heightmap and mesh from terrain specification.

    Args:
        terrain_spec: Terrain parameters from scene spec (type, elevation_range, etc.)
        config: Stage 1 configuration.
        scene_extent: Scene size in meters. Defaults to 50.

    Returns:
        Tuple of (heightmap as 2D float32 array, terrain Trimesh).
    """
    if scene_extent is None:
        scene_extent = 50.0

    resolution = config.terrain_resolution
    seed = config.noise_seed

    terrain_type = terrain_spec.get("type", "hilly")
    base_elevation = float(terrain_spec.get("base_elevation", 0.0))
    elevation_range = float(terrain_spec.get("elevation_range", 10.0))
    roughness = float(terrain_spec.get("roughness", 0.5))
    octaves = int(terrain_spec.get("noise_octaves", 4))
    water_level = terrain_spec.get("water_level")

    logger.info(
        f"Generating terrain: type={terrain_type}, resolution={resolution}, "
        f"elevation_range={elevation_range}m, roughness={roughness}"
    )

    # Generate base heightmap using multi-octave simplex noise
    heightmap = _generate_noise_heightmap(resolution, seed, roughness, octaves)

    # Apply terrain-type-specific modifiers
    heightmap = _apply_terrain_modifiers(heightmap, terrain_type, resolution)

    # Scale to desired elevation range
    h_min, h_max = heightmap.min(), heightmap.max()
    if h_max - h_min > 1e-6:
        heightmap = (heightmap - h_min) / (h_max - h_min)
    heightmap = heightmap * elevation_range + base_elevation

    # Apply water level flattening
    if water_level is not None:
        water_level = float(water_level)
        heightmap = np.maximum(heightmap, water_level)

    # Convert to mesh
    mesh = heightmap_to_mesh(heightmap, extent=scene_extent, center=True)

    logger.info(
        f"Terrain generated: {resolution}x{resolution} grid, "
        f"height range [{heightmap.min():.1f}, {heightmap.max():.1f}]m, "
        f"{len(mesh.vertices)} vertices, {len(mesh.faces)} faces"
    )

    return heightmap, mesh


def _generate_noise_heightmap(
    resolution: int, seed: int, roughness: float, octaves: int
) -> np.ndarray:
    """Generate a heightmap using multi-octave simplex noise.

    Args:
        resolution: Grid resolution (cells per axis).
        seed: Random seed for noise generator.
        roughness: Controls base frequency (0=smooth, 1=rough).
        octaves: Number of noise octaves for fractal detail.

    Returns:
        2D float32 array with values roughly in [-1, 1].
    """
    gen = OpenSimplex(seed=seed)
    heightmap = np.zeros((resolution, resolution), dtype=np.float32)

    # Base frequency scales with roughness
    base_freq = 1.0 + roughness * 3.0  # Range: 1.0 to 4.0

    for octave in range(octaves):
        freq = base_freq * (2.0 ** octave) / resolution
        amplitude = 0.5 ** octave  # Each octave has half the amplitude

        for i in range(resolution):
            for j in range(resolution):
                heightmap[i, j] += amplitude * gen.noise2(i * freq, j * freq)

    return heightmap


def _apply_terrain_modifiers(
    heightmap: np.ndarray, terrain_type: str, resolution: int
) -> np.ndarray:
    """Apply terrain-type-specific height modifiers.

    Each terrain type has characteristic features that modify the base noise.
    """
    from scipy.ndimage import gaussian_filter

    if terrain_type == "flat":
        # Flatten significantly, keep only subtle variation
        heightmap = gaussian_filter(heightmap, sigma=resolution * 0.15)
        heightmap *= 0.3

    elif terrain_type == "hilly":
        # Moderate smoothing for rolling hills
        heightmap = gaussian_filter(heightmap, sigma=resolution * 0.05)

    elif terrain_type == "mountainous":
        # Amplify peaks, add sharp features
        heightmap = np.where(heightmap > 0, heightmap * 1.5, heightmap * 0.8)
        # Add ridges using absolute value of noise
        heightmap = np.abs(heightmap) * np.sign(heightmap + 0.1)

    elif terrain_type == "coastal":
        # Create a gradient from land to sea
        gradient = np.linspace(1.0, -0.5, resolution).reshape(-1, 1)
        gradient = np.broadcast_to(gradient, (resolution, resolution))
        heightmap = heightmap * 0.7 + gradient * 0.5

    elif terrain_type == "canyon":
        # Create valley structure
        center = resolution // 2
        y_coords = np.arange(resolution)
        valley = 1.0 - np.exp(-((y_coords - center) ** 2) / (resolution * 0.1) ** 2)
        valley = valley.reshape(1, -1)
        valley = np.broadcast_to(valley, (resolution, resolution))
        heightmap = heightmap * 0.5 + valley * 0.8

    elif terrain_type == "desert":
        # Smooth dunes
        heightmap = gaussian_filter(heightmap, sigma=resolution * 0.08)
        heightmap = np.abs(heightmap) * 0.6

    elif terrain_type == "volcanic":
        # Central peak with surrounding flatland
        center = resolution // 2
        y, x = np.mgrid[0:resolution, 0:resolution]
        dist = np.sqrt((x - center) ** 2 + (y - center) ** 2) / (resolution * 0.3)
        volcano = np.exp(-(dist ** 2)) * 2.0
        # Add crater
        crater = np.where(dist < 0.15, -0.5 * (1 - dist / 0.15), 0)
        heightmap = heightmap * 0.3 + volcano + crater

    return heightmap
