"""Three-tier hierarchical asset placement with Poisson disk sampling."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

from src.config import Stage1Config
from src.stage1_planning.partitioning import Region
from src.utils.mesh_utils import sample_height_at_xz

logger = logging.getLogger(__name__)

# Approximate bounding box sizes for asset types (width, height, depth) in meters
_ASSET_SIZES: dict[str, dict[str, np.ndarray]] = {
    "small": {
        "default": np.array([0.5, 0.5, 0.5]),
        "rock": np.array([0.8, 0.5, 0.6]),
        "bush": np.array([1.0, 0.8, 1.0]),
        "barrel": np.array([0.5, 0.8, 0.5]),
        "crate": np.array([0.6, 0.6, 0.6]),
        "flower": np.array([0.3, 0.4, 0.3]),
        "mushroom": np.array([0.3, 0.3, 0.3]),
        "log": np.array([2.0, 0.4, 0.4]),
        "lantern": np.array([0.3, 0.5, 0.3]),
    },
    "medium": {
        "default": np.array([2.0, 3.0, 2.0]),
        "house": np.array([6.0, 5.0, 5.0]),
        "tree": np.array([3.0, 6.0, 3.0]),
        "rock_formation": np.array([4.0, 3.0, 3.0]),
        "tent": np.array([3.0, 2.5, 3.0]),
        "well": np.array([1.5, 2.0, 1.5]),
        "cart": np.array([2.5, 1.5, 1.5]),
        "statue": np.array([1.0, 3.0, 1.0]),
        "fence": np.array([4.0, 1.2, 0.3]),
        "bridge": np.array([6.0, 2.0, 3.0]),
    },
    "large": {
        "default": np.array([8.0, 10.0, 8.0]),
        "castle": np.array([15.0, 15.0, 15.0]),
        "temple": np.array([12.0, 10.0, 10.0]),
        "tower": np.array([5.0, 15.0, 5.0]),
        "large_tree": np.array([6.0, 12.0, 6.0]),
        "ruins": np.array([10.0, 6.0, 10.0]),
        "ship": np.array([12.0, 8.0, 5.0]),
        "windmill": np.array([5.0, 12.0, 5.0]),
        "fortress": np.array([18.0, 12.0, 18.0]),
    },
    "huge": {
        "default": np.array([20.0, 20.0, 20.0]),
    },
}


@dataclass
class Placement:
    """A placed asset in the scene."""

    asset_type: str
    tier: str  # hero, medium, small
    position: np.ndarray  # (x, y, z) world coordinates
    size: np.ndarray  # (w, h, d) bounding box
    rotation_y: float  # degrees
    scale_factor: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "asset_type": self.asset_type,
            "tier": self.tier,
            "position": self.position.tolist(),
            "size": self.size.tolist(),
            "rotation_y": self.rotation_y,
            "scale_factor": self.scale_factor,
        }


def place_assets(
    assets_spec: dict[str, Any],
    regions: list[Region],
    heightmap: np.ndarray,
    config: Stage1Config,
    scene_extent: float = 50.0,
) -> list[dict[str, Any]]:
    """Place assets in three tiers: hero, medium, small.

    Args:
        assets_spec: Asset specification from scene spec.
        regions: Spatial regions from partitioning.
        heightmap: Terrain heightmap for Y positioning.
        config: Stage 1 configuration.
        scene_extent: Scene size in meters.

    Returns:
        List of placement dictionaries.
    """
    rng = np.random.RandomState(config.noise_seed + 100)
    half = scene_extent / 2
    placements: list[Placement] = []
    exclusion_zones: list[tuple[np.ndarray, float]] = []  # (center_xz, radius)

    # Pass 1: Hero assets
    for hero in assets_spec.get("hero", []):
        p = _place_hero(hero, regions, heightmap, scene_extent, rng)
        if p is not None:
            placements.append(p)
            r = max(p.size[0], p.size[2]) * 0.7
            exclusion_zones.append((p.position[[0, 2]], r))

    # Pass 2: Medium assets
    for medium in assets_spec.get("medium", []):
        count = int(medium.get("count", 10))
        for _ in range(count):
            p = _place_distributed(
                medium, "medium", regions, heightmap, scene_extent,
                rng, exclusion_zones, config.poisson_disk_min_distance,
            )
            if p is not None:
                placements.append(p)
                r = max(p.size[0], p.size[2]) * 0.5
                exclusion_zones.append((p.position[[0, 2]], r))

    # Pass 3: Small assets
    for small in assets_spec.get("small", []):
        count = int(small.get("count", 20))
        for _ in range(count):
            p = _place_distributed(
                small, "small", regions, heightmap, scene_extent,
                rng, exclusion_zones, config.poisson_disk_min_distance * 0.5,
            )
            if p is not None:
                placements.append(p)

    logger.info(
        f"Placed {len(placements)} assets: "
        f"{sum(1 for p in placements if p.tier == 'hero')} hero, "
        f"{sum(1 for p in placements if p.tier == 'medium')} medium, "
        f"{sum(1 for p in placements if p.tier == 'small')} small"
    )

    return [p.to_dict() for p in placements]


def _place_hero(
    spec: dict[str, Any],
    regions: list[Region],
    heightmap: np.ndarray,
    scene_extent: float,
    rng: np.random.RandomState,
) -> Placement | None:
    """Place a hero asset at the specified position hint."""
    asset_type = spec.get("type", "structure")
    scale_name = spec.get("scale", "large")
    position_hint = spec.get("position_hint", "center")

    size = _get_asset_size(asset_type, scale_name)
    half = scene_extent / 2

    # Resolve position hint to XZ coordinates
    hint_map = {
        "center": np.array([0.0, 0.0]),
        "hilltop_center": None,  # Find highest point
        "edge_north": np.array([0.0, -half * 0.7]),
        "edge_south": np.array([0.0, half * 0.7]),
        "edge_east": np.array([half * 0.7, 0.0]),
        "edge_west": np.array([-half * 0.7, 0.0]),
        "random": rng.uniform(-half * 0.6, half * 0.6, size=2),
    }

    xz = hint_map.get(position_hint)
    if xz is None:
        # Find hilltop: highest point in heightmap
        idx = np.unravel_index(heightmap.argmax(), heightmap.shape)
        rows, cols = heightmap.shape
        xz = np.array([
            (idx[1] / cols - 0.5) * scene_extent,
            (idx[0] / rows - 0.5) * scene_extent,
        ])

    y = sample_height_at_xz(heightmap, xz[0], xz[1], scene_extent)
    position = np.array([xz[0], y, xz[1]])
    rotation = rng.uniform(0, 360)

    return Placement(
        asset_type=asset_type,
        tier="hero",
        position=position,
        size=size,
        rotation_y=rotation,
        scale_factor=1.0,
    )


def _place_distributed(
    spec: dict[str, Any],
    tier: str,
    regions: list[Region],
    heightmap: np.ndarray,
    scene_extent: float,
    rng: np.random.RandomState,
    exclusion_zones: list[tuple[np.ndarray, float]],
    min_distance: float,
) -> Placement | None:
    """Place an asset using distribution rules, avoiding exclusion zones."""
    asset_type = spec.get("type", "object")
    distribution = spec.get("distribution", "scattered")
    scale_name = "medium" if tier == "medium" else "small"
    size = _get_asset_size(asset_type, scale_name)
    half = scene_extent / 2

    # Try up to 30 times to find a valid position
    for _ in range(30):
        if distribution == "clustered" and regions:
            # Pick a random region and place near its centroid
            region = rng.choice(regions)
            offset = rng.normal(0, 3.0, size=2)
            xz = region.centroid + offset
        elif distribution == "ring":
            # Place in a ring around center
            angle = rng.uniform(0, 2 * np.pi)
            radius = rng.uniform(half * 0.3, half * 0.6)
            xz = np.array([radius * np.cos(angle), radius * np.sin(angle)])
        elif distribution == "near_hero" and exclusion_zones:
            # Place near the first hero (but outside exclusion)
            hero_xz = exclusion_zones[0][0]
            offset = rng.normal(0, 5.0, size=2)
            xz = hero_xz + offset
        else:
            # Scattered: random within bounds
            xz = rng.uniform(-half * 0.85, half * 0.85, size=2)

        # Clamp to bounds
        xz = np.clip(xz, -half * 0.9, half * 0.9)

        # Check exclusion zones
        if _check_exclusion(xz, exclusion_zones, min_distance):
            y = sample_height_at_xz(heightmap, xz[0], xz[1], scene_extent)
            position = np.array([xz[0], y, xz[1]])

            # Random rotation and slight scale variation
            rotation = rng.uniform(0, 360)
            scale = rng.uniform(0.8, 1.2)

            return Placement(
                asset_type=asset_type,
                tier=tier,
                position=position,
                size=size * scale,
                rotation_y=rotation,
                scale_factor=scale,
            )

    return None  # Failed to find valid position


def _check_exclusion(
    xz: np.ndarray,
    exclusion_zones: list[tuple[np.ndarray, float]],
    min_distance: float,
) -> bool:
    """Check if a position is far enough from all exclusion zones."""
    for center, radius in exclusion_zones:
        dist = np.linalg.norm(xz - center)
        if dist < radius + min_distance:
            return False
    return True


def _get_asset_size(asset_type: str, scale_name: str) -> np.ndarray:
    """Look up the approximate bounding box size for an asset type."""
    sizes = _ASSET_SIZES.get(scale_name, _ASSET_SIZES["medium"])
    if asset_type in sizes:
        return sizes[asset_type].copy()
    return sizes["default"].copy()
