"""Scene manifest generation for game engine consumption."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def generate_manifest(
    objects: list[dict[str, Any]],
    ground_path: str | None = None,
    navmesh_path: str | None = None,
    scene_file: str | None = None,
) -> dict[str, Any]:
    """Generate a scene manifest JSON for Unity/Unreal import.

    The manifest describes the scene structure so the game engine
    can properly load and configure all elements.

    Args:
        objects: List of object metadata dicts.
        ground_path: Path to the ground mesh file.
        navmesh_path: Path to the navigation mesh file.
        scene_file: Path to the combined scene file.

    Returns:
        Manifest dictionary ready for JSON serialization.
    """
    return {
        "version": "1.0",
        "generator": "worldgen",
        "scene_file": scene_file,
        "navmesh_file": navmesh_path,
        "ground_file": ground_path,
        "objects": [
            {
                "name": f"object_{obj.get('index', i):03d}",
                "file": obj.get("glb_path"),
                "texture": obj.get("texture_path"),
                "vertex_count": obj.get("vertex_count", 0),
                "face_count": obj.get("face_count", 0),
                "interactable": obj.get("interactable", False),
            }
            for i, obj in enumerate(objects)
        ],
    }


def save_manifest(manifest: dict[str, Any], path: str | Path) -> None:
    """Save manifest to a JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
