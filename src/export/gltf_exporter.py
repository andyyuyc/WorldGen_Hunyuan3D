"""Export final scene as glTF/GLB for Unity/Unreal consumption."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import trimesh

from src.config import ExportConfig

logger = logging.getLogger(__name__)


def export_scene(
    final_objects: list[dict[str, Any]],
    output_dir: Path,
    config: ExportConfig,
) -> dict[str, Any]:
    """Export the complete scene as a glTF/GLB file with manifest.

    Args:
        final_objects: List of dicts from texture_baking, each with:
            - glb_path: path to individual object GLB
            - texture_path: path to texture PNG
            - vertex_count, face_count
        output_dir: Directory for final output.
        config: Export configuration.

    Returns:
        Dict with paths to exported files.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Combine all objects into one scene
    scene = trimesh.Scene()

    for obj in final_objects:
        glb_path = obj.get("glb_path")
        if glb_path and Path(glb_path).exists():
            try:
                mesh = trimesh.load(glb_path, force="mesh")
                scene.add_geometry(
                    mesh,
                    node_name=f"object_{obj['index']:03d}",
                )
            except Exception as e:
                logger.warning(f"Failed to load {glb_path}: {e}")

    # Export combined scene
    ext = "glb" if config.format == "glb" else "gltf"
    scene_path = output_dir / f"scene.{ext}"

    try:
        scene.export(str(scene_path))
        logger.info(f"Scene exported to {scene_path}")
    except Exception as e:
        logger.error(f"Scene export failed: {e}")
        # Fallback: copy individual GLBs
        scene_path = None

    # Generate manifest
    manifest = {
        "scene_file": scene_path.name if scene_path else None,
        "format": config.format,
        "objects": [
            {
                "index": obj["index"],
                "glb_path": obj["glb_path"],
                "texture_path": obj.get("texture_path"),
                "vertex_count": obj.get("vertex_count", 0),
                "face_count": obj.get("face_count", 0),
            }
            for obj in final_objects
        ],
        "total_vertices": sum(obj.get("vertex_count", 0) for obj in final_objects),
        "total_faces": sum(obj.get("face_count", 0) for obj in final_objects),
    }

    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    logger.info(f"Manifest written to {manifest_path}")

    return {
        "scene_path": str(scene_path) if scene_path else None,
        "manifest_path": str(manifest_path),
        "object_count": len(final_objects),
    }
