"""VLM-based per-object description generation.

Uses a vision-language model to describe each decomposed object,
producing text prompts for downstream texture generation.
"""

from __future__ import annotations

import base64
import io
import logging
from typing import Any

import numpy as np
import trimesh
from PIL import Image

from src.config import Stage4Config
from src.utils.render_utils import render_object_front
from src.vram_manager import VRAMManager

logger = logging.getLogger(__name__)


def describe_objects(
    object_meshes: list[trimesh.Trimesh],
    scene_prompt: str,
    config: Stage4Config,
    vram: VRAMManager,
) -> list[dict[str, Any]]:
    """Generate text descriptions for all objects using a VLM.

    Batch processes all objects with a single model load to minimize VRAM churn.

    Args:
        object_meshes: List of individual object meshes.
        scene_prompt: Original scene style/description for context.
        config: Stage 4 configuration.
        vram: VRAM manager.

    Returns:
        List of description dicts, one per object, containing:
        - name: Inferred object name
        - material: Material description
        - colors: Color palette description
        - texture_prompt: Detailed prompt for texture generation
    """
    logger.info(f"Describing {len(object_meshes)} objects with VLM")

    if config.vlm_provider == "anthropic":
        return _describe_with_anthropic(object_meshes, scene_prompt, config)
    else:
        return _describe_with_heuristic(object_meshes, scene_prompt)


def _describe_with_anthropic(
    meshes: list[trimesh.Trimesh],
    scene_prompt: str,
    config: Stage4Config,
) -> list[dict[str, Any]]:
    """Use Claude Vision API to describe objects."""
    import anthropic
    import json

    client = anthropic.Anthropic(timeout=30.0)
    descriptions = []

    for i, mesh in enumerate(meshes):
        # Render object from front view
        color, _depth = render_object_front(mesh, resolution=512)
        image = Image.fromarray(color)

        # Encode image to base64
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        prompt = (
            f'This 3D object is part of a scene described as: "{scene_prompt}". '
            f"Object #{i + 1} has {len(mesh.faces)} faces and approximate size "
            f"{mesh.bounding_box.extents[0]:.1f}x"
            f"{mesh.bounding_box.extents[1]:.1f}x"
            f"{mesh.bounding_box.extents[2]:.1f} meters.\n\n"
            "Analyze this rendered 3D object and respond with ONLY valid JSON:\n"
            "{\n"
            '  "name": "short object name (e.g. wooden house, pine tree)",\n'
            '  "material": "primary material (e.g. rough stone, polished wood)",\n'
            '  "colors": "dominant color palette (e.g. brown, dark green, grey)",\n'
            '  "texture_prompt": "detailed texture description for generation"\n'
            "}"
        )

        try:
            response = client.messages.create(
                model=config.vlm_model,
                max_tokens=500,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": img_b64,
                                },
                            },
                            {"type": "text", "text": prompt},
                        ],
                    }
                ],
            )

            text = response.content[0].text.strip()
            # Strip markdown fences
            if text.startswith("```"):
                lines = text.split("\n")
                lines = [l for l in lines if not l.strip().startswith("```")]
                text = "\n".join(lines)

            desc = json.loads(text)
            desc.setdefault("name", f"object_{i}")
            desc.setdefault("material", "unknown")
            desc.setdefault("colors", "grey")
            desc.setdefault("texture_prompt", f"{scene_prompt} object")
            descriptions.append(desc)

            logger.info(f"Object {i}: {desc['name']} ({desc['material']})")

        except Exception as e:
            logger.warning(f"VLM description failed for object {i}: {e}")
            descriptions.append(_fallback_description(i, scene_prompt, mesh))

    return descriptions


def _describe_with_heuristic(
    meshes: list[trimesh.Trimesh],
    scene_prompt: str,
) -> list[dict[str, Any]]:
    """Simple heuristic-based description without VLM."""
    descriptions = []

    for i, mesh in enumerate(meshes):
        extents = mesh.bounding_box.extents
        height = extents[1]
        width = max(extents[0], extents[2])

        # Rough classification by shape
        if height > width * 2:
            name = "tall structure"
            material = "stone or wood"
        elif width > height * 3:
            name = "flat structure"
            material = "stone"
        elif height > 3:
            name = "building"
            material = "stone and wood"
        else:
            name = "prop"
            material = "mixed materials"

        descriptions.append({
            "name": f"{name}_{i}",
            "material": material,
            "colors": "natural tones",
            "texture_prompt": f"{scene_prompt}, {name}, {material}, detailed texture",
        })

    return descriptions


def _fallback_description(
    index: int, scene_prompt: str, mesh: trimesh.Trimesh
) -> dict[str, Any]:
    """Generate a basic fallback description."""
    return {
        "name": f"object_{index}",
        "material": "unknown material",
        "colors": "neutral tones",
        "texture_prompt": f"{scene_prompt}, 3D object, detailed surface texture",
    }
