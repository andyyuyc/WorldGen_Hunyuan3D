"""LLM-based text prompt parser: converts natural language to JSON scene specification."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from src.config import Stage1Config

logger = logging.getLogger(__name__)

# Default prompt template path
_PROMPT_TEMPLATE_PATH = Path(__file__).parent.parent.parent / "configs" / "prompts" / "scene_planning.txt"

# Fallback scene spec if LLM fails
_FALLBACK_SPEC: dict[str, Any] = {
    "terrain": {
        "type": "hilly",
        "base_elevation": 0.0,
        "elevation_range": 10.0,
        "roughness": 0.5,
        "noise_octaves": 4,
        "water_level": None,
    },
    "partitioning": {
        "method": "voronoi",
        "density": 0.5,
        "regularity": 0.3,
    },
    "assets": {
        "hero": [{"type": "large_tree", "position_hint": "center", "scale": "large"}],
        "medium": [
            {"type": "tree", "count": 15, "distribution": "scattered"},
            {"type": "rock_formation", "count": 5, "distribution": "scattered"},
        ],
        "small": [
            {"type": "bush", "count": 30, "distribution": "scattered"},
            {"type": "rock", "count": 20, "distribution": "scattered"},
        ],
    },
    "style": "natural landscape",
}


def parse_scene_prompt(
    prompt: str,
    config: Stage1Config,
    max_retries: int = 3,
) -> dict[str, Any]:
    """Parse a text prompt into a structured JSON scene specification using an LLM.

    Args:
        prompt: User's text description of the desired 3D world.
        config: Stage 1 configuration.
        max_retries: Number of retry attempts on parse failure.

    Returns:
        Validated scene specification dictionary.
    """
    if config.llm_provider == "anthropic":
        return _parse_with_anthropic(prompt, config, max_retries)
    else:
        raise ValueError(f"Unsupported LLM provider: {config.llm_provider}")


def _parse_with_anthropic(
    prompt: str, config: Stage1Config, max_retries: int
) -> dict[str, Any]:
    """Use Anthropic's Claude API to parse the scene prompt."""
    import anthropic

    client = anthropic.Anthropic()

    # Load prompt template
    template = _load_prompt_template()
    system_prompt = template.replace("{user_prompt}", prompt)

    for attempt in range(max_retries):
        try:
            response = client.messages.create(
                model=config.llm_model,
                max_tokens=2000,
                messages=[{"role": "user", "content": system_prompt}],
            )

            text = response.content[0].text.strip()

            # Strip markdown code fences if present
            if text.startswith("```"):
                lines = text.split("\n")
                # Remove first and last lines (``` markers)
                lines = [l for l in lines if not l.strip().startswith("```")]
                text = "\n".join(lines)

            spec = json.loads(text)
            validated = _validate_spec(spec)
            logger.info(f"Scene spec parsed successfully (attempt {attempt + 1})")
            return validated

        except json.JSONDecodeError as e:
            logger.warning(f"JSON parse error (attempt {attempt + 1}): {e}")
            if attempt == max_retries - 1:
                logger.error("All retries failed, using fallback spec")
                return _FALLBACK_SPEC.copy()

        except Exception as e:
            logger.warning(f"LLM call error (attempt {attempt + 1}): {e}")
            if attempt == max_retries - 1:
                logger.error("All retries failed, using fallback spec")
                return _FALLBACK_SPEC.copy()

    return _FALLBACK_SPEC.copy()


def _load_prompt_template() -> str:
    """Load the scene planning prompt template."""
    if _PROMPT_TEMPLATE_PATH.exists():
        return _PROMPT_TEMPLATE_PATH.read_text(encoding="utf-8")

    # Inline fallback template
    return (
        "You are a 3D scene planning AI. Given the following description, "
        "produce a JSON specification with keys: terrain, partitioning, assets, style. "
        "Respond with ONLY valid JSON.\n\n"
        "Description: {user_prompt}"
    )


def _validate_spec(spec: dict[str, Any]) -> dict[str, Any]:
    """Validate and sanitize the scene specification, filling defaults for missing fields."""

    # Ensure top-level keys
    if "terrain" not in spec:
        spec["terrain"] = _FALLBACK_SPEC["terrain"].copy()
    if "partitioning" not in spec:
        spec["partitioning"] = _FALLBACK_SPEC["partitioning"].copy()
    if "assets" not in spec:
        spec["assets"] = _FALLBACK_SPEC["assets"].copy()
    if "style" not in spec:
        spec["style"] = "generic"

    # Validate terrain
    t = spec["terrain"]
    valid_types = {"flat", "hilly", "mountainous", "coastal", "canyon", "desert", "volcanic"}
    if t.get("type") not in valid_types:
        t["type"] = "hilly"
    t.setdefault("base_elevation", 0.0)
    t["elevation_range"] = max(0.1, min(100.0, float(t.get("elevation_range", 10.0))))
    t["roughness"] = max(0.0, min(1.0, float(t.get("roughness", 0.5))))
    t["noise_octaves"] = max(1, min(8, int(t.get("noise_octaves", 4))))

    # Validate partitioning
    p = spec["partitioning"]
    valid_methods = {"voronoi", "bsp", "grid"}
    if p.get("method") not in valid_methods:
        p["method"] = "voronoi"
    p["density"] = max(0.0, min(1.0, float(p.get("density", 0.5))))
    p["regularity"] = max(0.0, min(1.0, float(p.get("regularity", 0.5))))

    # Validate assets
    a = spec["assets"]
    a.setdefault("hero", [])
    a.setdefault("medium", [])
    a.setdefault("small", [])

    valid_scales = {"small", "medium", "large", "huge"}
    for hero in a["hero"]:
        hero.setdefault("type", "structure")
        hero.setdefault("position_hint", "center")
        if hero.get("scale") not in valid_scales:
            hero["scale"] = "large"

    for item in a["medium"] + a["small"]:
        item.setdefault("type", "object")
        item["count"] = max(1, min(100, int(item.get("count", 10))))
        item.setdefault("distribution", "scattered")

    return spec
