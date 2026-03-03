"""Re-run Stage 3+ using existing Stage 2 output (skip Stage 1 & 2)."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
load_dotenv(project_root / ".env")

from src.config import load_config
from src.pipeline import WorldGenPipeline

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)

config = load_config(project_root / "configs" / "default.yaml")
pipeline = WorldGenPipeline(config=config)

# Use existing Stage 1 & 2 results
stage1_dir = project_root / "data" / "intermediate" / "stage1"
stage2_dir = project_root / "data" / "intermediate" / "stage2"

stage1_result = {
    "scene_spec": json.loads((stage1_dir / "scene_spec.json").read_text()),
    "reference_image_path": str(stage1_dir / "reference.png"),
}

stage2_result = {
    "scene_mesh_path": str(stage2_dir / "scene_mesh_aligned.glb"),
    "original_mesh_path": str(stage2_dir / "scene_mesh.glb"),
}

print(f"Using Stage 2 mesh: {stage2_result['scene_mesh_path']}")

# Re-run Stage 3 -> 4 -> Export
stage3_result = pipeline._run_stage3(stage2_result, "rerun")
print(f"Stage 3: {len(stage3_result['manifest'])} objects found")

stage4_result = pipeline._run_stage4(stage3_result, stage1_result, "rerun")
export_result = pipeline._export(stage4_result, stage2_result, "rerun")

pipeline.vram.force_unload_all()
print(f"Done! Export: {export_result}")
