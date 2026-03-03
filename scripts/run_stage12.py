"""Run only Stage 1+2 and export the mesh directly (skip Stage 3/4)."""

from __future__ import annotations

import logging
import shutil
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
load_dotenv(project_root / ".env")

from src.config import load_config
from src.pipeline import WorldGenPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)

prompt = sys.argv[1] if len(sys.argv) > 1 else "a medieval village on a hilltop with a castle"

config = load_config(project_root / "configs" / "default.yaml")
pipeline = WorldGenPipeline(config=config)

print(f"Prompt: {prompt}")
print("Running Stage 1 (scene planning)...")
stage1 = pipeline._run_stage1(prompt, "quick_test")

print("Running Stage 2 (3D reconstruction with texture)...")
stage2 = pipeline._run_stage2(stage1, "quick_test")

pipeline.vram.force_unload_all()

# Copy the mesh to output
mesh_path = stage2["scene_mesh_path"]
output_path = project_root / "data" / "output" / "scene_mesh.glb"
shutil.copy2(mesh_path, output_path)

print(f"\nDone! Output mesh: {output_path}")
print(f"Open this file in a 3D viewer (e.g. https://3dviewer.net/) to check the result.")
