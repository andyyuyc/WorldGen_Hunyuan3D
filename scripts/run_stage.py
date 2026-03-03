"""Run individual stages for debugging and testing."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import load_config
from src.pipeline import WorldGenPipeline


def main():
    parser = argparse.ArgumentParser(description="Run individual WorldGen stages")
    parser.add_argument("stage", type=int, choices=[1, 2, 3, 4], help="Stage number to run")
    parser.add_argument("--prompt", type=str, default="a medieval village with a castle on a hill")
    parser.add_argument("--config", type=str, default=str(project_root / "configs" / "default.yaml"))
    parser.add_argument("--verbose", "-v", action="store_true")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    config = load_config(args.config)
    pipeline = WorldGenPipeline(config=config)
    run_id = f"stage{args.stage}_test"

    if args.stage == 1:
        result = pipeline._run_stage1(args.prompt, run_id)
        print(f"\nStage 1 outputs saved to: {pipeline.intermediate_dir / 'stage1'}")
        for key, val in result.items():
            if isinstance(val, str) and Path(val).exists():
                print(f"  {key}: {val}")

    elif args.stage == 2:
        # Stage 2 needs Stage 1 output - check if it exists
        stage1_dir = pipeline.intermediate_dir / "stage1"
        if not (stage1_dir / "reference.png").exists():
            print("Stage 1 output not found. Running Stage 1 first...")
            stage1_result = pipeline._run_stage1(args.prompt, run_id)
        else:
            import json
            stage1_result = {
                "reference_path": str(stage1_dir / "reference.png"),
                "navmesh_path": str(stage1_dir / "navmesh.obj"),
                "blockout_mesh_path": str(stage1_dir / "blockout.obj"),
            }
        result = pipeline._run_stage2(stage1_result, run_id)
        print(f"\nStage 2 outputs saved to: {pipeline.intermediate_dir / 'stage2'}")

    elif args.stage == 3:
        stage2_dir = pipeline.intermediate_dir / "stage2"
        mesh_candidates = list(stage2_dir.glob("scene_mesh*.glb")) + list(stage2_dir.glob("scene_mesh*.obj"))
        if not mesh_candidates:
            print("Stage 2 output not found. Run stages 1-2 first.")
            sys.exit(1)
        stage2_result = {"scene_mesh_path": str(mesh_candidates[0])}
        result = pipeline._run_stage3(stage2_result, run_id)
        print(f"\nStage 3 outputs: {len(result['object_paths'])} objects")

    elif args.stage == 4:
        stage3_dir = pipeline.intermediate_dir / "stage3"
        if not (stage3_dir / "manifest.json").exists():
            print("Stage 3 output not found. Run stages 1-3 first.")
            sys.exit(1)
        import json
        with open(stage3_dir / "manifest.json") as f:
            manifest = json.load(f)
        stage3_result = {
            "ground_path": manifest["ground"],
            "object_paths": [o["path"] for o in manifest["objects"]],
            "manifest": manifest["objects"],
        }
        stage1_result = {"scene_spec": {"style": args.prompt}}
        result = pipeline._run_stage4(stage3_result, stage1_result, run_id)
        print(f"\nStage 4 outputs: {len(result['final_objects'])} enhanced objects")


if __name__ == "__main__":
    main()
