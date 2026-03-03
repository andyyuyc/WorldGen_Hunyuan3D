"""CLI entry point for running the WorldGen pipeline."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
load_dotenv(project_root / ".env")

from src.config import load_config
from src.pipeline import WorldGenPipeline


def main():
    parser = argparse.ArgumentParser(
        description="WorldGen: Text to Traversable 3D Worlds",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_pipeline.py "a medieval village on a hilltop with a castle"
  python run_pipeline.py "sci-fi base station on Mars" --config configs/default.yaml
  python run_pipeline.py "cartoon forest with mushroom houses" --stage1-only
        """,
    )
    parser.add_argument(
        "prompt",
        type=str,
        help="Text description of the desired 3D world",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(project_root / "configs" / "default.yaml"),
        help="Path to configuration YAML file",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Custom run identifier (default: timestamp)",
    )
    parser.add_argument(
        "--stage1-only",
        action="store_true",
        help="Run only Stage I (scene planning) for quick iteration",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # Load config
    config = load_config(args.config)

    # Create and run pipeline
    pipeline = WorldGenPipeline(config=config)

    if args.stage1_only:
        logging.info("Running Stage I only")
        result = pipeline._run_stage1(args.prompt, args.run_id or "stage1_test")
        logging.info(f"Stage I output: {result}")
    else:
        result = pipeline.run(args.prompt, run_id=args.run_id)
        logging.info(f"Pipeline complete: {result.get('export', {})}")


if __name__ == "__main__":
    main()
