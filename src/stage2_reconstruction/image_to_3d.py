"""Image-to-3D mesh generation using TRELLIS.2 or fallback models."""

from __future__ import annotations

import json
import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from PIL import Image

from src.config import Stage2Config
from src.vram_manager import VRAMManager

logger = logging.getLogger(__name__)


def image_to_mesh(
    reference_image_path: str,
    config: Stage2Config,
    vram: VRAMManager,
    output_dir: Path,
) -> str:
    """Convert a reference image to a 3D mesh.

    Args:
        reference_image_path: Path to the reference image (from Stage I).
        config: Stage 2 configuration.
        vram: VRAM manager.
        output_dir: Directory to save the output mesh.

    Returns:
        Path to the generated mesh file (GLB or OBJ).
    """
    backend = config.backend
    logger.info(f"Image-to-3D backend: {backend}")

    if backend == "trellis2":
        return _generate_trellis2(reference_image_path, config, vram, output_dir)
    elif backend == "triposr":
        return _generate_triposr(reference_image_path, config, vram, output_dir)
    elif backend == "hunyuan3d":
        return _generate_hunyuan3d(reference_image_path, config, vram, output_dir)
    elif backend == "instantmesh":
        return _generate_instantmesh(reference_image_path, config, vram, output_dir)
    else:
        raise ValueError(f"Unknown image-to-3D backend: {backend}")


def _generate_trellis2(
    image_path: str,
    config: Stage2Config,
    vram: VRAMManager,
    output_dir: Path,
) -> str:
    """Generate 3D mesh using TRELLIS.2 (Microsoft).

    TRELLIS.2 is Linux-only, so on Windows we invoke it via WSL2 subprocess.
    """
    output_path = output_dir / "scene_mesh.glb"
    trellis_config = config.trellis2

    if trellis_config.use_wsl2:
        return _run_trellis2_wsl2(image_path, trellis_config, output_path)
    else:
        return _run_trellis2_native(image_path, trellis_config, vram, output_path)


def _run_trellis2_wsl2(
    image_path: str,
    trellis_config: Any,
    output_path: Path,
) -> str:
    """Run TRELLIS.2 via WSL2 subprocess on Windows.

    This assumes TRELLIS.2 is installed in the WSL2 environment at
    ~/TRELLIS.2/ with a conda environment 'trellis2'.
    """
    # Convert Windows paths to WSL paths
    image_wsl = _win_to_wsl_path(image_path)
    output_wsl = _win_to_wsl_path(str(output_path))

    # Create a temporary Python script to run in WSL2
    script = f"""
import sys
sys.path.insert(0, '/root/TRELLIS.2')
from PIL import Image
from trellis.pipelines import TrellisImageTo3DPipeline

pipeline = TrellisImageTo3DPipeline.from_pretrained('microsoft/TRELLIS-image-large')
pipeline.cuda()

image = Image.open('{image_wsl}')
outputs = pipeline.run(image, seed=42)

# Export as GLB
mesh = outputs[0]
mesh.export('{output_wsl}')
print('SUCCESS')
"""

    # Write script to temp file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(script)
        script_path = f.name

    script_wsl = _win_to_wsl_path(script_path)

    logger.info("Running TRELLIS.2 via WSL2...")
    try:
        result = subprocess.run(
            ["wsl", "bash", "-c",
             f"source ~/miniconda3/etc/profile.d/conda.sh && "
             f"conda activate trellis2 && "
             f"python {script_wsl}"],
            capture_output=True,
            text=True,
            timeout=600,  # 10 minute timeout
        )

        if "SUCCESS" in result.stdout:
            logger.info("TRELLIS.2 generation complete")
            return str(output_path)
        else:
            logger.error(f"TRELLIS.2 failed: {result.stderr}")
            raise RuntimeError(f"TRELLIS.2 WSL2 execution failed: {result.stderr[:500]}")

    except subprocess.TimeoutExpired:
        raise RuntimeError("TRELLIS.2 timed out after 10 minutes")
    finally:
        Path(script_path).unlink(missing_ok=True)


def _run_trellis2_native(
    image_path: str,
    trellis_config: Any,
    vram: VRAMManager,
    output_path: Path,
) -> str:
    """Run TRELLIS.2 natively (Linux or WSL2 environment)."""

    def _load_trellis():
        from trellis.pipelines import TrellisImageTo3DPipeline
        pipeline = TrellisImageTo3DPipeline.from_pretrained("microsoft/TRELLIS-image-large")
        pipeline.cuda()
        return pipeline

    with vram.load_model("trellis2", _load_trellis) as pipeline:
        image = Image.open(image_path)
        outputs = pipeline.run(image, seed=42)
        mesh = outputs[0]
        mesh.export(str(output_path))

    logger.info(f"TRELLIS.2 mesh saved to {output_path}")
    return str(output_path)


def _generate_triposr(
    image_path: str,
    config: Stage2Config,
    vram: VRAMManager,
    output_dir: Path,
) -> str:
    """Generate 3D mesh using TripoSR (lightweight fallback)."""
    output_path = output_dir / "scene_mesh.obj"

    def _load_triposr():
        import sys
        import torch
        triposr_path = str(Path(__file__).parent.parent.parent / "third_party" / "TripoSR")
        if triposr_path not in sys.path:
            sys.path.insert(0, triposr_path)
        from tsr.system import TSR

        model = TSR.from_pretrained(
            "stabilityai/TripoSR",
            config_name="config.yaml",
            weight_name="model.ckpt",
        )
        model.to("cuda")
        return model

    with vram.load_model("triposr", _load_triposr) as model:
        image = Image.open(image_path).convert("RGB")

        with __import__("torch").no_grad():
            scene_codes = model([image], device="cuda")

        meshes = model.extract_mesh(scene_codes, has_vertex_color=True, resolution=config.triposr.resolution)

        if meshes:
            mesh = meshes[0]
            mesh.export(str(output_path))
            logger.info(f"TripoSR mesh saved to {output_path}")
        else:
            raise RuntimeError("TripoSR produced no mesh output")

    return str(output_path)


def _generate_hunyuan3d(
    image_path: str,
    config: Stage2Config,
    vram: VRAMManager,
    output_dir: Path,
) -> str:
    """Generate 3D mesh using Hunyuan3D-2 (Tencent) - shape + texture."""
    import sys
    from PIL import Image
    output_path = output_dir / "scene_mesh.glb"

    hunyuan_path = str(Path(__file__).parent.parent.parent / "third_party" / "Hunyuan3D-2")
    if hunyuan_path not in sys.path:
        sys.path.insert(0, hunyuan_path)

    # Step 1: Shape generation
    logger.info("Hunyuan3D: generating shape...")
    def _load_shapegen():
        from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
        return Hunyuan3DDiTFlowMatchingPipeline.from_pretrained("tencent/Hunyuan3D-2")

    h3d_cfg = config.hunyuan3d
    with vram.load_model("hunyuan3d_shape", _load_shapegen) as pipeline:
        mesh = pipeline(
            image=image_path,
            num_inference_steps=h3d_cfg.num_inference_steps,
            octree_resolution=h3d_cfg.octree_resolution,
            guidance_scale=h3d_cfg.guidance_scale,
            num_chunks=h3d_cfg.num_chunks,
        )[0]

    logger.info(f"Hunyuan3D shape done: {len(mesh.vertices):,} verts, {len(mesh.faces):,} faces")

    # Step 2: Texture generation (requires custom_rasterizer CUDA extension)
    try:
        logger.info("Hunyuan3D: generating texture...")
        def _load_texgen():
            from hy3dgen.texgen import Hunyuan3DPaintPipeline
            return Hunyuan3DPaintPipeline.from_pretrained(
                "tencent/Hunyuan3D-2",
                subfolder="hunyuan3d-paint-v2-0-turbo",
            )

        image = Image.open(image_path).convert("RGBA")

        with vram.load_model("hunyuan3d_tex", _load_texgen) as tex_pipeline:
            textured_mesh = tex_pipeline(mesh, image=image)

        textured_mesh.export(str(output_path))
        logger.info(f"Hunyuan3D textured mesh saved to {output_path}")
        return str(output_path)

    except (ImportError, ModuleNotFoundError) as e:
        logger.warning(
            f"Hunyuan3D texgen unavailable ({e}). "
            "To enable texture generation, install CUDA Toolkit 12.8 and "
            "compile custom_rasterizer: cd third_party/Hunyuan3D-2/hy3dgen/texgen/custom_rasterizer && python setup.py install"
        )
        logger.info("Exporting shape-only mesh (no texture)...")
        mesh.export(str(output_path))
        logger.info(f"Hunyuan3D shape mesh saved to {output_path}")
        return str(output_path)


def _generate_instantmesh(
    image_path: str,
    config: Stage2Config,
    vram: VRAMManager,
    output_dir: Path,
) -> str:
    """Generate 3D mesh using InstantMesh (medium quality fallback)."""
    output_path = output_dir / "scene_mesh.obj"

    # InstantMesh is typically invoked as a script
    logger.info("Running InstantMesh...")

    result = subprocess.run(
        [
            "python", "-m", "instantmesh.run",
            "--input", image_path,
            "--output", str(output_dir),
            "--device", "cuda",
        ],
        capture_output=True,
        text=True,
        timeout=300,
    )

    if result.returncode != 0:
        raise RuntimeError(f"InstantMesh failed: {result.stderr[:500]}")

    # Find the output mesh
    for ext in [".obj", ".glb", ".ply"]:
        candidates = list(output_dir.glob(f"*{ext}"))
        if candidates:
            output_path = candidates[0]
            break

    logger.info(f"InstantMesh mesh saved to {output_path}")
    return str(output_path)


def _win_to_wsl_path(win_path: str) -> str:
    """Convert a Windows path to a WSL2 path."""
    win_path = win_path.replace("\\", "/")
    if len(win_path) >= 2 and win_path[1] == ":":
        drive = win_path[0].lower()
        return f"/mnt/{drive}{win_path[2:]}"
    return win_path
