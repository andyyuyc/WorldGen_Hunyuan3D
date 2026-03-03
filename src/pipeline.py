"""Main orchestrator: runs Stage I -> IV sequentially."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

from src.config import WorldGenConfig, load_config
from src.vram_manager import VRAMManager

logger = logging.getLogger(__name__)


class WorldGenPipeline:
    """End-to-end pipeline that converts a text prompt into a traversable 3D world.

    Stages:
        I.   Scene Planning     - Text -> blockout mesh + navmesh + reference image
        II.  Scene Reconstruction - Reference image -> monolithic 3D mesh
        III. Scene Decomposition - Monolithic mesh -> individual objects
        IV.  Scene Enhancement   - Per-object refinement + texturing
        Export                   - Package as glTF/GLB for game engine
    """

    def __init__(self, config: WorldGenConfig | None = None, config_path: str | None = None):
        if config is not None:
            self.config = config
        else:
            self.config = load_config(config_path)

        self.vram = VRAMManager(
            device=self.config.gpu.device,
            vram_limit_gb=self.config.gpu.vram_limit_gb,
        )

        # Resolve paths relative to project root
        self.project_root = Path(__file__).parent.parent
        self.intermediate_dir = self.project_root / self.config.project.intermediate_dir
        self.output_dir = self.project_root / self.config.project.output_dir

        # Ensure directories exist
        for stage_num in range(1, 5):
            (self.intermediate_dir / f"stage{stage_num}").mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(self, prompt: str, run_id: str | None = None) -> dict[str, Any]:
        """Run the full pipeline from text prompt to exported 3D scene.

        Args:
            prompt: Text description of the desired 3D world.
            run_id: Optional identifier for this run. Defaults to timestamp.

        Returns:
            Dictionary with paths to all output files.
        """
        if run_id is None:
            run_id = f"run_{int(time.time())}"

        logger.info(f"=== WorldGen Pipeline Start (run_id={run_id}) ===")
        logger.info(f"Prompt: {prompt}")
        logger.info(f"Config: {self.config.project.name}")
        logger.info(f"GPU: {self.vram}")

        results: dict[str, Any] = {"run_id": run_id, "prompt": prompt}
        total_start = time.time()

        try:
            # Stage I: Scene Planning
            stage1_result = self._run_stage1(prompt, run_id)
            results["stage1"] = stage1_result

            # Stage II: Scene Reconstruction
            stage2_result = self._run_stage2(stage1_result, run_id)
            results["stage2"] = stage2_result

            # Stage III: Scene Decomposition
            stage3_result = self._run_stage3(stage2_result, run_id)
            results["stage3"] = stage3_result

            # Stage IV: Scene Enhancement
            stage4_result = self._run_stage4(stage3_result, stage1_result, run_id)
            results["stage4"] = stage4_result

            # Export
            export_result = self._export(stage4_result, stage2_result, run_id)
            results["export"] = export_result

        finally:
            self.vram.force_unload_all()

        total_time = time.time() - total_start
        results["total_time_seconds"] = total_time
        logger.info(f"=== Pipeline Complete in {total_time:.1f}s ===")

        return results

    def _run_stage1(self, prompt: str, run_id: str) -> dict[str, Any]:
        """Stage I: Scene Planning."""
        logger.info("--- Stage I: Scene Planning ---")
        start = time.time()
        stage_dir = self.intermediate_dir / "stage1"

        from src.stage1_planning.llm_parser import parse_scene_prompt
        from src.stage1_planning.terrain import generate_terrain
        from src.stage1_planning.partitioning import partition_space
        from src.stage1_planning.asset_placement import place_assets
        from src.stage1_planning.blockout import assemble_blockout
        from src.stage1_planning.navmesh import extract_navmesh
        from src.stage1_planning.depth_renderer import render_depth
        from src.stage1_planning.reference_image import generate_reference_image

        # 1. LLM parses text -> JSON scene spec
        scene_spec = parse_scene_prompt(prompt, self.config.stage1)
        _save_json(stage_dir / "scene_spec.json", scene_spec)

        # 2. Generate terrain
        heightmap, terrain_mesh = generate_terrain(
            scene_spec["terrain"], self.config.stage1
        )
        terrain_mesh.export(str(stage_dir / "terrain.obj"))

        # 3. Partition space
        regions = partition_space(
            scene_spec["partitioning"],
            self.config.stage1,
            scene_extent=self.config.project.scene_extent_meters,
        )

        # 4. Place assets
        placements = place_assets(
            scene_spec["assets"], regions, heightmap, self.config.stage1
        )
        _save_json(stage_dir / "placements.json", placements)

        # 5. Assemble blockout
        blockout_mesh = assemble_blockout(terrain_mesh, placements)
        blockout_mesh.export(str(stage_dir / "blockout.obj"))

        # 6. Extract navmesh
        navmesh_mesh = extract_navmesh(blockout_mesh, self.config.stage1.navmesh)
        navmesh_mesh.export(str(stage_dir / "navmesh.obj"))

        # 7. Render depth map
        depth_map = render_depth(blockout_mesh, self.config.stage1)
        _save_image(stage_dir / "depth.png", depth_map)

        # 8. Generate reference image (requires GPU - ControlNet)
        reference_img = generate_reference_image(
            depth_map,
            scene_spec.get("style", ""),
            prompt,
            self.config.stage1,
            self.vram,
        )
        _save_image(stage_dir / "reference.png", reference_img)

        # Unload ControlNet after reference image generation
        self.vram.unload_current()

        elapsed = time.time() - start
        logger.info(f"Stage I complete in {elapsed:.1f}s")

        return {
            "scene_spec": scene_spec,
            "heightmap": heightmap,
            "terrain_mesh_path": str(stage_dir / "terrain.obj"),
            "blockout_mesh_path": str(stage_dir / "blockout.obj"),
            "navmesh_path": str(stage_dir / "navmesh.obj"),
            "depth_path": str(stage_dir / "depth.png"),
            "reference_path": str(stage_dir / "reference.png"),
            "placements": placements,
        }

    def _run_stage2(self, stage1: dict[str, Any], run_id: str) -> dict[str, Any]:
        """Stage II: Scene Reconstruction."""
        logger.info("--- Stage II: Scene Reconstruction ---")
        start = time.time()
        stage_dir = self.intermediate_dir / "stage2"

        from src.stage2_reconstruction.image_to_3d import image_to_mesh
        from src.stage2_reconstruction.navmesh_align import align_to_navmesh

        # 1. Image -> 3D mesh
        scene_mesh_path = image_to_mesh(
            stage1["reference_path"], self.config.stage2, self.vram, stage_dir
        )

        # Unload 3D generation model
        self.vram.unload_current()

        # 2. Align to navmesh
        aligned_path = align_to_navmesh(
            scene_mesh_path,
            stage1["navmesh_path"],
            stage1["blockout_mesh_path"],
            self.config.stage2,
            stage_dir,
        )

        elapsed = time.time() - start
        logger.info(f"Stage II complete in {elapsed:.1f}s")

        return {
            "scene_mesh_path": aligned_path,
            "original_mesh_path": scene_mesh_path,
        }

    def _run_stage3(self, stage2: dict[str, Any], run_id: str) -> dict[str, Any]:
        """Stage III: Scene Decomposition."""
        logger.info("--- Stage III: Scene Decomposition ---")
        start = time.time()
        stage_dir = self.intermediate_dir / "stage3"

        from src.stage3_decomposition.connectivity import analyze_connectivity
        from src.stage3_decomposition.ground_detection import detect_ground
        from src.stage3_decomposition.mesh_splitter import split_mesh
        from src.stage3_decomposition.part_merger import merge_small_parts

        import trimesh

        scene_mesh = trimesh.load(stage2["scene_mesh_path"], force="mesh")
        logger.info(f"Loaded mesh: {len(scene_mesh.vertices):,} verts, {len(scene_mesh.faces):,} faces")

        # Decimate if mesh is too large for efficient decomposition
        STAGE3_MAX_FACES = 200_000
        if len(scene_mesh.faces) > STAGE3_MAX_FACES:
            ratio = 1.0 - STAGE3_MAX_FACES / len(scene_mesh.faces)
            logger.info(f"Decimating mesh: {len(scene_mesh.faces):,} -> ~{STAGE3_MAX_FACES:,} faces")
            scene_mesh = scene_mesh.simplify_quadric_decimation(ratio)
            logger.info(f"Decimated: {len(scene_mesh.vertices):,} verts, {len(scene_mesh.faces):,} faces")

        # 1. Split into connected components
        components, degrees, connectivity = analyze_connectivity(
            scene_mesh, self.config.stage3
        )

        # 2. Detect ground
        ground_idx, ground_mesh = detect_ground(
            components, degrees, self.config.stage3
        )

        # 3. Further split large components
        objects = split_mesh(components, ground_idx, self.config.stage3)

        # If single-component mesh, extract ground from the original mesh
        # by removing the object faces (ground = full mesh minus objects)
        if len(components) == 1 and objects:
            import numpy as np
            logger.info("Re-extracting ground from single-component mesh")
            obj_verts = set()
            for obj in objects:
                obj_verts.update(range(len(obj.vertices)))
            # Use height + normal based ground extraction
            centroids = scene_mesh.triangles_center
            normals = scene_mesh.face_normals
            heights = centroids[:, 1]
            height_range = heights.max() - heights.min()
            height_cutoff = heights.min() + height_range * 0.3
            threshold = self.config.stage3.ground_normal_threshold
            ground_mask = (normals[:, 1] > threshold) & (heights < height_cutoff)
            if ground_mask.sum() > 0:
                ground_faces = scene_mesh.faces[ground_mask]
                unique_verts = np.unique(ground_faces.flatten())
                vert_map = {old: new for new, old in enumerate(unique_verts)}
                new_faces = np.vectorize(vert_map.get)(ground_faces)
                new_verts = scene_mesh.vertices[unique_verts]
                ground_mesh = trimesh.Trimesh(vertices=new_verts, faces=new_faces)
                if scene_mesh.visual.kind == "vertex" and scene_mesh.visual.vertex_colors is not None:
                    ground_mesh.visual.vertex_colors = scene_mesh.visual.vertex_colors[unique_verts]

        # 4. Merge small parts
        objects = merge_small_parts(objects, self.config.stage3)

        # Save results
        ground_mesh.export(str(stage_dir / "ground.obj"))
        manifest = []
        for i, obj_mesh in enumerate(objects):
            obj_path = stage_dir / f"object_{i:03d}.obj"
            obj_mesh.export(str(obj_path))
            manifest.append({
                "index": i,
                "path": str(obj_path),
                "face_count": len(obj_mesh.faces),
                "bounds": obj_mesh.bounds.tolist(),
            })
        _save_json(stage_dir / "manifest.json", {"ground": str(stage_dir / "ground.obj"), "objects": manifest})

        elapsed = time.time() - start
        logger.info(f"Stage III complete in {elapsed:.1f}s ({len(objects)} objects)")

        return {
            "ground_path": str(stage_dir / "ground.obj"),
            "object_paths": [m["path"] for m in manifest],
            "manifest": manifest,
        }

    def _run_stage4(
        self, stage3: dict[str, Any], stage1: dict[str, Any], run_id: str
    ) -> dict[str, Any]:
        """Stage IV: Scene Enhancement."""
        logger.info("--- Stage IV: Scene Enhancement ---")
        start = time.time()
        stage_dir = self.intermediate_dir / "stage4"

        from src.stage4_enhancement.vlm_describer import describe_objects
        from src.stage4_enhancement.image_enhancer import enhance_object_images
        from src.stage4_enhancement.multiview_texture import generate_textures
        from src.stage4_enhancement.delighter import delight_textures
        from src.stage4_enhancement.uv_unwrapper import unwrap_uvs
        from src.stage4_enhancement.texture_baking import bake_textures

        import trimesh

        object_meshes = [
            trimesh.load(p, force="mesh") for p in stage3["object_paths"]
        ]
        prompt = stage1["scene_spec"].get("style", "")

        # Phase A: VLM descriptions (batch all objects)
        descriptions = describe_objects(
            object_meshes, prompt, self.config.stage4, self.vram
        )
        self.vram.unload_current()

        # Phase B: Image enhancement (batch all objects)
        enhanced_images = enhance_object_images(
            object_meshes, descriptions, self.config.stage4, self.vram
        )
        self.vram.unload_current()

        # Phase D: Multi-view texture generation (batch all objects)
        multiview_images = generate_textures(
            object_meshes, enhanced_images, descriptions, self.config.stage4, self.vram
        )
        self.vram.unload_current()

        # Phase E: Delight textures (batch all objects)
        delighted = delight_textures(
            multiview_images, self.config.stage4, self.vram
        )
        self.vram.unload_current()

        # Phase F: UV unwrap + bake (CPU only)
        uv_meshes = unwrap_uvs(object_meshes, self.config.stage4)
        final_objects = bake_textures(
            uv_meshes, delighted, self.config.stage4, stage_dir
        )

        elapsed = time.time() - start
        logger.info(f"Stage IV complete in {elapsed:.1f}s")

        return {"final_objects": final_objects}

    def _export(
        self, stage4: dict[str, Any], stage2: dict[str, Any], run_id: str
    ) -> dict[str, Any]:
        """Export final scene as glTF/GLB."""
        logger.info("--- Export ---")

        from src.export.gltf_exporter import export_scene

        result = export_scene(
            stage4["final_objects"],
            self.output_dir,
            self.config.export,
        )

        logger.info(f"Scene exported to {self.output_dir}")
        return result


def _save_json(path: Path, data: Any) -> None:
    import json
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)


def _save_image(path: Path, image) -> None:
    from PIL import Image
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(image, np.ndarray):
        if image.dtype == np.float32 or image.dtype == np.float64:
            image = (image * 255).clip(0, 255).astype(np.uint8)
        Image.fromarray(image).save(str(path))
    elif isinstance(image, Image.Image):
        image.save(str(path))
    else:
        raise TypeError(f"Unsupported image type: {type(image)}")
