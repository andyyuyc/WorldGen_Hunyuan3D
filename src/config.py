"""Configuration loading and validation for WorldGen pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class NavmeshConfig(BaseModel):
    cell_size: float = 0.3
    cell_height: float = 0.2
    agent_height: float = 2.0
    agent_radius: float = 0.6
    agent_max_climb: float = 0.9
    agent_max_slope: float = 45.0


class ReferenceImageConfig(BaseModel):
    resolution: int = 1024
    controlnet_model: str = "diffusers/controlnet-depth-sdxl-1.0"
    base_model: str = "stabilityai/stable-diffusion-xl-base-1.0"
    num_inference_steps: int = 30
    controlnet_conditioning_scale: float = 0.8


class Stage1Config(BaseModel):
    llm_provider: str = "anthropic"
    llm_model: str = "claude-sonnet-4-20250514"
    terrain_resolution: int = 256
    noise_seed: int = 42
    default_partition_method: str = "voronoi"
    voronoi_relaxation_iters: int = 5
    poisson_disk_min_distance: float = 2.0
    navmesh: NavmeshConfig = Field(default_factory=NavmeshConfig)
    reference_image: ReferenceImageConfig = Field(default_factory=ReferenceImageConfig)


class Trellis2Config(BaseModel):
    resolution: int = 512
    use_wsl2: bool = True


class Hunyuan3DConfig(BaseModel):
    num_inference_steps: int = 100
    octree_resolution: int = 512
    guidance_scale: float = 7.5
    num_chunks: int = 20000


class TripoSRConfig(BaseModel):
    resolution: int = 256
    half_precision: bool = True


class AlignmentConfig(BaseModel):
    scale_tolerance: float = 0.1
    ground_plane_threshold: float = 0.3
    raycast_deviation_max: float = 0.5


class Stage2Config(BaseModel):
    backend: str = "trellis2"
    trellis2: Trellis2Config = Field(default_factory=Trellis2Config)
    hunyuan3d: Hunyuan3DConfig = Field(default_factory=Hunyuan3DConfig)
    triposr: TripoSRConfig = Field(default_factory=TripoSRConfig)
    alignment: AlignmentConfig = Field(default_factory=AlignmentConfig)


class Stage3Config(BaseModel):
    vertex_weld_epsilon: float = 0.001
    proximity_threshold: float = 0.05
    min_face_count: int = 500
    target_part_range: list[int] = Field(default=[3, 15])
    ground_normal_threshold: float = 0.7
    dbscan_eps: float = 0.5
    dbscan_min_samples: int = 50


class EnhancementConfig(BaseModel):
    strength: float = 0.6
    iou_threshold: float = 0.85
    max_retries: int = 3


class TextureConfig(BaseModel):
    backend: str = "mvpaint"
    num_side_views: int = 8
    texture_resolution: int = 2048


class DelightConfig(BaseModel):
    backend: str = "ic_light"
    num_inference_steps: int = 20


class UVConfig(BaseModel):
    max_stretch: float = 0.3
    padding: int = 4


class Stage4Config(BaseModel):
    vlm_provider: str = "anthropic"
    vlm_model: str = "claude-sonnet-4-20250514"
    enhancement: EnhancementConfig = Field(default_factory=EnhancementConfig)
    texture: TextureConfig = Field(default_factory=TextureConfig)
    delight: DelightConfig = Field(default_factory=DelightConfig)
    uv: UVConfig = Field(default_factory=UVConfig)


class ExportConfig(BaseModel):
    format: str = "glb"
    embed_textures: bool = True
    compress_meshes: bool = False


class GPUConfig(BaseModel):
    device: str = "cuda"
    vram_limit_gb: float = 24.0
    dtype: str = "float16"


class ProjectConfig(BaseModel):
    name: str = "worldgen"
    seed: int = 42
    scene_extent_meters: float = 50.0
    output_dir: str = "data/output"
    intermediate_dir: str = "data/intermediate"


class WorldGenConfig(BaseModel):
    project: ProjectConfig = Field(default_factory=ProjectConfig)
    gpu: GPUConfig = Field(default_factory=GPUConfig)
    stage1: Stage1Config = Field(default_factory=Stage1Config)
    stage2: Stage2Config = Field(default_factory=Stage2Config)
    stage3: Stage3Config = Field(default_factory=Stage3Config)
    stage4: Stage4Config = Field(default_factory=Stage4Config)
    export: ExportConfig = Field(default_factory=ExportConfig)


def load_config(config_path: str | Path | None = None) -> WorldGenConfig:
    """Load configuration from YAML file, falling back to defaults."""
    if config_path is None:
        return WorldGenConfig()

    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        raw: dict[str, Any] = yaml.safe_load(f) or {}

    return WorldGenConfig(**raw)
