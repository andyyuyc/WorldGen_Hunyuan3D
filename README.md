# WorldGen

Text-to-3D world generation pipeline. Given a text prompt, generates a traversable 3D scene as a GLB file.

## Pipeline Overview

```
Stage 1: Scene Planning     — LLM generates scene layout, terrain, object placements
Stage 2: 3D Reconstruction  — Hunyuan3D-2.1 converts reference image to 3D mesh + texture
Stage 3: Mesh Decomposition — Splits monolithic mesh into individual objects (ground + 3-15 objects)
Stage 4: Scene Enhancement  — VLM-guided texture refinement per object
Export:  GLB                — Final textured scene file
```

## Requirements

- Python 3.10+
- CUDA GPU (recommended: 16GB+ VRAM for local, 80GB+ for HPC)
- [Anthropic API key](https://console.anthropic.com/) (for Stage 1 planning and Stage 4 VLM)
- ~30GB disk space for model weights

## Setup

### 1. Clone the repository

```bash
git clone <repo-url>
cd WorldGen
```

### 2. Initialize submodules

The Hunyuan3D-2 implementation is included as source in `third_party/`. No submodule init needed.

### 3. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # Linux/Mac
# or
venv\Scripts\activate           # Windows
```

### 4. Install PyTorch (with CUDA)

```bash
# CUDA 12.4 (recommended for H100/A100)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 5. Install dependencies

```bash
pip install -e ".[ml]"
pip install xxhash  # Optional but strongly recommended (50x faster mesh hashing)
```

### 6. Compile C++ extensions (required for texgen)

```bash
# On Linux/HPC (load CUDA module first):
module load cuda/12.4   # or cuda/12.6

cd third_party/Hunyuan3D-2/hy3dgen/texgen/differentiable_renderer
python setup.py build_ext --inplace

cd ../custom_rasterizer
TORCH_CUDA_ARCH_LIST="9.0" python setup.py build_ext --inplace
# For H100: 9.0, A100: 8.0, L40/L40S: 8.9, V100: 7.0
cd ../../../../..
```

```powershell
# On Windows (run from repo root, CUDA toolkit must be installed):
cd third_party\Hunyuan3D-2\hy3dgen\texgen\differentiable_renderer
python setup.py build_ext --inplace
cd ..\custom_rasterizer
python setup.py build_ext --inplace
cd ..\..\..\..\..
```

### 7. Set up environment variables

Copy `.env.example` to `.env` and fill in your API key:

```bash
cp .env.example .env
```

Edit `.env`:

```
ANTHROPIC_API_KEY=sk-ant-...
HF_HOME=/path/to/huggingface_cache   # Optional: redirect model downloads
```

### 8. Download models

Models are downloaded automatically from Hugging Face on first run (~30GB total):
- `tencent/Hunyuan3D-2.1` — Shape generation (3.0B)
- `tencent/Hunyuan3D-2` — Texture generation

To pre-download (recommended for HPC where compute nodes may not have internet):

```bash
python -c "
from huggingface_hub import snapshot_download
snapshot_download('tencent/Hunyuan3D-2.1', ignore_patterns=['*.bin'])
snapshot_download('tencent/Hunyuan3D-2', ignore_patterns=['*.bin'])
"
```

## Running the Pipeline

### Full pipeline

```bash
python scripts/run_pipeline.py "a medieval village on a hilltop with a castle"
```

Output is written to `data/output/`.

### Options

```bash
python scripts/run_pipeline.py "your prompt" \
  --config configs/default.yaml \   # Config file (default: configs/default.yaml)
  --run-id my_run \                  # Custom run ID (default: timestamp)
  --verbose                          # Enable debug logging
```

### Stage 1 only (fast, no GPU needed)

```bash
python scripts/run_pipeline.py "your prompt" --stage1-only
```

### Re-run Stage 3+ (skip Stage 1 & 2, use existing mesh)

```bash
python scripts/rerun_stage3.py
```

Reads from `data/intermediate/stage2/scene_mesh_aligned.glb`. Useful for iterating on decomposition and enhancement without re-running the slow 3D generation.

### Test Stage 2 only (Hunyuan3D shapegen + texgen)

```bash
python scripts/test_hunyuan3d_direct.py
```

Uses `data/intermediate/stage1/reference.png` as input. Outputs to `data/output/`.

## Running on NCSU Hazel HPC

See [`hpc/README.md`](hpc/README.md) for full HPC setup instructions.

Quick summary:

```bash
# 1. Upload project
scp -r WorldGen/ yyu55@login.hpc.ncsu.edu:/share/genlarp/yyu55/

# 2. SSH and set up environment
ssh yyu55@login.hpc.ncsu.edu
cd /share/genlarp/yyu55/WorldGen
bash hpc/setup_env.sh

# 3. Submit job
bsub < hpc/job_run.sh
```

## Configuration

Edit `configs/default.yaml` to change pipeline behavior. Key settings:

| Section | Key | Default | Description |
|---------|-----|---------|-------------|
| `stage2.hunyuan3d` | `octree_resolution` | 512 | Mesh detail (max 512) |
| `stage2.hunyuan3d` | `num_inference_steps` | 100 | More steps = better shape |
| `stage2.hunyuan3d` | `guidance_scale` | 7.5 | Image adherence strength |
| `stage3` | `min_face_count` | 500 | Minimum faces per object |
| `stage3` | `target_part_range` | [3, 15] | Target object count |
| `stage4.texture` | `texture_resolution` | 2048 | Output texture size |
| `stage4.vlm_model` | | claude-sonnet-4-... | VLM for object description |

## Output Structure

```
data/
  intermediate/
    stage1/   scene_spec.json, reference.png, terrain.obj, navmesh.obj
    stage2/   scene_mesh.glb, scene_mesh_aligned.glb
    stage3/   ground.obj, object_000.obj ... object_N.obj, manifest.json
    stage4/   object_000/final.glb, object_001/final.glb, ...
  output/
    scene.glb           Final scene (all objects merged)
```

## Troubleshooting

**texgen takes hours** — C++ extensions not compiled. Run the `setup.py build_ext --inplace` steps above.

**xatlas hangs on UV wrap** — Known issue with float64 inputs. Fixed in this repo (dtype conversion applied automatically).

**OOM during shapegen** — Reduce `octree_resolution` to 384 in `configs/default.yaml`.

**Stage 3 hangs on connectivity** — Usually caused by large mesh with many tiny fragments. The pipeline auto-decimates to 200k faces before decomposition.

**`CUDA_HOME not set` error** — Run `module load cuda/12.4` before compiling extensions (HPC only).
