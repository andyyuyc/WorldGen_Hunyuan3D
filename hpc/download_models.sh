#!/bin/bash
# ============================================================
# Download all required models to HPC scratch storage.
# Run this on a node with internet access (login node).
# ============================================================

GROUP_NAME="${GROUP_NAME:-your_group}"
PROJECT_DIR="/share/${GROUP_NAME}/${USER}/WorldGen"
VENV_DIR="${PROJECT_DIR}/venv"
HF_CACHE="/share/${GROUP_NAME}/${USER}/huggingface_cache"

source "${VENV_DIR}/bin/activate"
export HF_HOME="${HF_CACHE}"

echo "=== Downloading Models to ${HF_CACHE} ==="
echo ""

# Hunyuan3D-2 (shapegen + texgen models, ~25GB total)
echo "[1/3] Downloading Hunyuan3D-2..."
python -c "
from huggingface_hub import snapshot_download
snapshot_download('tencent/Hunyuan3D-2', cache_dir='${HF_CACHE}')
print('Hunyuan3D-2 download complete')
"

# Stable Diffusion (for ControlNet reference image generation in Stage 1)
echo "[2/3] Downloading Stable Diffusion v1.5..."
python -c "
from huggingface_hub import snapshot_download
snapshot_download('stable-diffusion-v1-5/stable-diffusion-v1-5', cache_dir='${HF_CACHE}')
print('SD v1.5 download complete')
"

# ControlNet depth model
echo "[3/3] Downloading ControlNet depth..."
python -c "
from huggingface_hub import snapshot_download
snapshot_download('lllyasviel/sd-controlnet-depth', cache_dir='${HF_CACHE}')
print('ControlNet download complete')
"

echo ""
echo "=== All models downloaded ==="
echo "Total size:"
du -sh "${HF_CACHE}"
