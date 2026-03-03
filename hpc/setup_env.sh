#!/bin/bash
# ============================================================
# WorldGen HPC Environment Setup for NCSU Hazel Cluster
# Run this ONCE on a login node to set up the environment.
# ============================================================

set -e

# --- Configuration ---
# Change GROUP_NAME to your research group on Hazel
GROUP_NAME="${GROUP_NAME:-your_group}"
PROJECT_DIR="/share/${GROUP_NAME}/${USER}/WorldGen"
HF_CACHE="/share/${GROUP_NAME}/${USER}/huggingface_cache"
VENV_DIR="${PROJECT_DIR}/venv"

echo "=== WorldGen HPC Setup ==="
echo "Project dir: ${PROJECT_DIR}"
echo "HF cache:    ${HF_CACHE}"
echo "Venv dir:    ${VENV_DIR}"
echo ""

# --- Step 1: Copy project files ---
if [ ! -d "${PROJECT_DIR}/src" ]; then
    echo "[1/6] Copying project files..."
    # If running from the project root:
    SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
    SOURCE_DIR="$(dirname "${SCRIPT_DIR}")"
    mkdir -p "${PROJECT_DIR}"
    rsync -av --exclude='venv' --exclude='.venv' --exclude='data' \
        --exclude='__pycache__' --exclude='*.egg-info' --exclude='*.pyc' \
        --exclude='build' --exclude='dist' \
        "${SOURCE_DIR}/" "${PROJECT_DIR}/"
    mkdir -p "${PROJECT_DIR}/data/intermediate/stage1"
    mkdir -p "${PROJECT_DIR}/data/intermediate/stage2"
    mkdir -p "${PROJECT_DIR}/data/intermediate/stage3"
    mkdir -p "${PROJECT_DIR}/data/intermediate/stage4"
    mkdir -p "${PROJECT_DIR}/data/output"
else
    echo "[1/6] Project already exists, skipping copy."
fi

# --- Step 2: Load modules ---
echo "[2/6] Loading modules..."
module load cuda
module load python  # or: module load anaconda3
echo "  CUDA: $(nvcc --version | grep release | awk '{print $6}')"
echo "  Python: $(python3 --version)"

# --- Step 3: Create virtual environment ---
if [ ! -d "${VENV_DIR}" ]; then
    echo "[3/6] Creating virtual environment..."
    python3 -m venv "${VENV_DIR}"
else
    echo "[3/6] Venv already exists, skipping."
fi

source "${VENV_DIR}/bin/activate"

# --- Step 4: Install PyTorch + dependencies ---
echo "[4/6] Installing dependencies..."
pip install --upgrade pip

# PyTorch with CUDA (check module cuda version and adjust)
# For CUDA 12.x:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Core dependencies
pip install pyyaml pydantic python-dotenv anthropic trimesh open3d
pip install numpy scipy scikit-learn Pillow opencv-python-headless
pip install diffusers transformers accelerate safetensors
pip install xatlas pybind11 ninja einops omegaconf pymeshlab rembg onnxruntime

# --- Step 5: Create .env file ---
echo "[5/6] Setting up .env file..."
ENV_FILE="${PROJECT_DIR}/.env"
if [ ! -f "${ENV_FILE}" ]; then
    echo "ANTHROPIC_API_KEY=\"your-api-key-here\"" > "${ENV_FILE}"
    echo "HF_HOME=${HF_CACHE}" >> "${ENV_FILE}"
    echo "  Created ${ENV_FILE} - EDIT THIS with your API key!"
else
    echo "  .env already exists."
fi

# --- Step 6: Compile C extensions ---
echo "[6/6] Compiling C extensions..."

# custom_rasterizer (CUDA extension)
cd "${PROJECT_DIR}/third_party/Hunyuan3D-2/hy3dgen/texgen/custom_rasterizer"
python setup.py install 2>&1 | tail -5
echo "  custom_rasterizer: done"

# mesh_processor (C++ extension)
cd "${PROJECT_DIR}/third_party/Hunyuan3D-2/hy3dgen/texgen/differentiable_renderer"
python setup.py install 2>&1 | tail -5
echo "  mesh_processor: done"

cd "${PROJECT_DIR}"

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Next steps:"
echo "  1. Edit ${ENV_FILE} with your ANTHROPIC_API_KEY"
echo "  2. Download HunyuanF3D models (see below)"
echo "  3. Submit jobs with: bsub < hpc/job_run.sh"
echo ""
echo "To download models (run on login node or node with internet):"
echo "  source ${VENV_DIR}/bin/activate"
echo "  HF_HOME=${HF_CACHE} python -c \\"
echo "    \"from huggingface_hub import snapshot_download; \\"
echo "    snapshot_download('tencent/Hunyuan3D-2', cache_dir='${HF_CACHE}')\""
echo ""
