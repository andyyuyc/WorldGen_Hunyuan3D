#!/bin/bash
# ============================================================
# Quick test: Hunyuan3D shapegen + texgen timing
# Submit with: bsub < hpc/job_test_hunyuan3d.sh
# ============================================================

#BSUB -J wg_test
#BSUB -n 4
#BSUB -W 480
#BSUB -q gpu
#BSUB -gpu "num=1:mode=exclusive_process:mps=no"
#BSUB -R "rusage[mem=64GB]"
#BSUB -B
#BSUB -N
#BSUB -o logs/test_h3d_%J.out
#BSUB -e logs/test_h3d_%J.err

GROUP_NAME="${GROUP_NAME:-your_group}"
PROJECT_DIR="/share/${GROUP_NAME}/${USER}/WorldGen"
VENV_DIR="${PROJECT_DIR}/venv"

cd "${PROJECT_DIR}"
mkdir -p logs

module load cuda/12.6
export CUDA_HOME=$CUDA_ROOT
export TORCH_CUDA_ARCH_LIST="9.0"
export HF_HUB_CACHE="/share/${GROUP_NAME}/${USER}/huggingface_cache"
source "${VENV_DIR}/bin/activate"
set -a; source .env; set +a

echo "=== Hunyuan3D Direct Test ==="
echo "Date: $(date)"
nvidia-smi
echo ""

python scripts/test_hunyuan3d_direct.py

echo "=== Done: $(date) ==="
