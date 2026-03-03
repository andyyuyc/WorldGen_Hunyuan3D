#!/bin/bash
# ============================================================
# WorldGen Stage 1+2 Only - LSF Job Script for NCSU Hazel
# Submit with: bsub < hpc/job_stage12.sh
# ============================================================

#BSUB -J wg_s12
#BSUB -n 4
#BSUB -W 60
#BSUB -q gpu
#BSUB -gpu "num=1:mode=shared:mps=no"
#BSUB -R "rusage[mem=32GB]"
#BSUB -o logs/stage12_%J.out
#BSUB -e logs/stage12_%J.err

# --- Configuration ---
GROUP_NAME="${GROUP_NAME:-your_group}"
PROJECT_DIR="/share/${GROUP_NAME}/${USER}/WorldGen"
VENV_DIR="${PROJECT_DIR}/venv"
PROMPT="${1:-a medieval village on a hilltop with a castle}"

# --- Setup ---
cd "${PROJECT_DIR}"
mkdir -p logs

module load cuda/12.6
export CUDA_HOME=$CUDA_ROOT
export TORCH_CUDA_ARCH_LIST="9.0"
export HF_HUB_CACHE="/share/${GROUP_NAME}/${USER}/huggingface_cache"
source "${VENV_DIR}/bin/activate"
set -a; source .env; set +a

echo "=== WorldGen Stage 1+2 ==="
echo "Date: $(date)"
echo "GPU:  $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo ""

python scripts/run_stage12.py "${PROMPT}"

echo "=== Done: $(date) ==="
