#!/bin/bash
# ============================================================
# WorldGen Full Pipeline - LSF Job Script for NCSU Hazel
# Submit with: bsub < hpc/job_run.sh
# ============================================================

#BSUB -J worldgen
#BSUB -n 8
#BSUB -W 120
#BSUB -q gpu
#BSUB -gpu "num=1:mode=shared:mps=no"
#BSUB -R "rusage[mem=64GB]"
#BSUB -o logs/worldgen_%J.out
#BSUB -e logs/worldgen_%J.err

# --- Configuration ---
GROUP_NAME="${GROUP_NAME:-your_group}"
PROJECT_DIR="/share/${GROUP_NAME}/${USER}/WorldGen"
VENV_DIR="${PROJECT_DIR}/venv"
PROMPT="${1:-a medieval village on a hilltop with a castle}"

# --- Setup environment ---
cd "${PROJECT_DIR}"
mkdir -p logs

module load cuda/12.6
export CUDA_HOME=$CUDA_ROOT
export TORCH_CUDA_ARCH_LIST="9.0"
export HF_HUB_CACHE="/share/${GROUP_NAME}/${USER}/huggingface_cache"
source "${VENV_DIR}/bin/activate"

# Load env vars (.env has ANTHROPIC_API_KEY and HF_HOME)
set -a
source .env
set +a

echo "=== WorldGen Job Start ==="
echo "Date:   $(date)"
echo "Host:   $(hostname)"
echo "GPU:    $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "Prompt: ${PROMPT}"
echo ""

# --- Run full pipeline ---
python scripts/run_pipeline.py "${PROMPT}"

echo ""
echo "=== Job Complete ==="
echo "Date: $(date)"
