# WorldGen on NCSU Hazel HPC

## Quick Start

### 1. Upload project to HPC

From your local machine:
```bash
# Replace YOUR_UNITY_ID with your NCSU unity ID
scp -r d:\Qiao\WorldGen YOUR_UNITY_ID@login.hpc.ncsu.edu:/share/YOUR_GROUP/YOUR_UNITY_ID/WorldGen
```

Or use git:
```bash
ssh YOUR_UNITY_ID@login.hpc.ncsu.edu
cd /share/YOUR_GROUP/YOUR_UNITY_ID
git clone <your-repo-url> WorldGen
```

### 2. Run setup (once)

```bash
ssh YOUR_UNITY_ID@login.hpc.ncsu.edu
export GROUP_NAME=your_group   # your research group name on Hazel
cd /share/${GROUP_NAME}/${USER}/WorldGen
bash hpc/setup_env.sh
```

This will:
- Create a Python virtual environment
- Install PyTorch + all dependencies
- Compile CUDA extensions (custom_rasterizer, mesh_processor)

### 3. Edit .env

```bash
nano /share/YOUR_GROUP/YOUR_UNITY_ID/WorldGen/.env
```

Set your `ANTHROPIC_API_KEY`:
```
ANTHROPIC_API_KEY="sk-ant-api03-..."
HF_HOME=/share/YOUR_GROUP/YOUR_UNITY_ID/huggingface_cache
```

### 4. Download models

```bash
bash hpc/download_models.sh
```

This downloads ~30GB of model weights to scratch storage. Run on login node (has internet).

### 5. Submit jobs

```bash
cd /share/YOUR_GROUP/YOUR_UNITY_ID/WorldGen

# Quick test (Hunyuan3D only, ~5 min)
bsub < hpc/job_test_hunyuan3d.sh

# Stage 1+2 only (~15 min)
bsub < hpc/job_stage12.sh

# Full pipeline (~1 hour)
bsub < hpc/job_run.sh
```

### 6. Check job status

```bash
bjobs           # list your jobs
bpeek <JOB_ID>  # see live output
cat logs/worldgen_<JOB_ID>.out  # see completed output
```

## File Structure

```
hpc/
├── README.md              # This file
├── setup_env.sh           # One-time environment setup
├── download_models.sh     # Download model weights
├── job_run.sh             # Full pipeline job (LSF)
├── job_stage12.sh         # Stage 1+2 only job (LSF)
└── job_test_hunyuan3d.sh  # Quick Hunyuan3D test job (LSF)
```

## Important Notes

- **Storage**: Use `/share/` (20TB) for data and models, NOT `/home/` (15GB limit)
- **GPU queue**: Jobs go to `gpu` queue. H100 and L40 GPUs available.
- **Time limits**: Default 2 hours for full pipeline. Adjust `-W` in job scripts.
- **Internet**: Compute nodes may not have internet. Download models first on login node.
- **API key**: Stage 1 calls Anthropic API. Ensure compute nodes can reach api.anthropic.com.

## Troubleshooting

### "module load cuda" fails
Try: `module avail cuda` to see available versions, then `module load cuda/12.x`

### CUDA extension compilation fails
Ensure CUDA toolkit matches PyTorch CUDA version:
```bash
python -c "import torch; print(torch.version.cuda)"
nvcc --version
```

### Models not found
Check HF_HOME is set correctly in .env and points to your scratch directory.

### API timeout
NCSU compute nodes might not have external network. Check with:
```bash
curl -s https://api.anthropic.com/ && echo "OK" || echo "No network"
```
If blocked, run Stage 1 locally and copy `data/intermediate/stage1/` to HPC.
