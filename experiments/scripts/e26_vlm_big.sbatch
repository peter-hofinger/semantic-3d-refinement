#!/bin/bash
#SBATCH --job-name=e26-vlmbig
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=/rhome/pr26rb01/mircea/logs/%x_%j.out
# E26: bigger VLM (Qwen3.8-27B, 4-bit nf4, thinking off, greedy) with both schemes (per-view union and progressive)
# -> SAM 3 image mode t0.3 with the two new lists, both ScanNet++ scenes. The top100 / 2B caches come from job 658
# (same cluster). Association + scoring run locally.
set -e
export PATH=$HOME/.local/bin:$PATH UV_CACHE_DIR=/scratch/vc18/$USER/.cache/uv UV_LINK_MODE=copy
export HF_HOME=/scratch/vc18/$USER/.cache/huggingface HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd ~/mircea/semantic-3d-refinement
echo "node $(hostname) | $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader) | $(git log --oneline -1) + local vocabulary.py"
nvidia-smi --query-gpu=timestamp,memory.used --format=csv,noheader -lms 500 > ~/mircea/logs/gpumem_$SLURM_JOB_ID.csv &
stage() { local t=$(date +%s%N); echo "=== $1"; shift; "$@"; echo "=== took $(( ($(date +%s%N) - t) / 1000000 )) ms"; }
P=data/processed/scannetpp
VIEWS="$P/5ee7c22ba0/views_n20_uniform_504.npz $P/7b6477cb95/views_n20_uniform_504.npz"
CK=Qwen/Qwen3.8-27B
stage "vlm union" uv run --no-sync vlm-vocabulary --checkpoint $CK --max-gpu-gb 31 $VIEWS
stage "vlm progressive" uv run --no-sync vlm-vocabulary --checkpoint $CK --progressive --max-gpu-gb 31 $VIEWS
for s in 5ee7c22ba0 7b6477cb95; do
  V=$P/$s/views_n20_uniform_504.npz
  for prompts in $P/$s/vlm-qwen3827b_views_n20_uniform_504.txt $P/$s/vlmprog-qwen3827b_views_n20_uniform_504.txt; do
    stage "$s sam3 $(basename $prompts)" uv run --no-sync sam3-segment --mode image --threshold 0.3 --max-gpu-gb 30 \
      --prompts $prompts $V
  done
done
kill %1
echo "peak GPU memory used (nvidia-smi): $(cut -d, -f2 ~/mircea/logs/gpumem_$SLURM_JOB_ID.csv | sort -n | tail -1)"
