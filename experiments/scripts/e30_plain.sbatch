#!/bin/bash
#SBATCH --job-name=e30-plain
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=/rhome/pr26rb01/mircea/logs/%x_%j.out
# E30, part 2: the three plain-camera scenes (ai_010_007, ai_006_007, ai_044_001, cam_00), same chain as
# e30_hypersim.sbatch, plus the restaurant + VLM SAM 3 run that failed in job 760 (ID_STRIDE overflow, fixed 60b3d68).
# Deja View K=16 -> VLM vocabulary (Qwen3.8-27B, union scheme, cap 64 tokens = E28 setting) ->
# SAM 3 image mode t0.3 with top100 and with the VLM list. Association + scoring run locally.
set -e
export PATH=$HOME/.local/bin:$PATH UV_CACHE_DIR=/scratch/vc18/$USER/.cache/uv UV_LINK_MODE=copy
export HF_HOME=/scratch/vc18/$USER/.cache/huggingface HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd ~/mircea/semantic-3d-refinement
echo "node $(hostname) | $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader) | $(git log --oneline -1)"
nvidia-smi --query-gpu=timestamp,memory.used --format=csv,noheader -lms 500 > ~/mircea/logs/gpumem_$SLURM_JOB_ID.csv &
stage() { local t=$(date +%s%N); echo "=== $1"; shift; "$@"; echo "=== took $(( ($(date +%s%N) - t) / 1000000 )) ms"; }
P=data/processed/hypersim
SCENES="ai_010_007_cam_00 ai_006_007_cam_00 ai_044_001_cam_00"
VIEWS=""
for s in $SCENES; do VIEWS="$VIEWS $P/$s/views_n20_uniform_504.npz"; done
for s in $SCENES; do
  stage "$s deja view" uv run --no-sync dejaview-predict $P/$s/views_n20_uniform_504.npz
done
stage "vlm union" uv run --no-sync vlm-vocabulary --checkpoint Qwen/Qwen3.8-27B --max-new-tokens 64 --max-gpu-gb 31 $VIEWS
for s in $SCENES; do
  V=$P/$s/views_n20_uniform_504.npz
  for prompts in data/scannetpp/metadata/semantic_benchmark/top100.txt $P/$s/vlm-qwen3827b-mt64_views_n20_uniform_504.txt; do
    stage "$s sam3 $(basename $prompts)" uv run --no-sync sam3-segment --mode image --threshold 0.3 --max-gpu-gb 30 \
      --prompts $prompts $V
  done
done
R=$P/ai_052_001_cam_00
stage "ai_052_001_cam_00 sam3 vlm (rerun)" uv run --no-sync sam3-segment --mode image --threshold 0.3 --max-gpu-gb 30   --prompts $R/vlm-qwen3827b-mt64_views_n20_uniform_504.txt $R/views_n20_uniform_504.npz
kill %1
echo "peak GPU memory used (nvidia-smi): $(cut -d, -f2 ~/mircea/logs/gpumem_$SLURM_JOB_ID.csv | sort -n | tail -1)"
