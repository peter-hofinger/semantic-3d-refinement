#!/bin/bash
#SBATCH --job-name=pipeline-sam3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/rhome/pr26rb01/mircea/logs/%x_%j.out
# End-to-end check of the packaged segmenter: run-pipeline with the cached Deja View geometry and
# --segmenter sam3 (SAM 3 top100 t0.3 + working association) on both ScanNet++ scenes, then the instance scorer.
# Expected: within bf16 noise of the cluster top100 rows of E25-E28.
set -e
export PATH=$HOME/.local/bin:$PATH UV_CACHE_DIR=/scratch/vc18/$USER/.cache/uv UV_LINK_MODE=copy
export HF_HOME=/scratch/vc18/$USER/.cache/huggingface HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd ~/mircea/semantic-3d-refinement
echo "node $(hostname) | $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader)"
P=data/processed/scannetpp
for s in 5ee7c22ba0 7b6477cb95; do
  V=$P/$s/views_n20_uniform_504.npz
  t=$(date +%s)
  uv run --no-sync run-pipeline $V --geometry $P/$s/dejaview_k16_views_n20_uniform_504.npz --segmenter sam3 \
    --out outputs/pipeline/$s/cloud.ply
  echo "=== $s pipeline took $(( $(date +%s) - t )) s"
  uv run --no-sync eval-instances $V outputs/pipeline/$s/cloud_segmentation.npz
done
