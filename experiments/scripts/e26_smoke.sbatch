#!/bin/bash
#SBATCH --job-name=e26-smoke
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:45:00
#SBATCH --output=/rhome/pr26rb01/mircea/logs/%x_%j.out
# E26 smoke test: Qwen3.8-27B (4-bit nf4) on one kitchen view, plain question and the progressive question with a
# short list. Measures load time, tokens/s and peak GPU memory, to size the full E26 job.
set -e
export PATH=$HOME/.local/bin:$PATH UV_CACHE_DIR=/scratch/vc18/$USER/.cache/uv UV_LINK_MODE=copy
export HF_HOME=/scratch/vc18/$USER/.cache/huggingface HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd ~/mircea/semantic-3d-refinement
echo "node $(hostname) | $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader)"
uv run --no-sync python - <<'EOF'
import time, torch
from PIL import Image
from src.loaders import Views
from src.segmentation.vocabulary import NEW_ONLY, QUESTION, VlmVocabulary

views = Views.load("data/processed/scannetpp/5ee7c22ba0/views_n20_uniform_504.npz")
vlm = VlmVocabulary(checkpoint="Qwen/Qwen3.8-27B")
t = time.perf_counter(); vlm._load(); print(f"load {time.perf_counter() - t:.0f}s, GPU {torch.cuda.memory_allocated() / 1e9:.1f} GB")
image = Image.fromarray(views.images[0])
for question in (QUESTION, QUESTION + NEW_ONLY.format(listed="wall, floor, cabinet, table")):
    t = time.perf_counter()
    text, truncated = vlm.ask(image, question)
    dt = time.perf_counter() - t
    n = len(vlm._processor.tokenizer(text)["input_ids"])
    print(f"--- {n} tokens, {dt:.1f}s, {n / dt:.1f} tok/s, truncated={truncated}, peak {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")
    print(text)
EOF
