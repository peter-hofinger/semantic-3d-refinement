#!/bin/bash
#SBATCH --job-name=e27-smoke
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --output=/rhome/pr26rb01/mircea/logs/%x_%j.out
# E27 smoke test: two answers that looped in E26 (greedy, Qwen3.8-27B), at repetition penalty 1.0 and 1.1.
# Office progressive view 2 (list = view 1's 14 names from E26) and kitchen union view 3 (plain question).
set -e
export PATH=$HOME/.local/bin:$PATH UV_CACHE_DIR=/scratch/vc18/$USER/.cache/uv UV_LINK_MODE=copy
export HF_HOME=/scratch/vc18/$USER/.cache/huggingface HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd ~/mircea/semantic-3d-refinement
echo "node $(hostname) | $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader)"
uv run --no-sync python - <<'PY'
import time
from PIL import Image
from src.loaders import Views
from src.segmentation.vocabulary import NEW_ONLY, QUESTION, VlmVocabulary, parse_names

P = "data/processed/scannetpp/{}/views_n20_uniform_504.npz"
office = ["door", "wall", "floor", "whiteboard", "chair", "fan", "bucket", "box", "table", "cabinet",
          "light switch", "handle", "paper", "cardboard"]
cases = [
    ("office progressive view 2", Views.load(P.format("7b6477cb95")), 1, QUESTION + NEW_ONLY.format(listed=", ".join(office))),
    ("kitchen union view 3", Views.load(P.format("5ee7c22ba0")), 2, QUESTION),
]
vlm = VlmVocabulary(checkpoint="Qwen/Qwen3.8-27B")
for label, views, i, question in cases:
    for rp in (1.0, 1.1):
        vlm.repetition_penalty = rp
        t = time.perf_counter()
        text, truncated = vlm.ask(Image.fromarray(views.images[i]), question)
        names = parse_names(text, truncated=truncated)
        print(f"=== {label} ({views.names[i]}), rp {rp}: {len(names)} names, truncated={truncated}, "
              f"{time.perf_counter() - t:.1f}s, lines without newline issues: {all(len(l.split()) <= 3 for l in text.splitlines() if l.strip())}")
        print(text)
PY
