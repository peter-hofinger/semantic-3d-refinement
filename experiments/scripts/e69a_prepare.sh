#!/usr/bin/env bash
# E69a: prepare the scenes the experiment reads, those of the given splits (default: train val dev).
# Each step skips what exists: download and views (CPU, network), Deja View (GPU, ~7 s a scene), SAM 3
# (GPU, ~20 s a scene; the nyu40 prompts at threshold 0.3 of E34). About 1 h for the 40 train scenes.
#   experiments/scripts/e69a_prepare.sh [split ...]
set -euo pipefail
splits=("$@")
((${#splits[@]})) || splits=(train val dev)
uv run hypersim-download --splits "${splits[@]}"
uv run hypersim-prepare --splits "${splits[@]}"
views=$(uv run python - "${splits[@]}" <<'PY'
import sys
from pathlib import Path
from src.loaders.hypersim.config import load_config
root = Path("data/processed/hypersim")
for scene, cam in load_config(Path("configs/hypersim.yaml")).scenes(sys.argv[1:]).items():
    print(root / f"{scene}_{cam}" / "views_n20_uniform_504.npz")
PY
)
for v in $views; do
  [[ -e "$(dirname "$v")/dejaview_k16_views_n20_uniform_504.npz" ]] || echo "$v"
done | xargs -r uv run dejaview-predict
for v in $views; do
  [[ -e "$(dirname "$v")/sam3image-nyu40-t0.3_views_n20_uniform_504.npz" ]] || echo "$v"
done | xargs -r uv run sam3-segment --prompts configs/prompts/nyu40.txt --threshold 0.3
