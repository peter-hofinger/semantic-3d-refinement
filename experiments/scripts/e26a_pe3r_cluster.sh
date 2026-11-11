#!/bin/bash
# E26a: PE3R on ETH3D's 13 high-res training scenes, then its scores (experiments/LOG.md).
# PE3R keeps five models on the GPU (>10 GB), so it runs on a 32 GB RTX 5090. Two steps,
# from the repo root:
#   setup  login node, needs git and internet: clones PE3R (hujiecpp/PE3R @ a6c3683),
#          builds its environment, fetches its checkpoints and Hugging Face models and
#          the ETH3D archives of the scenes without prepared views.
#   run    GPU job, offline: prepares the missing views from the archives, runs
#          experiments/scripts/pe3r_predict.py per scene (finished scenes are skipped)
#          and scores the points with experiments/scripts/e26a_score_points.py.
# E26A_DATA (optional) puts everything large outside the repo: ETH3D's archives and
# views, PE3R and its environment, the Hugging Face and uv caches; data/eth3d,
# data/processed/eth3d and data/third_party/pe3r become links into it.
set -euo pipefail
step=${1:?usage: e26a_pe3r_cluster.sh setup|run}
SCENES="courtyard delivery_area electro facade kicker meadow office pipes playground relief relief_2 terrace terrains"
PE3R=data/third_party/pe3r
HF_MODELS="naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric facebook/sam2.1-hiera-large facebook/sam-vit-huge google/siglip-large-patch16-256"

if [ -n "${E26A_DATA:-}" ]; then
  export HF_HOME=$E26A_DATA/hf UV_CACHE_DIR=$E26A_DATA/uv-cache
  mkdir -p data/processed data/third_party
  for pair in "data/eth3d:eth3d" "data/processed/eth3d:processed_eth3d" "$PE3R:pe3r"; do
    link=${pair%%:*} target=$E26A_DATA/${pair##*:}
    mkdir -p "$target"
    if [ ! -L "$link" ]; then
      [ -e "$link" ] && { echo "$link exists and is not a link; move it into $target"; exit 1; }
      ln -s "$target" "$link"
    fi
  done
fi

missing() {  # scenes without prepared views
  for s in $SCENES; do
    [ -f data/processed/eth3d/$s/gt_views_n20_uniform_504.npz ] || echo $s
  done
}

case $step in
setup)
  [ -d $PE3R/.git ] || git clone -q https://github.com/hujiecpp/PE3R.git $PE3R
  git -C $PE3R checkout -q a6c3683
  if [ ! -x $PE3R/.venv/bin/python ]; then
    uv venv --no-config -q -p 3.11 $PE3R/.venv
    # the RTX 5090 (sm_120) needs a CUDA 12.8 build; SAM in transformers needs torch >= 2.6
    uv pip install --no-config -q -p $PE3R/.venv/bin/python torch==2.7.1 torchvision==0.22.1 \
      --index-url https://download.pytorch.org/whl/cu128
    printf 'torch==2.7.1\ntorchvision==0.22.1\n' > $PE3R/constraints.txt
    uv pip install --no-config -q -p $PE3R/.venv/bin/python -r $PE3R/requirements.txt \
      requests -c $PE3R/constraints.txt
  fi
  mkdir -p $PE3R/checkpoints
  for f in ObjectAwareModel.pt Prompt_guided_Mask_Decoder.pt; do
    [ -s $PE3R/checkpoints/$f ] || curl -sfL -o $PE3R/checkpoints/$f \
      https://github.com/hujiecpp/PE3R/releases/download/checkpoints/$f
  done
  for m in $HF_MODELS; do
    $PE3R/.venv/bin/python -c "import sys; from huggingface_hub import snapshot_download as s; s(sys.argv[1])" "$m"
  done
  for s in $(missing); do uv run --no-sync eth3d-download --scenes $s; done
  echo "setup done; scenes to prepare in the job: $(missing | tr '\n' ' ')"
  ;;
run)
  export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
  for s in $(missing); do uv run --no-sync eth3d-prepare --scenes $s; done
  mkdir -p runs/e26a/pe3r
  for s in $SCENES; do
    [ -f runs/e26a/pe3r/$s.npz ] && continue
    t=$(date +%s)
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=experiments/scripts \
      $PE3R/.venv/bin/python experiments/scripts/pe3r_predict.py \
      data/processed/eth3d/$s/views_n20_uniform_504.npz runs/e26a/pe3r/$s.npz
    echo "$s: PE3R in $(( $(date +%s) - t )) s"
  done
  uv run --no-sync python experiments/scripts/e26a_score_points.py runs/e26a/pe3r
  ;;
*) echo "unknown step: $step"; exit 2 ;;
esac
