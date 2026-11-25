#!/usr/bin/env bash
# E69c: the whole ladder of docs/experiments.md for the O-Voxel generators of issue #69, in order, one gate per
# rung (experiments/scripts/e69c_gate.py): it stops where a gate lets nothing through. Every step skips the
# models and scenes it already has, so a stopped run resumes by running this again.
#   experiments/scripts/e69c.sh               # rungs 0 to 5; the ablations run when rung 5 passes
#   ALLOW_LONG=1 ...                          # accept a job over the 30 minute limit (rung 2 prints them)
# TensorBoard: uv run tensorboard --logdir runs/e69c/<rung>/tb; kill an arm whose train loss is flat after a few minutes.
set -euo pipefail
cd "$(dirname "$0")/../.."
cfg=configs/e69c.yaml
gate="uv run python -m experiments.scripts.e69c_gate"
oracle="uv run python -m experiments.scripts.e69c_oracle"
: "${PYTORCH_CUDA_ALLOC_CONF:=expandable_segments:True}"
export PYTORCH_CUDA_ALLOC_CONF
mkdir -p runs/e69c

train() {  # train <rung> <arm>...: one process per arm; EXTRA adds flags (the escalation's `--set`, the rates)
  local rung=$1; shift
  for arm in "$@"; do
    uv run generate-cloud --config "$cfg" --rung "$rung" --arms "$arm" ${EXTRA:-}
  done
}
score() {  # score <rung>: cloud_eval on the rung's clouds
  read -r config splits < <($gate config "$1")
  uv run python -m src.analysis.cloud_eval --cfg "$config" --splits $splits
}
passed() { tr '\n' ' ' < "runs/e69c/$1.pass"; }

echo "== data: scenes, DINOv2 features and their PCA (the PCA sees the train split only)"
experiments/scripts/e69a_prepare.sh train val dev
uv run python -m src.backbone.dinov2 "$cfg"

echo "== rung 0: the formulation without training (CPU): oracle, Déjà View and the density filter, then the damage map"
$oracle refs "$cfg" dev val
$oracle density "$cfg" dev val
score rung0
$gate gate rung0
split=$(sed -n 's/^split: //p' runs/e69c/split.yaml)
[[ "$split" == "$(sed -n 's/^split: *\([0-9]*\).*/\1/p' "$cfg" | head -1)" ]] || { echo "rung 0 chose split $split: set it in $cfg, commit, and run again"; exit 1; }
$oracle damage "$cfg" "$split" dev val

echo "== rung 1: the data figure, looked at before any training"
uv run python -m experiments.scripts.e69c_data "$cfg"

echo "== rung 2: speed of every arm; stops if a train step takes 1 s or a job passes 30 minutes"
uv run generate-cloud --config "$cfg" --rung overfit1 --arms ovoxel ovoxel-dino --profile --out runs/e69c
uv run generate-cloud --config "$cfg" --rung overfit1a --profile --out runs/e69c/latent_profile
cat runs/e69c/latent_profile/profile.yaml >> runs/e69c/profile.yaml
$gate profile

echo "== learning rate of each arm"
for rate in 3e-4 1e-3 3e-3; do train "lr_probe_$rate" ovoxel ovoxel-dino; done
$gate lr

echo "== rung 3a: the output alone, a free latent per voxel of the bedroom"
EXTRA="--lr-file runs/e69c/lr.yaml" train overfit1a latent
score overfit1a
$gate gate overfit1a

echo "== rung 3b: input to output on the bedroom; the pre-registered escalation, one change at a time"
declare -A setting=(
  [overfit1]=""
  [overfit1_e1]="steps=4800"
  [overfit1_e2]="steps=4800 widths=[32,64,128] max_parameters=1500000"
  [overfit1_e3]="steps=4800 widths=[32,64,128] max_parameters=1500000 cosine=true"
)
rung3=""
for r in overfit1 overfit1_e1 overfit1_e2 overfit1_e3; do
  EXTRA="--lr-file runs/e69c/lr.yaml" train "$r" ovoxel ovoxel-dino
  score "$r"
  if $gate gate "$r"; then rung3=$r; break; fi
done
[[ -n "$rung3" ]] || { echo "no arm closes half the gap after E3: not yet working"; exit 1; }
set_flags=()
[[ -z "${setting[$rung3]}" ]] || set_flags=(--set ${setting[$rung3]})

echo "== rung 4: the three dev scenes, the arms that passed, the setting that passed ($rung3)"
EXTRA="--lr-file runs/e69c/lr.yaml ${set_flags[*]}" train overfit3 $(passed "$rung3")
score overfit3
$gate gate overfit3

echo "== rung 5: generalize, 5 and 40 train scenes; the kept step is chosen on dev, val is held out"
EXTRA="--lr-file runs/e69c/lr.yaml ${set_flags[*]}" train generalize $(passed overfit3)
score generalize
$gate gate generalize

echo "== ablations of the best arm, three seeds: without DINOv2 (or with), vertex at the centre, SAM 3's classes"
best=$(head -1 runs/e69c/generalize.pass)
other=$([[ "$best" == *-dino ]] && echo "${best%-dino}" || echo "$best-dino")
EXTRA="--lr-file runs/e69c/lr.yaml --sizes 40 ${set_flags[*]}" train ablation "$best" "$other" "${best%-dino}-centre" "$best-sam3"
score ablation
$gate gate ablation
