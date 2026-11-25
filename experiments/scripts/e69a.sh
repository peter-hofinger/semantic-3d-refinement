#!/usr/bin/env bash
# E69a: the whole ladder of docs/experiments.md for the generators of issue #69, in order, one gate per
# rung (experiments/scripts/e69a_gate.py): it stops when a gate lets no arm through. Every step skips
# the models and scenes it already has, so a stopped run resumes by running this again.
#   JOBS=3 experiments/scripts/e69a.sh      # arms in parallel: each process needs at most 3.0 GiB of GPU memory and, with
#                                            the 40 train scenes loaded, about 10 GiB of RAM
#   ALLOW_LONG=1 ...                         # accept a job over the 30 minute limit (rung 2 prints them)
# TensorBoard: uv run tensorboard --logdir runs/e69a/<rung>/tb; kill an arm whose train loss is flat after a few minutes.
set -euo pipefail
cd "$(dirname "$0")/../.."
cfg=configs/e69a.yaml
jobs=${JOBS:-1}
gate="uv run python experiments/scripts/e69a_gate.py"
: "${PYTORCH_CUDA_ALLOC_CONF:=expandable_segments:True}"
export PYTORCH_CUDA_ALLOC_CONF

train() {  # train <rung> <arm>...: one process per arm, $jobs at a time; EXTRA adds flags
  local rung=$1; shift
  printf '%s\n' "$@" | xargs -P "$jobs" -I{} \
    uv run generate-cloud --config "$cfg" --rung "$rung" --arms {} ${EXTRA:-}
}
score() {  # score <rung>: cloud_eval on the rung's clouds
  read -r splits config < <($gate config "$1")
  uv run python -m src.analysis.cloud_eval --cfg "$config" --splits $splits
}
passed() { tr '\n' ' ' < "runs/e69a/$1.pass"; }

echo "== data (rung 1): scenes, reachability table and figure, the two oracles"
experiments/scripts/e69a_prepare.sh train val dev
uv run python experiments/scripts/e69a_data.py "$cfg" overfit3 dev val
uv run python experiments/scripts/e69a_oracle.py "$cfg" overfit1 dev val

echo "== rung 2: speed of every arm; stops if a train step takes 1 s or a job passes 30 minutes"
uv run generate-cloud --config "$cfg" --rung overfit1 --profile --out runs/e69a
$gate profile

echo "== learning rate of each arm"
for rate in 3e-4 1e-3 3e-3; do train "lr_probe_$rate" lattice gated implicit split seeded seeded-render; done
$gate lr

echo "== rung 3: overfit the bedroom"
EXTRA="--lr-file runs/e69a/lr.yaml" train overfit1 lattice gated implicit split seeded seeded-render
score overfit1
$gate gate overfit1

echo "== rung 4: overfit the three dev scenes, arms that passed rung 3"
EXTRA="--lr-file runs/e69a/lr.yaml" train overfit3 $(passed overfit1)
score overfit3
$gate gate overfit3

echo "== rung 5: generalize, 5 and 40 train scenes, validated on val, arms that passed rung 4"
EXTRA="--lr-file runs/e69a/lr.yaml" train generalize $(passed overfit3)
score generalize
$gate gate generalize

echo "== class sources on the best arm of rung 5: none, SAM 3, ground truth; three seeds"
best=$(head -1 runs/e69a/generalize.pass)
EXTRA="--lr-file runs/e69a/lr.yaml" train semantics "$best" "$best-sam3" "$best-gt"
score semantics
$gate gate semantics || echo "no class source beats the same arm without classes"
