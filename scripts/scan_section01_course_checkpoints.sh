#!/usr/bin/env bash

set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "usage: $0 ENV_NAME RUN_DIR_OR_CHECKPOINT GPU [EVAL_SEED ...]" >&2
  exit 2
fi

ENV_NAME=$1
RUN_DIR=$2
GPU=$3
shift 3

if [[ $# -eq 0 ]]; then
  set -- 2026 2028
fi

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON=${SECTION01_TORCH_PYTHON:-python3}
export CUDA_VISIBLE_DEVICES=$GPU
export PYTHONPATH="$ROOT/motrix_envs/src:$ROOT/motrix_rl/src${PYTHONPATH:+:$PYTHONPATH}"

shopt -s nullglob
if [[ -f "$RUN_DIR" && "$RUN_DIR" == *.pt ]]; then
    checkpoints=("$RUN_DIR")
elif [[ -f "$RUN_DIR" ]]; then
  checkpoints=()
  while IFS= read -r checkpoint; do
    [[ -n "$checkpoint" ]] && checkpoints+=("$checkpoint")
  done < "$RUN_DIR"
else
  checkpoints=("$RUN_DIR"/checkpoints/agent_*.pt)
fi
if [[ ${#checkpoints[@]} -eq 0 ]]; then
  echo "no checkpoint found at or under $RUN_DIR" >&2
  exit 1
fi

for checkpoint in "${checkpoints[@]}"; do
  for seed in "$@"; do
    "$PYTHON" "$ROOT/scripts/evaluate_section01_course_torch.py" \
      --env="$ENV_NAME" \
      --policy="$checkpoint" \
      --num-envs=1 \
      --episodes=1 \
      --max-steps=9500 \
      --seed="$seed"
  done
done
