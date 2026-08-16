#!/usr/bin/env bash

set -euo pipefail

if [[ $# -ne 5 ]]; then
  echo "usage: $0 ENV_NAME CHECKPOINT_DIR GPU OUTPUT_DIR EVAL_SEED" >&2
  exit 2
fi

ENV_NAME=$1
CHECKPOINT_DIR=$2
GPU=$3
OUTPUT_DIR=$4
EVAL_SEED=$5

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON=${SECTION01_TORCH_PYTHON:-python3}
NUM_ENVS=${SECTION01_EVAL_NUM_ENVS:-8}
EPISODES=${SECTION01_EVAL_EPISODES:-8}

export CUDA_VISIBLE_DEVICES=$GPU
export PYTHONPATH="$ROOT/motrix_envs/src:$ROOT/motrix_rl/src${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$OUTPUT_DIR"

shopt -s nullglob
checkpoints=("$CHECKPOINT_DIR"/agent_*.pt)
if [[ ${#checkpoints[@]} -eq 0 ]]; then
  echo "no checkpoints found in $CHECKPOINT_DIR" >&2
  exit 1
fi

for checkpoint in "${checkpoints[@]}"; do
  name=$(basename "$checkpoint" .pt)
  "$PYTHON" "$ROOT/scripts/evaluate_section01_course_torch.py" \
    --env="$ENV_NAME" \
    --policy="$checkpoint" \
    --num-envs="$NUM_ENVS" \
    --episodes="$EPISODES" \
    --max-steps=10000 \
    --seed="$EVAL_SEED" \
    > "$OUTPUT_DIR/${name}_seed${EVAL_SEED}.log" 2>&1
done
