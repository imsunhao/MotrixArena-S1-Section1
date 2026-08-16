#!/usr/bin/env bash

set -Eeuo pipefail

# Usage: record_section01_course_torch.sh POLICY OUTPUT [SEED] [SKIP_EPISODES] [GPU] [SECONDS] [ENV]

POLICY=${1:?policy checkpoint is required}
OUTPUT=${2:?output path is required}
SEED=${3:-2026}
SKIP_EPISODES=${4:-7}
GPU=${5:-0}
RECORD_SECONDS=${6:-65}
ENV_NAME=${7:-vbot-section01-peer-xy-yaw-stable-v4-course}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
PYTHON=${SECTION01_TORCH_PYTHON:-python3}

mkdir -p "$(dirname "$OUTPUT")"
export PYTHONPATH="$ROOT/motrix_envs/src:$ROOT/motrix_rl/src"
export MOTRIX_FOLLOW_CAMERA=1
export MOTRIX_PLAY_START_DELAY_SECONDS=2
export CUDA_VISIBLE_DEVICES="$GPU"
export DISPLAY=:98

rm -f /tmp/.X98-lock
PLAY_LOG=/tmp/section1_course_play_record_$$.log
GST_LOG=/tmp/section1_course_gst_record_$$.log
READY_FILE=/tmp/section1_course_render_ready_$$
rm -f "$PLAY_LOG" "$GST_LOG" "$READY_FILE"
export MOTRIX_RENDER_READY_FILE="$READY_FILE"

Xvfb "$DISPLAY" -screen 0 1280x720x24 -ac >/tmp/section1_course_xvfb_record.log 2>&1 &
XVFB_PID=$!
GST_PID=""
PLAY_PID=""
cleanup() {
  set +e
  if [[ -n "$PLAY_PID" ]] && kill -0 "$PLAY_PID" 2>/dev/null; then
    kill -TERM "$PLAY_PID" 2>/dev/null
    wait "$PLAY_PID" 2>/dev/null
  fi
  if [[ -n "$GST_PID" ]] && kill -0 "$GST_PID" 2>/dev/null; then
    kill -INT "$GST_PID" 2>/dev/null
    wait "$GST_PID" 2>/dev/null
  fi
  kill "$XVFB_PID" 2>/dev/null
}
trap cleanup EXIT INT TERM

sleep 2
"$PYTHON" "$ROOT/scripts/play_section01_course_torch_episode.py" \
  --env="$ENV_NAME" \
  --policy="$POLICY" \
  --seed="$SEED" \
  --skip-episodes="$SKIP_EPISODES" \
  --fps=100 >"$PLAY_LOG" 2>&1 &
PLAY_PID=$!

for _ in $(seq 1 300); do
  if [[ -f "$READY_FILE" ]]; then
    break
  fi
  kill -0 "$PLAY_PID" 2>/dev/null || {
    echo "play exited before the renderer became ready; see $PLAY_LOG" >&2
    exit 1
  }
  sleep 1
done
[[ -f "$READY_FILE" ]] || {
  echo "renderer did not become ready within 300 seconds" >&2
  exit 1
}

/usr/bin/python3 "$ROOT/scripts/record_x11_gst.py" \
  --display="$DISPLAY" --seconds="$RECORD_SECONDS" --output="$OUTPUT" >"$GST_LOG" 2>&1 &
GST_PID=$!

wait "$GST_PID" || true
GST_PID=""
wait "$PLAY_PID" || true
PLAY_PID=""

echo "recorded $OUTPUT"
echo "play log: $PLAY_LOG"
