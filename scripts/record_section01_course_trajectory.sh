#!/usr/bin/env bash

set -Eeuo pipefail

# Usage: record_section01_course_trajectory.sh TRAJECTORY OUTPUT [GPU] [SECONDS|auto] [ENV]

TRAJECTORY=${1:?captured trajectory is required}
OUTPUT=${2:?output path is required}
GPU=${3:-0}
RECORD_SECONDS=${4:-auto}
ENV_NAME=${5:-vbot-section01-xy-yaw-stable-v4-course}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
PYTHON=${SECTION01_REPLAY_PYTHON:-python3}
FPS=100
HOLD_SECONDS=5
START_DELAY_SECONDS=2
AUTO_RECORD_MARGIN_SECONDS=60

if [[ "$RECORD_SECONDS" == "auto" ]]; then
  RECORD_SECONDS=$("$PYTHON" - "$TRAJECTORY" "$FPS" "$HOLD_SECONDS" "$START_DELAY_SECONDS" "$AUTO_RECORD_MARGIN_SECONDS" <<'PY'
import math
import sys

import numpy as np

trajectory = np.load(sys.argv[1], allow_pickle=False)
frame_count = len(trajectory["dof_pos"])
seconds = frame_count / float(sys.argv[2]) + float(sys.argv[3]) + float(sys.argv[4])
print(math.ceil(seconds + float(sys.argv[5])))
PY
  )
fi

mkdir -p "$(dirname "$OUTPUT")"
export PYTHONPATH="$ROOT/motrix_envs/src:$ROOT/motrix_rl/src"
export MOTRIX_FOLLOW_CAMERA=1
export MOTRIX_HIDE_COLLISION_GEOMS=1
export MOTRIX_RECORDING_QUALITY=1
export MOTRIX_PLAY_START_DELAY_SECONDS="$START_DELAY_SECONDS"
export MOTRIX_RENDER_ENV_INDEX=0
export CUDA_VISIBLE_DEVICES="$GPU"
export DISPLAY=${SECTION01_RECORD_DISPLAY:-:98}
DISPLAY_NUMBER=${DISPLAY#:}

rm -f "/tmp/.X${DISPLAY_NUMBER}-lock"
PLAY_LOG=/tmp/section1_course_replay_record_$$.log
GST_LOG=/tmp/section1_course_replay_gst_$$.log
READY_FILE=/tmp/section1_course_replay_ready_$$
STOP_FILE=/tmp/section1_course_replay_stop_$$
rm -f "$PLAY_LOG" "$GST_LOG" "$READY_FILE" "$STOP_FILE"
export MOTRIX_RENDER_READY_FILE="$READY_FILE"

Xvfb "$DISPLAY" -screen 0 1280x720x24 -ac >"/tmp/section1_course_replay_xvfb_${DISPLAY_NUMBER}.log" 2>&1 &
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
"$PYTHON" "$ROOT/scripts/replay_section01_course_trajectory.py" \
  --env="$ENV_NAME" \
  --trajectory="$TRAJECTORY" \
  --fps="$FPS" \
  --hold-seconds="$HOLD_SECONDS" >"$PLAY_LOG" 2>&1 &
PLAY_PID=$!

for _ in $(seq 1 300); do
  if [[ -f "$READY_FILE" ]]; then
    break
  fi
  kill -0 "$PLAY_PID" 2>/dev/null || {
    echo "replay exited before the renderer became ready; see $PLAY_LOG" >&2
    exit 1
  }
  sleep 1
done
[[ -f "$READY_FILE" ]] || {
  echo "renderer did not become ready within 300 seconds" >&2
  exit 1
}

/usr/bin/python3 "$ROOT/scripts/record_x11_gst.py" \
  --display="$DISPLAY" --seconds="$RECORD_SECONDS" --stop-file="$STOP_FILE" \
  --output="$OUTPUT" >"$GST_LOG" 2>&1 &
GST_PID=$!

wait "$PLAY_PID" || true
PLAY_PID=""
touch "$STOP_FILE"
wait "$GST_PID" || true
GST_PID=""

echo "recorded $OUTPUT"
echo "replay log: $PLAY_LOG"
