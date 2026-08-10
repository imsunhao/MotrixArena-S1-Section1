#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CACHE_DIR="${ROOT_DIR}/arena_assets"
DEST_DIR="${ROOT_DIR}/motrix_envs/src/motrix_envs/navigation/vbot/xmls"
STARTER_URL="https://dist.bj.bcebos.com/motphys-arena/starter_kit.zip"
STARTER_ZIP="${CACHE_DIR}/starter_kit.zip"

mkdir -p "${CACHE_DIR}" "${DEST_DIR}"
curl -L --fail --retry 5 -C - -o "${STARTER_ZIP}" "${STARTER_URL}"

WORK_DIR="$(mktemp -d "${CACHE_DIR}/extract.XXXXXX")"
trap 'find "${WORK_DIR}" -depth -delete 2>/dev/null || true' EXIT

unzip -q "${STARTER_ZIP}" \
  'starter_kit/starter_kit/navigation2.zip' -d "${WORK_DIR}"
unzip -q "${WORK_DIR}/starter_kit/starter_kit/navigation2.zip" \
  'navigation2/navigation/vbot_0218/navigation_vbot.zip' -d "${WORK_DIR}"
unzip -q "${WORK_DIR}/navigation2/navigation/vbot_0218/navigation_vbot.zip" \
  'vbot/xmls/assets/*' 'vbot/xmls/meshes/*' -d "${WORK_DIR}/vbot"

cp -a "${WORK_DIR}/vbot/vbot/xmls/assets" "${DEST_DIR}/"
cp -a "${WORK_DIR}/vbot/vbot/xmls/meshes" "${DEST_DIR}/"

echo "MotrixArena Section 1 assets installed in ${DEST_DIR}"
