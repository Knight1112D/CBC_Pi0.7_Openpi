#!/usr/bin/env bash
# 把一个本地 LeRobot v3 数据集转换为 OpenPI 当前兼容的 v2.1 数据。

set -euo pipefail

OPENPI_DIR="${OPENPI_DIR:-$(pwd)}"
RAW_DATASET="${RAW_DATASET:-${OPENPI_DIR}/data/zhuji_pick_and_place}"
LEROBOT_ROOT="${LEROBOT_ROOT:-${OPENPI_DIR}/data/lerobot}"
REPO_ID="${REPO_ID:-caobochun/zhuji_pick_and_place}"
CONVERTER_DIR="${CONVERTER_DIR:-${OPENPI_DIR}/scripts/lerobot_conversion}"
PYTHON="${PYTHON:-${OPENPI_DIR}/.venv/bin/python}"

DATASET_LINK="${LEROBOT_ROOT}/${REPO_ID}"
DATASET_PARENT="$(dirname "${DATASET_LINK}")"

mkdir -p "${DATASET_PARENT}"

if [[ -e "${DATASET_LINK}" && ! -L "${DATASET_LINK}" ]]; then
  echo "目标数据集路径已存在且不是软链接：${DATASET_LINK}" >&2
  echo "为避免覆盖已有数据，请先手动确认该目录是否可以转换。" >&2
  exit 1
fi

if [[ ! -e "${DATASET_LINK}" ]]; then
  ln -s "${RAW_DATASET}" "${DATASET_LINK}"
fi

cd "${CONVERTER_DIR}"

if [[ ! -x "${PYTHON}" ]]; then
  echo "找不到 OpenPI Python 环境：${PYTHON}" >&2
  exit 1
fi

"${PYTHON}" convert_v3_to_v2.py --repo-id "${REPO_ID}" --root "${LEROBOT_ROOT}"

echo "转换完成：${DATASET_LINK}"
echo "原 v3 数据备份/引用路径：${DATASET_LINK}_v3.0"
