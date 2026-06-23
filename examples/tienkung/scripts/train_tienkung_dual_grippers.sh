#!/usr/bin/env bash
# 训练天工双臂双夹爪 pi0.5 策略。

set -euo pipefail

OPENPI_DIR="${OPENPI_DIR:-/data/caobochun/openpi}"
CONFIG_NAME="${CONFIG_NAME:-pi05_tienkung_dual_grippers_finetune}"
EXP_NAME="${EXP_NAME:-tienkung_dual_grippers_take_box_pi05}"

cd "${OPENPI_DIR}"
. .venv/bin/activate

uv run scripts/train_pytorch.py "${CONFIG_NAME}" --exp_name "${EXP_NAME}"
