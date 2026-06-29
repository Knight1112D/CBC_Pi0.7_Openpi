"""快速计算 LeRobot 示例数据集的 OpenPI 归一化参数。

通用 scripts/compute_norm_stats.py 会经过完整图像 transform，计算示例全量数据时会浪费大量时间在视频解码上。
这个脚本直接读取 LeRobot v2.1 parquet 中的 state/action，并复现训练侧的 50 步 action chunk 与 delta action 规则。
"""

import os
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import tqdm

import openpi.shared.normalize as normalize

OPENPI_DIR = Path(os.environ.get("OPENPI_DIR", Path.cwd()))
DATASET_ROOT = Path(os.environ.get("LEROBOT_EXAMPLE_DATASET_ROOT", OPENPI_DIR / "data/lerobot/example/lerobot_v3_task"))
OUTPUT_DIR = Path(
    os.environ.get(
        "LEROBOT_EXAMPLE_NORM_STATS_DIR",
        OPENPI_DIR / "assets/pi05_lerobot_example_finetune/example/lerobot_v3_task",
    )
)
ACTION_HORIZON = 50


def main() -> None:
    # 示例动作顺序：左臂7 + 右臂7 + 头部2 + 左右夹爪2；夹爪保持绝对值。
    delta_mask = np.array([True] * 16 + [False] * 2)
    stats = {"state": normalize.RunningStats(), "actions": normalize.RunningStats()}

    episode_files = sorted((DATASET_ROOT / "data" / "chunk-000").glob("episode_*.parquet"))
    if not episode_files:
        raise FileNotFoundError(f"没有找到 LeRobot 示例 episode parquet：{DATASET_ROOT}")

    for path in tqdm.tqdm(episode_files, desc="Computing LeRobot example stats"):
        table = pq.read_table(path, columns=["observation.state", "action"])
        rows = table.to_pylist()
        state = np.asarray([row["observation.state"] for row in rows], dtype=np.float32)
        action = np.asarray([row["action"] for row in rows], dtype=np.float32)
        if len(state) == 0:
            continue

        indices = np.arange(len(state))[:, None] + np.arange(ACTION_HORIZON)[None, :]
        indices = np.minimum(indices, len(state) - 1)
        action_chunks = action[indices].copy()
        action_chunks[..., delta_mask] -= state[:, None, delta_mask]

        stats["state"].update(state)
        stats["actions"].update(action_chunks)

    norm_stats = {key: value.get_statistics() for key, value in stats.items()}
    normalize.save(OUTPUT_DIR, norm_stats)
    print(f"Writing stats to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
