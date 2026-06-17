#!/usr/bin/env python3
"""把天工双手采集数据转换为 LeRobot 数据集。

默认用 13 条轨迹训练、1 条轨迹留出：`0000` 到 `0012` 写入训练集，
`0013` 不写入。状态和动作按真实采集数据保留完整手部 6 维，
形成左臂7 + 左手6 + 右臂7 + 右手6 的 26 维向量。
"""

from __future__ import annotations

from pathlib import Path
import shutil

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
from PIL import Image
import tqdm
import tyro

DEFAULT_RAW_DIR = Path("/data/caobochun/openpi/data/Tienkung_dual_hands_take_box_14")
DEFAULT_REPO_ID = "caobochun/tienkung_dual_hands_take_box_13_26d"
DEFAULT_ROOT = Path("/data/caobochun/openpi/data/lerobot") / DEFAULT_REPO_ID
DEFAULT_TASK = "Pick up the black box on the table with both hands, hold it briefly, then put the box down."


def _read_image(path: Path) -> np.ndarray:
    """读取 RGB 图像为 HWC uint8 数组。"""
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def _make_state_or_action(arm: np.ndarray, left_hand: np.ndarray, right_hand: np.ndarray) -> np.ndarray:
    """拼接为左臂7 + 左手6 + 右臂7 + 右手6 的 26 维向量。"""
    return np.concatenate(
        [
            arm[:7],
            left_hand[:6],
            arm[7:14],
            right_hand[:6],
        ]
    ).astype(np.float32)


def _episode_dirs(raw_dir: Path, exclude_episodes: set[str]) -> list[Path]:
    """列出需要转换的轨迹目录。"""
    episode_dirs = sorted(path for path in raw_dir.iterdir() if path.is_dir() and (path / "arm.npz").exists())
    return [path for path in episode_dirs if path.name not in exclude_episodes]


def _create_dataset(repo_id: str, *, root: Path, fps: int, overwrite: bool) -> LeRobotDataset:
    """创建空的 LeRobot 数据集。"""
    output_path = root.expanduser().resolve()
    if output_path.exists():
        if not overwrite:
            raise FileExistsError(f"数据集已存在: {output_path}。如需覆盖请传入 --overwrite")
        shutil.rmtree(output_path)

    return LeRobotDataset.create(
        repo_id=repo_id,
        root=output_path,
        robot_type="tienkung_dual_hands",
        fps=fps,
        features={
            "base_image": {
                "dtype": "image",
                "shape": (480, 640, 3),
                "names": ["height", "width", "channel"],
            },
            "left_image": {
                "dtype": "image",
                "shape": (240, 424, 3),
                "names": ["height", "width", "channel"],
            },
            "right_image": {
                "dtype": "image",
                "shape": (240, 424, 3),
                "names": ["height", "width", "channel"],
            },
            "state": {
                "dtype": "float32",
                "shape": (26,),
                "names": ["state"],
            },
            "actions": {
                "dtype": "float32",
                "shape": (26,),
                "names": ["actions"],
            },
        },
        use_videos=True,
        image_writer_threads=8,
        image_writer_processes=4,
    )


def convert(
    raw_dir: Path = DEFAULT_RAW_DIR,
    repo_id: str = DEFAULT_REPO_ID,
    root: Path = DEFAULT_ROOT,
    task: str = DEFAULT_TASK,
    *,
    fps: int = 20,
    exclude_episodes: tuple[str, ...] = ("0013",),
    overwrite: bool = True,
    push_to_hub: bool = False,
) -> None:
    """执行转换，并可选上传到 Hugging Face Hub。"""
    raw_dir = raw_dir.expanduser().resolve()
    if not raw_dir.exists():
        raise FileNotFoundError(f"原始数据目录不存在: {raw_dir}")

    dataset = _create_dataset(repo_id, root=root, fps=fps, overwrite=overwrite)
    train_episodes = _episode_dirs(raw_dir, set(exclude_episodes))
    if not train_episodes:
        raise ValueError(f"没有找到可转换的轨迹，raw_dir={raw_dir}")

    for episode_dir in tqdm.tqdm(train_episodes, desc="转换轨迹"):
        arm_data = np.load(episode_dir / "arm.npz")
        frame_count = len(arm_data["status_positions"])
        for frame_idx in range(frame_count):
            state = _make_state_or_action(
                arm_data["status_positions"][frame_idx],
                arm_data["left_hand_state_positions"][frame_idx],
                arm_data["right_hand_state_positions"][frame_idx],
            )
            action = _make_state_or_action(
                arm_data["cmd_positions"][frame_idx],
                arm_data["left_hand_cmd_positions"][frame_idx],
                arm_data["right_hand_cmd_positions"][frame_idx],
            )
            dataset.add_frame(
                {
                    "base_image": _read_image(episode_dir / "head" / f"{frame_idx:06d}.png"),
                    "left_image": _read_image(episode_dir / "hand_left" / f"{frame_idx:06d}.png"),
                    "right_image": _read_image(episode_dir / "hand_right" / f"{frame_idx:06d}.png"),
                    "state": state,
                    "actions": action,
                    "task": task,
                }
            )
        dataset.save_episode()

    if push_to_hub:
        dataset.push_to_hub(
            tags=["tienkung", "dual-hands", "openpi"],
            private=True,
            push_videos=True,
            license="apache-2.0",
        )

    print(f"已转换 {len(train_episodes)} 条训练轨迹到 {root.expanduser().resolve()}")
    if exclude_episodes:
        print(f"留出轨迹未写入训练集: {', '.join(exclude_episodes)}")


if __name__ == "__main__":
    tyro.cli(convert)
