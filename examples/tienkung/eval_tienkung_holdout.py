#!/usr/bin/env python3
"""用留出轨迹验证天工双手 pi0.5 checkpoint 的离线输出。

脚本读取原始留出 episode 的三路图像和真实状态，调用训练后的 policy
预测 50 步动作 chunk，并和同一 episode 中未来 50 步命令动作做数值对比。
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
from PIL import Image
import tyro

from openpi.policies import policy_config
from openpi.training import config as training_config


DEFAULT_CHECKPOINT = Path(
    "/data/caobochun/openpi/checkpoints/pi05_tienkung_finetune/"
    "tienkung_take_box_26d_20fps_h50_full_10k/10000"
)
DEFAULT_EPISODE = Path("/data/caobochun/openpi/data/Tienkung_dual_hands_take_box_14/0013")
DEFAULT_OUTPUT = Path(
    "/data/caobochun/openpi/eval_outputs/tienkung_take_box_26d_20fps_h50_full_10k_holdout0013"
)
DEFAULT_PROMPT = "Pick up the black box on the table with both hands, hold it briefly, then put the box down."


def _read_image(path: Path) -> np.ndarray:
    """读取 RGB 图像。"""
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def _make_state_or_action(arm: np.ndarray, left_hand: np.ndarray, right_hand: np.ndarray) -> np.ndarray:
    """按训练顺序拼接为 26 维：左臂7 + 左手6 + 右臂7 + 右手6。"""
    return np.concatenate([arm[:7], left_hand[:6], arm[7:14], right_hand[:6]]).astype(np.float32)


def _future_actions(arm_cmd: np.ndarray, left_cmd: np.ndarray, right_cmd: np.ndarray, start: int, horizon: int) -> np.ndarray:
    """取未来 horizon 步真实命令；越界时用最后一帧补齐。"""
    actions = []
    last = len(arm_cmd) - 1
    for idx in range(start, start + horizon):
        src = min(idx, last)
        actions.append(_make_state_or_action(arm_cmd[src], left_cmd[src], right_cmd[src]))
    return np.stack(actions, axis=0)


def _save_contact_sheet(output_dir: Path, episode_dir: Path, frame_indices: list[int]) -> None:
    """保存验证帧的三路图像拼图，便于人工确认视觉输入。"""
    rows = []
    for frame_idx in frame_indices:
        images = [
            Image.open(episode_dir / "head" / f"{frame_idx:06d}.png").convert("RGB").resize((320, 240)),
            Image.open(episode_dir / "hand_left" / f"{frame_idx:06d}.png").convert("RGB").resize((320, 240)),
            Image.open(episode_dir / "hand_right" / f"{frame_idx:06d}.png").convert("RGB").resize((320, 240)),
        ]
        row = Image.new("RGB", (960, 240))
        for col, image in enumerate(images):
            row.paste(image, (col * 320, 0))
        rows.append(row)

    sheet = Image.new("RGB", (960, 240 * len(rows)))
    for row_idx, row in enumerate(rows):
        sheet.paste(row, (0, row_idx * 240))
    sheet.save(output_dir / "holdout_visual_inputs.jpg", quality=95)


def main(
    checkpoint: Path = DEFAULT_CHECKPOINT,
    episode_dir: Path = DEFAULT_EPISODE,
    output_dir: Path = DEFAULT_OUTPUT,
    *,
    config_name: str = "pi05_tienkung_finetune",
    prompt: str = DEFAULT_PROMPT,
    num_frames: int = 6,
    seed: int = 0,
    device: str = "cuda",
) -> None:
    """执行留出集离线评估。"""
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = training_config.get_config(config_name)
    # 离线评估不需要 torch.compile，关闭后启动更快、显存更稳；权重形状不变。
    cfg = dataclasses.replace(cfg, model=dataclasses.replace(cfg.model, pytorch_compile_mode=None))
    policy = policy_config.create_trained_policy(
        cfg,
        checkpoint,
        default_prompt=prompt,
        sample_kwargs={"num_steps": 10},
        pytorch_device=device,
    )

    arm_data = np.load(episode_dir / "arm.npz")
    frame_count = len(arm_data["status_positions"])
    horizon = cfg.model.action_horizon
    max_start = max(0, frame_count - horizon - 1)
    frame_indices = np.linspace(0, max_start, num=min(num_frames, max_start + 1), dtype=int).tolist()
    rng = np.random.default_rng(seed)

    rows = []
    predictions = {}
    targets = {}

    for frame_idx in frame_indices:
        state = _make_state_or_action(
            arm_data["status_positions"][frame_idx],
            arm_data["left_hand_state_positions"][frame_idx],
            arm_data["right_hand_state_positions"][frame_idx],
        )
        target = _future_actions(
            arm_data["cmd_positions"],
            arm_data["left_hand_cmd_positions"],
            arm_data["right_hand_cmd_positions"],
            frame_idx,
            horizon,
        )
        obs = {
            "observation/base_image": _read_image(episode_dir / "head" / f"{frame_idx:06d}.png"),
            "observation/left_image": _read_image(episode_dir / "hand_left" / f"{frame_idx:06d}.png"),
            "observation/right_image": _read_image(episode_dir / "hand_right" / f"{frame_idx:06d}.png"),
            "observation/state": state,
            "prompt": prompt,
        }
        noise = rng.standard_normal((horizon, cfg.model.action_dim), dtype=np.float32)
        result = policy.infer(obs, noise=noise)
        pred = np.asarray(result["actions"], dtype=np.float32)

        arm_dims = [*range(0, 7), *range(13, 20)]
        hand_dims = [*range(7, 13), *range(20, 26)]
        abs_err = np.abs(pred[:, :26] - target)
        rows.append(
            {
                "frame": frame_idx,
                "infer_ms": float(result["policy_timing"]["infer_ms"]),
                "mae_all26": float(abs_err.mean()),
                "mae_arm14": float(abs_err[:, arm_dims].mean()),
                "mae_hand12": float(abs_err[:, hand_dims].mean()),
                "first_step_l2": float(np.linalg.norm(pred[0, :26] - target[0])),
                "chunk_l2_mean": float(np.linalg.norm(pred[:, :26] - target, axis=1).mean()),
            }
        )
        predictions[f"frame_{frame_idx:06d}"] = pred
        targets[f"frame_{frame_idx:06d}"] = target

    np.savez_compressed(output_dir / "holdout_predictions.npz", **predictions)
    np.savez_compressed(output_dir / "holdout_targets.npz", **targets)
    _save_contact_sheet(output_dir, episode_dir, frame_indices)

    csv_path = output_dir / "metrics.csv"
    with csv_path.open("w", encoding="utf-8") as f:
        f.write("frame,infer_ms,mae_all26,mae_arm14,mae_hand12,first_step_l2,chunk_l2_mean\n")
        for row in rows:
            f.write(
                f"{row['frame']},{row['infer_ms']:.3f},{row['mae_all26']:.6f},{row['mae_arm14']:.6f},"
                f"{row['mae_hand12']:.6f},{row['first_step_l2']:.6f},{row['chunk_l2_mean']:.6f}\n"
            )

    print(f"评估帧: {frame_indices}")
    print(f"输出目录: {output_dir}")
    for row in rows:
        print(
            "frame={frame} infer_ms={infer_ms:.1f} mae_all26={mae_all26:.4f} "
            "mae_arm14={mae_arm14:.4f} mae_hand12={mae_hand12:.4f} chunk_l2_mean={chunk_l2_mean:.4f}".format(**row)
        )
    print(f"指标 CSV: {csv_path}")


if __name__ == "__main__":
    tyro.cli(main)
