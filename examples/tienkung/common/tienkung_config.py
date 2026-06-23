"""天工双手部署配置。"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

import numpy as np

STATE_DIM = 26
MODEL_IMAGE_SIZE = 224


@dataclasses.dataclass
class Args:
    """天工双手部署参数。"""

    config_path: str | None = None

    # ROS 参数
    ros_domain_id: str = "0"
    arm_status_topic: str = "/arm/status"
    arm_command_topic: str = "/arm/cmd_pos"
    base_image_topic: str = "/camera/color/image_raw/compressed"
    left_wrist_image_topic: str = "/camera/d405_left/color/image_h264"
    right_wrist_image_topic: str = "/camera/d405_right/color/image_h264"
    base_image_type: str = "compressed_image"
    left_wrist_image_type: str = "compressed_video"
    right_wrist_image_type: str = "compressed_video"

    # 电机顺序必须和训练数据一致：左臂7 + 左手6 + 右臂7 + 右手6。
    left_arm_joint_ids: str = "11,12,13,14,15,16,17"
    left_hand_joint_ids: str = ""
    right_arm_joint_ids: str = "21,22,23,24,25,26,27"
    right_hand_joint_ids: str = ""

    # policy server 参数
    remote_host: str = "0.0.0.0"
    remote_port: int = 8000
    prompt: str = "Use both hands to pick up the black box on the table, hold it briefly, then put it down."

    # 异步推理和控制参数
    controller_mode: str = "async"
    control_hz: float = 20.0
    policy_action_hz: float = 20.0
    open_loop_horizon: int = 8
    max_action_chunk_len: int = 32
    interpolation: str = "linear"
    request_when_remaining_steps: int = 10
    request_immediately_after_chunk: bool = True
    rtc_min_horizon: int = 8
    rtc_delay_buffer_size: int = 8
    rtc_initial_delay_steps: int = 8
    command_speed: float = 5.0
    command_current: float = 20.0

    # 安全限幅。手部默认按 0~1 的开合/归一化位置处理，如真实范围不同请通过参数覆盖。
    arm_lower_limits: str = "-2.96,-3.4,-1.74,-2.61,-2.96,-1.3,-1.04"
    arm_upper_limits: str = "2.96,0.2618,2.96,0.26,2.96,1.65,0.78"
    hand_lower_limits: str = "0,0,0,0,0,0"
    hand_upper_limits: str = "1,1,1,1,1,1"

    # 日志
    log_dir: str = "/data/caobochun/openpi/examples/tienkung/logs"
    save_action_chunks: bool = True


def load_json_config(config_path: str) -> dict[str, Any]:
    """读取 JSON 配置文件。"""
    with Path(config_path).open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("配置文件顶层必须是 JSON object。")
    return data


def merge_config_file(args: Args) -> Args:
    """合并配置文件和命令行参数，命令行显式传入的值优先。"""
    if args.config_path is None:
        return args

    defaults = Args(config_path=args.config_path)
    config_data = load_json_config(args.config_path)
    field_names = {field.name for field in dataclasses.fields(Args)}
    unknown_keys = sorted(set(config_data) - field_names)
    if unknown_keys:
        raise ValueError(f"配置文件包含未知字段: {unknown_keys}")

    merged = dataclasses.asdict(args)
    for key, value in config_data.items():
        if key == "config_path":
            continue
        if getattr(args, key) == getattr(defaults, key):
            merged[key] = value
    return Args(**merged)


def parse_csv_ints(value: str) -> list[int]:
    """解析逗号分隔的整数列表。"""
    if not value.strip():
        return []
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def parse_csv_floats(value: str) -> np.ndarray:
    """解析逗号分隔的浮点数组。"""
    if not value.strip():
        return np.array([], dtype=np.float32)
    return np.array([float(x.strip()) for x in value.split(",") if x.strip()], dtype=np.float32)


@dataclasses.dataclass(frozen=True)
class RobotLayout:
    """机器人 26 维状态布局和限幅。"""

    state_joint_ids: list[int]
    lower_limits: np.ndarray
    upper_limits: np.ndarray


def build_robot_layout(args: Args) -> RobotLayout:
    """根据配置生成状态顺序和限幅。"""
    left_arm_joint_ids = parse_csv_ints(args.left_arm_joint_ids)
    left_hand_joint_ids = parse_csv_ints(args.left_hand_joint_ids)
    right_arm_joint_ids = parse_csv_ints(args.right_arm_joint_ids)
    right_hand_joint_ids = parse_csv_ints(args.right_hand_joint_ids)
    state_joint_ids = [*left_arm_joint_ids, *left_hand_joint_ids, *right_arm_joint_ids, *right_hand_joint_ids]
    if len(state_joint_ids) != STATE_DIM:
        raise ValueError(
            "电机 ID 数量必须为 26，顺序为左臂7 + 左手6 + 右臂7 + 右手6；" f"当前数量为 {len(state_joint_ids)}。"
        )

    arm_lower = parse_csv_floats(args.arm_lower_limits)
    arm_upper = parse_csv_floats(args.arm_upper_limits)
    hand_lower = parse_csv_floats(args.hand_lower_limits)
    hand_upper = parse_csv_floats(args.hand_upper_limits)
    if len(arm_lower) != 7 or len(arm_upper) != 7 or len(hand_lower) != 6 or len(hand_upper) != 6:
        raise ValueError("限幅参数长度必须为：手臂 7 维、单手 6 维。")

    return RobotLayout(
        state_joint_ids=state_joint_ids,
        lower_limits=np.concatenate([arm_lower, hand_lower, arm_lower, hand_lower]).astype(np.float32),
        upper_limits=np.concatenate([arm_upper, hand_upper, arm_upper, hand_upper]).astype(np.float32),
    )
