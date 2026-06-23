#!/usr/bin/env python3
"""天工双手 pi0.5 异步推理部署入口。"""
# ruff: noqa: E402

from __future__ import annotations

from collections import deque
import os
from pathlib import Path
import sys
import time

import numpy as np
import rclpy
from rclpy.node import Node
import tyro

OPENPI_ROOT = Path(__file__).resolve().parents[3]
if str(OPENPI_ROOT) not in sys.path:
    sys.path.insert(0, str(OPENPI_ROOT))

from examples.tienkung.common.async_policy import AsyncPolicyProcess
from examples.tienkung.common.ros_io import COMPRESSED_IMAGE_MSG
from examples.tienkung.common.ros_io import TienkungRosIO
from examples.tienkung.common.tienkung_config import STATE_DIM
from examples.tienkung.common.tienkung_config import Args
from examples.tienkung.common.tienkung_config import build_robot_layout
from examples.tienkung.common.tienkung_config import merge_config_file
from examples.tienkung.common.trajectory import make_execution_plan
from examples.tienkung.rtc.rtc_chunker import RtcChunker


def _format_timing(result: dict) -> str:
    """把推理耗时整理成一行日志。"""
    policy_timing = result.get("policy_timing", {})
    model_timing = result.get("model_timing", {})
    server_timing = result.get("server_timing", {})
    client_timing = result.get("client_timing", {})
    fields = [
        ("client", client_timing.get("websocket_infer_ms")),
        ("server", server_timing.get("infer_ms")),
        ("ready", policy_timing.get("action_ready_ms")),
        ("tokenize", policy_timing.get("observation_tokenize_ms")),
        ("vlm", model_timing.get("vlm_prefix_forward_ms")),
        ("flow", model_timing.get("flow_denoise_ms")),
        ("steps", model_timing.get("flow_denoise_steps")),
        ("out", policy_timing.get("output_transform_ms")),
    ]
    parts = []
    for name, value in fields:
        if value is None:
            continue
        if name == "steps":
            parts.append(f"{name}={int(value)}")
        else:
            parts.append(f"{name}={float(value):.1f}ms")
    return " ".join(parts)


class TienkungDualHandsController(Node):
    """控制循环和异步推理调度。"""

    def __init__(self, args: Args) -> None:
        super().__init__("tienkung_dual_hands_openpi_async_client")
        self.args = args
        self.layout = build_robot_layout(args)
        self.ros_io = TienkungRosIO(self, args, self.layout)
        self.policy = AsyncPolicyProcess(remote_host=args.remote_host, remote_port=args.remote_port)
        self.policy.start()

        self.log_dir = Path(args.log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.execution_queue: deque[np.ndarray] = deque()
        self.last_action: np.ndarray | None = None
        self.inference_idx = 0
        self.last_request_time = 0.0
        self.submit_next_after_result = False

        self.create_timer(1.0 / args.control_hz, self.control_loop)
        self.get_logger().info(f"连接 policy server: {args.remote_host}:{args.remote_port}")
        self.get_logger().info(f"状态电机顺序: {self.layout.state_joint_ids}")
        self.get_logger().info(f"图像消息类型: {COMPRESSED_IMAGE_MSG.__name__}")
        self.get_logger().info(
            f"控制频率 {args.control_hz}Hz，模型动作频率 {args.policy_action_hz}Hz，"
            f"插值方式 {args.interpolation}，open_loop_horizon={args.open_loop_horizon}"
        )

    def destroy_node(self) -> bool:
        """节点销毁时关闭推理进程。"""
        self.policy.stop()
        return super().destroy_node()

    def control_loop(self) -> None:
        """控制循环，不等待推理返回。"""
        if not self.ros_io.is_ready():
            return

        self._poll_policy_result()
        self._maybe_submit_inference()
        self._publish_next_action()

    def _poll_policy_result(self) -> None:
        """取异步推理结果，并转成控制频率执行队列。"""
        result = self.policy.poll_latest()
        if result is None:
            return
        if not result.get("ok", False):
            self.get_logger().error(f"推理进程报错:\n{result.get('error')}")
            return

        plan = make_execution_plan(
            result["actions"],
            state_dim=STATE_DIM,
            max_action_chunk_len=self.args.max_action_chunk_len,
            policy_action_hz=self.args.policy_action_hz,
            control_hz=self.args.control_hz,
            interpolation=self.args.interpolation,
            lower_limits=self.layout.lower_limits,
            upper_limits=self.layout.upper_limits,
        )
        execute_len = min(len(plan), self.args.open_loop_horizon)
        self.execution_queue = deque(plan[:execute_len])
        if self.args.save_action_chunks:
            np.savez_compressed(
                self.log_dir / f"async_action_chunk_{self.inference_idx:06d}.npz",
                raw_actions=result["actions"],
                execution_plan=plan,
                policy_timing=result.get("policy_timing", {}),
                model_timing=result.get("model_timing", {}),
                server_timing=result.get("server_timing", {}),
                client_timing=result.get("client_timing", {}),
                state=self.ros_io.get_state(),
                timestamp=time.time(),
            )
        self.submit_next_after_result = True
        self.get_logger().info(
            f"收到动作 chunk #{self.inference_idx}: raw={result['actions'].shape}, "
            f"plan={plan.shape}, execute={execute_len} timing=[{_format_timing(result)}]"
        )
        self.inference_idx += 1

    def _maybe_submit_inference(self) -> None:
        """在执行队列即将耗尽时提交最新观测给推理进程。"""
        if self.policy.inflight:
            return
        should_submit_after_chunk = self.args.request_immediately_after_chunk and self.submit_next_after_result
        if not should_submit_after_chunk and len(self.execution_queue) > self.args.request_when_remaining_steps:
            return
        build_start = time.monotonic()
        observation = self.ros_io.build_observation(self.args.prompt)
        build_ms = (time.monotonic() - build_start) * 1000
        request_id = self.policy.submit_latest(observation)
        self.last_request_time = time.time()
        self.submit_next_after_result = False
        reason = "chunk_ready" if should_submit_after_chunk else f"remaining={len(self.execution_queue)}"
        self.get_logger().info(
            f"提交异步推理请求 #{request_id}: reason={reason}, build_observation={build_ms:.1f}ms",
            throttle_duration_sec=1.0,
        )

    def _publish_next_action(self) -> None:
        """按配置频率发布下一条动作；没有新动作时保持当前位置。"""
        if self.execution_queue:
            action = self.execution_queue.popleft()
            self.last_action = action
        elif self.last_action is not None:
            action = self.last_action
        else:
            action = self.ros_io.get_state()
            self.last_action = action
        self.ros_io.publish_action(action)


class TienkungDualHandsRtcController(Node):
    """Training-time RTC 调度版本：控制循环 + 独立推理进程 + hard-prefix 条件化。"""

    def __init__(self, args: Args) -> None:
        super().__init__("tienkung_dual_hands_openpi_rtc_client")
        self.args = args
        self.layout = build_robot_layout(args)
        self.ros_io = TienkungRosIO(self, args, self.layout)
        self.policy = AsyncPolicyProcess(remote_host=args.remote_host, remote_port=args.remote_port)
        self.policy.start()
        self.rtc = RtcChunker(
            horizon=args.max_action_chunk_len,
            min_horizon=args.rtc_min_horizon,
            delay_buffer_size=args.rtc_delay_buffer_size,
            initial_delay_steps=args.rtc_initial_delay_steps,
        )

        self.log_dir = Path(args.log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.last_action: np.ndarray | None = None
        self.inference_idx = 0

        self.create_timer(1.0 / args.control_hz, self.control_loop)
        self.get_logger().info(f"连接 policy server: {args.remote_host}:{args.remote_port}")
        self.get_logger().info(f"状态电机顺序: {self.layout.state_joint_ids}")
        self.get_logger().info(f"图像消息类型: {COMPRESSED_IMAGE_MSG.__name__}")
        self.get_logger().info(
            "Training-time RTC 调度模式：控制循环不等待推理，使用延迟估计和上一 chunk hard-prefix。"
            f"执行计划长度={args.max_action_chunk_len}，最小重规划间隔={args.rtc_min_horizon}，"
            "模型侧使用 rtc_prefix 条件化采样。"
        )

    def destroy_node(self) -> bool:
        """节点销毁时关闭推理进程。"""
        self.policy.stop()
        return super().destroy_node()

    def control_loop(self) -> None:
        """RTC 控制循环。"""
        if not self.ros_io.is_ready():
            return

        self._poll_policy_result()
        self._maybe_submit_inference()
        self._publish_next_action()
        self.rtc.record_control_step()

    def _poll_policy_result(self) -> None:
        """接收推理结果，插值后交给 RTC chunker 对齐 prefix。"""
        result = self.policy.poll_latest()
        if result is None:
            return
        if not result.get("ok", False):
            self.get_logger().error(f"推理进程报错:\n{result.get('error')}")
            return

        plan = make_execution_plan(
            result["actions"],
            state_dim=STATE_DIM,
            max_action_chunk_len=self.args.max_action_chunk_len,
            policy_action_hz=self.args.policy_action_hz,
            control_hz=self.args.control_hz,
            interpolation=self.args.interpolation,
            lower_limits=self.layout.lower_limits,
            upper_limits=self.layout.upper_limits,
        )
        merged_plan, observed_delay = self.rtc.accept_new_chunk(result["request_id"], plan)
        if self.args.save_action_chunks:
            np.savez_compressed(
                self.log_dir / f"rtc_action_chunk_{self.inference_idx:06d}.npz",
                raw_actions=result["actions"],
                execution_plan=plan,
                merged_plan=merged_plan,
                observed_delay_steps=observed_delay,
                policy_timing=result.get("policy_timing", {}),
                model_timing=result.get("model_timing", {}),
                server_timing=result.get("server_timing", {}),
                client_timing=result.get("client_timing", {}),
                state=self.ros_io.get_state(),
                timestamp=time.time(),
            )
        self.get_logger().info(
            f"RTC 收到 chunk #{self.inference_idx}: raw={result['actions'].shape}, "
            f"plan={plan.shape}, merged={merged_plan.shape}, delay_steps={observed_delay} "
            f"timing=[{_format_timing(result)}]"
        )
        self.inference_idx += 1

    def _maybe_submit_inference(self) -> None:
        """根据 RTC 条件提交下一次推理。"""
        if self.policy.inflight or not self.rtc.should_request():
            return
        build_start = time.monotonic()
        observation = self.ros_io.build_observation(self.args.prompt)
        build_ms = (time.monotonic() - build_start) * 1000
        prefix = self.rtc.make_action_prefix()
        if prefix is not None:
            action_prefix, delay = prefix
            observation["rtc_prefix"] = {
                "action_prefix": action_prefix,
                "delay": delay,
            }
        request_id = self.policy.submit_latest(observation)
        context = self.rtc.make_request_context(request_id)
        self.get_logger().info(
            f"RTC 提交推理请求 #{request_id}: s={context.executed_since_swap}, "
            f"d_est={context.delay_estimate_steps}, "
            f"prefix_steps={0 if prefix is None else prefix[1]}, suffix={context.previous_suffix.shape}, "
            f"build_observation={build_ms:.1f}ms",
            throttle_duration_sec=1.0,
        )

    def _publish_next_action(self) -> None:
        """发布 RTC 当前动作。"""
        fallback = self.last_action if self.last_action is not None else self.ros_io.get_state()
        action = self.rtc.consume_action(fallback)
        self.last_action = action
        self.ros_io.publish_action(action)


def main(args: Args) -> None:
    args = merge_config_file(args)
    os.environ.setdefault("ROS_DOMAIN_ID", args.ros_domain_id)
    rclpy.init()
    match args.controller_mode:
        case "async":
            node = TienkungDualHandsController(args)
        case "rtc":
            node = TienkungDualHandsRtcController(args)
        case _:
            raise ValueError("controller_mode 必须是 async 或 rtc。")
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    tyro.cli(main)
