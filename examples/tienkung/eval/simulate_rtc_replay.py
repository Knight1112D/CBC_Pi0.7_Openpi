#!/usr/bin/env python3
"""离线模拟 training-time RTC action chunk 调度。

该脚本只读取动作 chunk 文件或生成合成 chunk，不连接 ROS、不访问 policy server、
不发布机器人动作。用于快速检查固定延迟下 hard-prefix 跳过和执行序列连续性。
"""
# ruff: noqa: E402

from __future__ import annotations

import dataclasses
from pathlib import Path
import sys

import numpy as np
import tyro

OPENPI_ROOT = Path(__file__).resolve().parents[3]
if str(OPENPI_ROOT) not in sys.path:
    sys.path.insert(0, str(OPENPI_ROOT))

from examples.tienkung.rtc.rtc_chunker import RtcChunker


@dataclasses.dataclass
class Args:
    """RTC 离线 replay 参数。"""

    chunk_dir: Path | None = None
    output: Path = Path("/data/caobochun/openpi/eval_outputs/rtc_replay_smoke.npz")
    horizon: int = 32
    min_horizon: int = 8
    fixed_delay_steps: int = 4
    num_chunks: int = 12
    action_dim: int = 26


def _load_chunks(chunk_dir: Path, *, horizon: int, action_dim: int) -> list[np.ndarray]:
    """从部署日志 npz 中读取 action chunk。"""
    chunks: list[np.ndarray] = []
    for path in sorted(chunk_dir.glob("*.npz")):
        data = np.load(path)
        key = "execution_plan" if "execution_plan" in data else "raw_actions"
        action = np.asarray(data[key], dtype=np.float32)[:horizon, :action_dim]
        if action.ndim == 2 and len(action) > 0:
            chunks.append(action)
    if not chunks:
        raise ValueError(f"没有在 {chunk_dir} 找到可用的 npz action chunk。")
    return chunks


def _make_synthetic_chunks(args: Args) -> list[np.ndarray]:
    """生成连续但带轻微相位变化的合成 chunk。"""
    base_t = np.linspace(0.0, 1.0, args.horizon, dtype=np.float32)
    dims = np.linspace(0.5, 1.5, args.action_dim, dtype=np.float32)
    chunks = []
    for idx in range(args.num_chunks):
        phase = idx * args.min_horizon / max(1, args.horizon)
        chunk = np.sin((base_t[:, None] + phase) * dims[None, :] * np.pi).astype(np.float32)
        chunks.append(chunk)
    return chunks


def _smoothness(actions: np.ndarray) -> dict[str, float]:
    """计算执行序列的一阶和二阶差分指标。"""
    if len(actions) < 3:
        return {"mean_abs_delta": 0.0, "max_abs_delta": 0.0, "mean_abs_ddelta": 0.0, "max_abs_ddelta": 0.0}
    delta = np.diff(actions, axis=0)
    ddelta = np.diff(delta, axis=0)
    return {
        "mean_abs_delta": float(np.mean(np.abs(delta))),
        "max_abs_delta": float(np.max(np.abs(delta))),
        "mean_abs_ddelta": float(np.mean(np.abs(ddelta))),
        "max_abs_ddelta": float(np.max(np.abs(ddelta))),
    }


def run(args: Args) -> None:
    """执行离线 RTC replay 并保存指标。"""
    chunks = (
        _load_chunks(args.chunk_dir, horizon=args.horizon, action_dim=args.action_dim)
        if args.chunk_dir is not None
        else _make_synthetic_chunks(args)
    )
    chunker = RtcChunker(
        horizon=args.horizon,
        min_horizon=args.min_horizon,
        delay_buffer_size=8,
        initial_delay_steps=args.fixed_delay_steps,
    )

    executed: list[np.ndarray] = []
    observed_delays: list[int] = []
    fallback = np.zeros(args.action_dim, dtype=np.float32)
    request_id = 0
    chunk_iter = iter(chunks)

    while True:
        try:
            chunk = next(chunk_iter)
        except StopIteration:
            break

        prefix = chunker.make_action_prefix()
        if prefix is not None:
            action_prefix, delay = prefix
            assert action_prefix.shape == (args.horizon, args.action_dim)
            assert delay <= args.fixed_delay_steps

        context = chunker.make_request_context(request_id)
        for _ in range(args.fixed_delay_steps):
            executed.append(chunker.consume_action(fallback))
            chunker.record_control_step()
        _, observed_delay = chunker.accept_new_chunk(context.request_id, chunk)
        observed_delays.append(observed_delay)

        for _ in range(args.min_horizon):
            executed.append(chunker.consume_action(fallback))
            fallback = executed[-1]
            chunker.record_control_step()
        request_id += 1

    executed_arr = np.stack(executed, axis=0)
    metrics = _smoothness(executed_arr)
    metrics["num_actions"] = float(len(executed_arr))
    metrics["mean_delay_steps"] = float(np.mean(observed_delays)) if observed_delays else 0.0
    metrics["max_delay_steps"] = float(np.max(observed_delays)) if observed_delays else 0.0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output, executed_actions=executed_arr, observed_delays=np.asarray(observed_delays), **metrics
    )
    print(f"RTC replay 已保存: {args.output}")
    for key, value in metrics.items():
        print(f"{key}={value:.6f}")


if __name__ == "__main__":
    run(tyro.cli(Args))
