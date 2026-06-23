"""动作 chunk 插值和执行计划。"""

from __future__ import annotations

import numpy as np


def resample_action_chunk(actions: np.ndarray, src_hz: float, dst_hz: float, *, method: str = "linear") -> np.ndarray:
    """把模型动作频率插值到控制频率。"""
    actions = np.asarray(actions, dtype=np.float32)
    if actions.ndim != 2:
        raise ValueError(f"actions 必须是二维数组，当前 shape={actions.shape}")
    if len(actions) < 2 or abs(src_hz - dst_hz) < 1e-6:
        return actions.copy()

    t_src = np.arange(len(actions), dtype=np.float32) / float(src_hz)
    t_end = float(t_src[-1])
    n_new = int(np.round(t_end * dst_hz)) + 1
    t_new = np.linspace(0.0, t_end, n_new, dtype=np.float32)

    if method == "linear":
        out = np.empty((n_new, actions.shape[1]), dtype=np.float32)
        for dim in range(actions.shape[1]):
            out[:, dim] = np.interp(t_new, t_src, actions[:, dim]).astype(np.float32)
        return out

    if method == "cubic":
        try:
            from scipy.interpolate import CubicSpline
        except ImportError as exc:
            raise RuntimeError("使用 cubic 插值需要安装 scipy。") from exc
        out = np.empty((n_new, actions.shape[1]), dtype=np.float32)
        for dim in range(actions.shape[1]):
            out[:, dim] = CubicSpline(t_src, actions[:, dim], bc_type="natural")(t_new).astype(np.float32)
        return out

    if method == "none":
        return actions.copy()

    raise ValueError(f"未知插值方式: {method}")


def make_execution_plan(
    action_chunk: np.ndarray,
    *,
    state_dim: int,
    max_action_chunk_len: int,
    policy_action_hz: float,
    control_hz: float,
    interpolation: str,
    lower_limits: np.ndarray,
    upper_limits: np.ndarray,
) -> np.ndarray:
    """把 policy 输出整理成控制循环可执行的轨迹。"""
    actions = np.asarray(action_chunk, dtype=np.float32)
    if actions.ndim == 1:
        actions = actions[None, :]
    actions = actions[:max_action_chunk_len, :state_dim]
    actions = np.clip(actions, lower_limits[None, :], upper_limits[None, :])
    actions = resample_action_chunk(actions, policy_action_hz, control_hz, method=interpolation)
    return np.clip(actions, lower_limits[None, :], upper_limits[None, :])
