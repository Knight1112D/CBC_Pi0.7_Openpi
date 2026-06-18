"""Training-time RTC 部署侧调度层。

这里维护异步推理请求、延迟估计和上一段 chunk 的 action prefix。模型侧使用
训练时 RTC 的 hard-prefix 条件化，部署侧只负责把旧 chunk 中推理期间会继续
执行的动作作为 ``rtc_prefix`` 传给 policy server，并在新 chunk 返回后跳过
已经执行过的 prefix。
"""

from __future__ import annotations

from collections import deque
import dataclasses

import numpy as np


@dataclasses.dataclass
class RtcRequestContext:
    """一次 RTC 推理请求的上下文。"""

    request_id: int
    start_step: int
    executed_since_swap: int
    delay_estimate_steps: int
    previous_suffix: np.ndarray


class RtcChunker:
    """维护 training-time RTC 状态并对齐 hard-prefix chunk。"""

    def __init__(
        self,
        *,
        horizon: int,
        min_horizon: int,
        delay_buffer_size: int,
        initial_delay_steps: int,
    ) -> None:
        if horizon <= 0:
            raise ValueError("horizon 必须大于 0。")
        if min_horizon <= 0:
            raise ValueError("min_horizon 必须大于 0。")
        if delay_buffer_size <= 0:
            raise ValueError("delay_buffer_size 必须大于 0。")
        self.horizon = int(horizon)
        self.min_horizon = min(int(min_horizon), self.horizon)
        self.delay_steps = deque([max(1, initial_delay_steps)], maxlen=delay_buffer_size)
        self.current_chunk: np.ndarray | None = None
        self.executed_since_swap = 0
        self.control_step = 0
        self.inflight_context: RtcRequestContext | None = None

    def has_chunk(self) -> bool:
        """当前是否有可执行 chunk。"""
        return self.current_chunk is not None and len(self.current_chunk) > 0

    def record_control_step(self) -> None:
        """记录控制循环步进。"""
        self.control_step += 1
        self.executed_since_swap += 1

    def should_request(self) -> bool:
        """是否应该启动下一次推理。"""
        if self.inflight_context is not None:
            return False
        if self.current_chunk is None:
            return True
        remaining = len(self.current_chunk) - self.executed_since_swap
        delay_estimate = max(self.delay_steps) if self.delay_steps else self.min_horizon
        return self.executed_since_swap >= self.min_horizon or remaining <= delay_estimate

    def make_request_context(self, request_id: int) -> RtcRequestContext:
        """创建推理请求上下文。"""
        delay_estimate = max(self.delay_steps) if self.delay_steps else self.min_horizon
        if self.current_chunk is None:
            suffix = np.empty((0, 0), dtype=np.float32)
        else:
            suffix = self.current_chunk[self.executed_since_swap :].copy()
        context = RtcRequestContext(
            request_id=request_id,
            start_step=self.control_step,
            executed_since_swap=self.executed_since_swap,
            delay_estimate_steps=max(delay_estimate, self.min_horizon),
            previous_suffix=suffix,
        )
        self.inflight_context = context
        return context

    def make_action_prefix(self) -> tuple[np.ndarray, int] | None:
        """生成传给模型的 hard-prefix 动作和估计延迟。

        返回的动作会补齐到 ``horizon``，但只有前 ``delay`` 步有效。
        """
        if self.current_chunk is None:
            return None
        suffix = self.current_chunk[self.executed_since_swap :].copy()
        if suffix.size == 0 or suffix.ndim != 2:
            return None

        delay_estimate = max(self.delay_steps) if self.delay_steps else self.min_horizon
        delay = min(max(0, int(delay_estimate)), len(suffix), self.horizon)
        if delay == 0:
            return None

        prefix = np.zeros((self.horizon, suffix.shape[1]), dtype=np.float32)
        copy_len = min(len(suffix), self.horizon)
        prefix[:copy_len] = suffix[:copy_len]
        return prefix, delay

    def consume_action(self, fallback_action: np.ndarray) -> np.ndarray:
        """返回当前步动作。"""
        if self.current_chunk is None or self.executed_since_swap >= len(self.current_chunk):
            return np.asarray(fallback_action, dtype=np.float32)
        return self.current_chunk[self.executed_since_swap].copy()

    def accept_new_chunk(self, request_id: int, new_chunk: np.ndarray) -> tuple[np.ndarray, int]:
        """接收推理结果并生成新的可执行 chunk。"""
        new_chunk = np.asarray(new_chunk, dtype=np.float32)
        if self.inflight_context is None or self.inflight_context.request_id != request_id:
            if self.current_chunk is None:
                self.current_chunk = new_chunk[: self.horizon].copy()
                self.executed_since_swap = 0
            return self.current_chunk.copy(), 0

        observed_delay = max(1, self.control_step - self.inflight_context.start_step)
        self.delay_steps.append(observed_delay)
        aligned = self._drop_executed_prefix(self.inflight_context.previous_suffix, new_chunk, observed_delay)
        self.current_chunk = aligned[: self.horizon].copy()
        self.executed_since_swap = 0
        self.inflight_context = None
        return self.current_chunk.copy(), observed_delay

    def _drop_executed_prefix(
        self, previous_suffix: np.ndarray, new_chunk: np.ndarray, observed_delay: int
    ) -> np.ndarray:
        """丢弃新 chunk 中已经由旧 chunk 执行过的 hard-prefix。"""
        new_chunk = np.asarray(new_chunk, dtype=np.float32)
        if previous_suffix.size == 0:
            return new_chunk
        if new_chunk.ndim != 2:
            return new_chunk
        skip = min(max(0, int(observed_delay)), len(new_chunk))
        return new_chunk[skip:]
