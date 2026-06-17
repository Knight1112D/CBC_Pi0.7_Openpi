"""Real-Time Chunking 调度层实现。

这里实现的是黑盒 policy server 版本的 RTC 调度：动态 horizon、延迟估计、
保留上一段未执行动作和软融合。论文算法里的 GuidedInference 需要访问 flow
policy 的 denoising 函数和 VJP，自带 websocket server 只暴露最终 action chunk，
所以这里不能实现真正的 soft-mask autodiff guidance。
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
    """维护 RTC 状态并把新 chunk 与旧 suffix 软融合。"""

    def __init__(
        self,
        *,
        horizon: int,
        min_horizon: int,
        delay_buffer_size: int,
        initial_delay_steps: int,
        blend_steps: int,
        preserve_weight: float,
    ) -> None:
        self.horizon = horizon
        self.min_horizon = min_horizon
        self.blend_steps = blend_steps
        self.preserve_weight = float(np.clip(preserve_weight, 0.0, 1.0))
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
        return self.executed_since_swap >= self.min_horizon

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

    def consume_action(self, fallback_action: np.ndarray) -> np.ndarray:
        """返回当前步动作。"""
        if self.current_chunk is None or self.executed_since_swap >= len(self.current_chunk):
            return fallback_action
        return self.current_chunk[self.executed_since_swap]

    def accept_new_chunk(self, request_id: int, new_chunk: np.ndarray) -> tuple[np.ndarray, int]:
        """接收推理结果并生成新的可执行 chunk。"""
        if self.inflight_context is None or self.inflight_context.request_id != request_id:
            self.current_chunk = new_chunk[: self.horizon].copy()
            self.executed_since_swap = 0
            return self.current_chunk, 0

        observed_delay = max(1, self.control_step - self.inflight_context.start_step)
        self.delay_steps.append(observed_delay)
        merged = self._soft_merge(self.inflight_context.previous_suffix, new_chunk, observed_delay)
        self.current_chunk = merged[: self.horizon].copy()
        self.executed_since_swap = 0
        self.inflight_context = None
        return self.current_chunk, observed_delay

    def _soft_merge(self, previous_suffix: np.ndarray, new_chunk: np.ndarray, observed_delay: int) -> np.ndarray:
        """用上一 chunk 的 suffix 软约束新 chunk 的前段。"""
        new_chunk = np.asarray(new_chunk, dtype=np.float32)
        if previous_suffix.size == 0:
            return new_chunk

        suffix = np.asarray(previous_suffix, dtype=np.float32)
        if suffix.ndim != 2 or new_chunk.ndim != 2:
            return new_chunk
        if suffix.shape[1] != new_chunk.shape[1]:
            return new_chunk

        # 推理期间旧 suffix 已经继续消耗了一部分，先对齐到当前时刻。
        suffix = suffix[min(observed_delay, len(suffix)) :]
        if len(suffix) == 0:
            return new_chunk

        merged = new_chunk.copy()
        preserve = min(len(suffix), len(merged), self.inflight_context.delay_estimate_steps)
        if preserve > 0:
            merged[:preserve] = (
                self.preserve_weight * suffix[:preserve] + (1.0 - self.preserve_weight) * merged[:preserve]
            )

        blend_start = preserve
        blend_end = min(len(suffix), len(merged), blend_start + self.blend_steps)
        if blend_end > blend_start:
            for idx in range(blend_start, blend_end):
                alpha = (idx - blend_start + 1) / (blend_end - blend_start + 1)
                old_weight = self.preserve_weight * (1.0 - alpha)
                merged[idx] = old_weight * suffix[idx] + (1.0 - old_weight) * merged[idx]

        return merged
