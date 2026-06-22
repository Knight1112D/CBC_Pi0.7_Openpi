"""Training-time RTC 调度层离线单元测试。"""

from __future__ import annotations

import numpy as np
from rtc_chunker import RtcChunker


def test_rtc_chunker_skips_executed_prefix_after_delay() -> None:
    """新 chunk 到达后，应跳过推理期间已经执行过的 hard-prefix。"""
    chunker = RtcChunker(
        horizon=8,
        min_horizon=3,
        delay_buffer_size=4,
        initial_delay_steps=2,
    )
    first = np.arange(16, dtype=np.float32).reshape(8, 2)
    second = np.full((8, 2), 100.0, dtype=np.float32)

    req0 = chunker.make_request_context(0)
    assert req0.previous_suffix.shape == (0, 0)
    chunker.record_control_step()
    merged0, delay0 = chunker.accept_new_chunk(0, first)
    assert delay0 == 1
    np.testing.assert_array_equal(merged0, first)

    for _ in range(3):
        chunker.record_control_step()
    req1 = chunker.make_request_context(1)
    assert req1.executed_since_swap == 3
    np.testing.assert_array_equal(req1.previous_suffix, first[3:])
    prefix = chunker.make_action_prefix()
    assert prefix is not None
    prefix_actions, prefix_steps = prefix
    assert prefix_steps == 2
    np.testing.assert_array_equal(prefix_actions[:prefix_steps], first[3:5])

    for _ in range(2):
        chunker.record_control_step()
    merged1, delay1 = chunker.accept_new_chunk(1, second)

    assert delay1 == 2
    np.testing.assert_array_equal(merged1, second[2:])


def test_rtc_chunker_ignores_stale_response_when_current_chunk_exists() -> None:
    """过期响应不应覆盖当前正在执行的 chunk。"""
    chunker = RtcChunker(
        horizon=4,
        min_horizon=2,
        delay_buffer_size=2,
        initial_delay_steps=1,
    )
    first = np.ones((4, 2), dtype=np.float32)
    stale = np.full((4, 2), 9.0, dtype=np.float32)

    chunker.accept_new_chunk(999, first)
    merged, delay = chunker.accept_new_chunk(123, stale)

    assert delay == 0
    np.testing.assert_array_equal(merged, first)
    np.testing.assert_array_equal(chunker.consume_action(np.zeros(2, dtype=np.float32)), first[0])
