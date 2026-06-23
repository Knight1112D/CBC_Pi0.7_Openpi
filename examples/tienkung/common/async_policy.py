"""独立进程异步 policy 推理。"""

from __future__ import annotations

import multiprocessing as mp
import queue
import time
import traceback
from typing import Any

import numpy as np


def _policy_worker(
    request_queue: mp.Queue,
    response_queue: mp.Queue,
    stop_event: mp.Event,
    remote_host: str,
    remote_port: int,
) -> None:
    """在独立进程中连接 policy server 并执行推理。"""
    try:
        from openpi_client import websocket_client_policy

        policy_client = websocket_client_policy.WebsocketClientPolicy(remote_host, remote_port)
        while not stop_event.is_set():
            try:
                request = request_queue.get(timeout=0.05)
            except queue.Empty:
                continue
            if request is None:
                break

            request_id = request["request_id"]
            observation = request["observation"]
            try:
                infer_start = time.monotonic()
                result = policy_client.infer(observation)
                client_infer_ms = (time.monotonic() - infer_start) * 1000
                response_queue.put(
                    {
                        "request_id": request_id,
                        "ok": True,
                        "actions": np.asarray(result["actions"], dtype=np.float32),
                        "policy_timing": result.get("policy_timing", {}),
                        "model_timing": result.get("model_timing", {}),
                        "server_timing": result.get("server_timing", {}),
                        "client_timing": {
                            "websocket_infer_ms": client_infer_ms,
                        },
                    }
                )
            except Exception:
                response_queue.put(
                    {
                        "request_id": request_id,
                        "ok": False,
                        "error": traceback.format_exc(),
                    }
                )
    except Exception:
        response_queue.put({"request_id": -1, "ok": False, "error": traceback.format_exc()})


class AsyncPolicyProcess:
    """维护一个独立 policy 推理进程，并只保留最新请求。"""

    def __init__(self, *, remote_host: str, remote_port: int) -> None:
        self._ctx = mp.get_context("spawn")
        self._request_queue: mp.Queue = self._ctx.Queue(maxsize=1)
        self._response_queue: mp.Queue = self._ctx.Queue(maxsize=4)
        self._stop_event = self._ctx.Event()
        self._process = self._ctx.Process(
            target=_policy_worker,
            args=(self._request_queue, self._response_queue, self._stop_event, remote_host, remote_port),
            daemon=True,
        )
        self._next_request_id = 0
        self._latest_request_id = -1
        self._inflight = False

    @property
    def inflight(self) -> bool:
        """是否有一个请求正在推理。"""
        return self._inflight

    def start(self) -> None:
        """启动推理进程。"""
        self._process.start()

    def stop(self) -> None:
        """停止推理进程。"""
        self._stop_event.set()
        self._drop_stale_requests()
        with SuppressQueueFull():
            self._request_queue.put_nowait(None)
        self._process.join(timeout=2.0)
        if self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=2.0)

    def submit_latest(self, observation: dict[str, Any]) -> int:
        """提交最新观测；队列里旧的未开始请求会被丢弃。"""
        self._drop_stale_requests()
        request_id = self._next_request_id
        self._next_request_id += 1
        self._latest_request_id = request_id
        self._request_queue.put({"request_id": request_id, "observation": observation})
        self._inflight = True
        return request_id

    def poll_latest(self) -> dict[str, Any] | None:
        """取最新推理结果，丢弃旧结果。"""
        latest = None
        while True:
            try:
                latest = self._response_queue.get_nowait()
            except queue.Empty:
                break

        if latest is None:
            return None
        if latest.get("request_id", -1) >= self._latest_request_id:
            self._inflight = False
            return latest
        return None

    def _drop_stale_requests(self) -> None:
        while True:
            try:
                self._request_queue.get_nowait()
            except queue.Empty:
                break


class SuppressQueueFull:
    """忽略 multiprocessing queue full 的小型上下文。"""

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return exc_type is queue.Full
