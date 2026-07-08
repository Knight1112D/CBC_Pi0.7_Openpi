"""CBC 训练扩展：RECAP/MEM/RL token 与知识隔离的兼容工具。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import dataclasses
import json
import pathlib
from typing import Any

import numpy as np
import torch


CBC_PREFIX = "cbc_"
ADVANTAGE_KEY = "cbc_advantage_indicator"
USE_ADVANTAGE_KEY = "cbc_use_advantage"
INTERVENTION_KEY = "cbc_is_human_intervention"
RL_WEIGHT_KEY = "cbc_rl_token_weight"
MEMORY_KEY = "cbc_memory"
NEXT_MEMORY_KEY = "cbc_next_memory"

NUMERIC_METADATA_KEYS = (
    ADVANTAGE_KEY,
    USE_ADVANTAGE_KEY,
    INTERVENTION_KEY,
    RL_WEIGHT_KEY,
)


@dataclasses.dataclass(frozen=True)
class SidecarConfig:
    """离线 sidecar 标签配置，默认关闭以保持上游训练行为。"""

    path: str | None = None
    enabled: bool = False
    append_memory_to_prompt: bool = False

    def __post_init__(self) -> None:
        if self.enabled and not self.path:
            raise ValueError("启用 sidecar 时必须设置 path。")


@dataclasses.dataclass(frozen=True)
class RLTokenConfig:
    """按 RECAP/RL 标签对 loss 做样本级加权。"""

    enabled: bool = False
    positive_weight: float = 1.0
    negative_weight: float = 1.0
    intervention_weight: float = 1.0
    min_weight: float = 0.0
    max_weight: float = 10.0

    def __post_init__(self) -> None:
        if self.min_weight < 0 or self.max_weight < self.min_weight:
            raise ValueError("RL token 权重范围不合法。")


@dataclasses.dataclass(frozen=True)
class KnowledgeInsulationConfig:
    """PyTorch pi0/pi0.5 的轻量知识隔离训练开关。"""

    enabled: bool = False
    freeze_vlm: bool = True
    train_action_expert: bool = True
    train_action_projections: bool = True


def is_cbc_key(key: str) -> bool:
    """判断字段是否属于 CBC 扩展命名空间。"""

    return key.startswith(CBC_PREFIX)


def read_sidecar(path: str | pathlib.Path) -> dict[tuple[int | None, int | None], dict[str, Any]]:
    """读取 JSON/JSONL sidecar，并按 episode/frame 建索引。"""

    sidecar_path = pathlib.Path(path)
    if not sidecar_path.exists():
        raise FileNotFoundError(f"sidecar 文件不存在：{sidecar_path}")

    if sidecar_path.suffix == ".jsonl":
        records = [json.loads(line) for line in sidecar_path.read_text().splitlines() if line.strip()]
    else:
        payload = json.loads(sidecar_path.read_text())
        records = payload["records"] if isinstance(payload, dict) and "records" in payload else payload
    if not isinstance(records, list):
        raise ValueError("sidecar 必须是记录列表，或包含 records 列表的 JSON 对象。")

    index: dict[tuple[int | None, int | None], dict[str, Any]] = {}
    for record in records:
        if not isinstance(record, Mapping):
            continue
        episode = _optional_int(record.get("episode_index"))
        frame = _optional_int(record.get("frame_index", record.get("index")))
        index[(episode, frame)] = dict(record)
    return index


class SidecarDataset:
    """在不修改原 LeRobot 数据集的情况下合并 CBC sidecar 字段。"""

    def __init__(
        self,
        dataset,
        sidecar: Mapping[tuple[int | None, int | None], Mapping[str, Any]],
        *,
        include_text: bool = False,
    ):
        self._dataset = dataset
        self._sidecar = sidecar
        self._include_text = include_text

    def __getitem__(self, index):
        item = dict(self._dataset[index])
        episode = _optional_int(item.get("episode_index"))
        frame = _optional_int(item.get("frame_index", item.get("index", index.__index__())))
        record = self._sidecar.get((episode, frame)) or self._sidecar.get((episode, None)) or {}
        item.update(_standardize_record(record, include_text=self._include_text))
        return item

    def __len__(self) -> int:
        return len(self._dataset)


class AppendMemoryToPrompt:
    """把可选 memory 字段追加到 prompt，用于 MEM/context smoke。"""

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        memory = data.pop(MEMORY_KEY, "")
        if isinstance(memory, np.ndarray):
            memory = memory.item()
        if memory:
            prompt = data.get("prompt", "")
            if isinstance(prompt, np.ndarray):
                prompt = prompt.item()
            data["prompt"] = f"{prompt}\nMemory: {memory}"
        data.pop(NEXT_MEMORY_KEY, None)
        return data


def make_rl_token_weights(metadata: Mapping[str, torch.Tensor] | None, config: RLTokenConfig) -> torch.Tensor | None:
    """根据 sidecar metadata 生成 batch 级 loss 权重。"""

    if not config.enabled or metadata is None:
        return None
    advantage = metadata.get(ADVANTAGE_KEY)
    intervention = metadata.get(INTERVENTION_KEY)
    explicit = metadata.get(RL_WEIGHT_KEY)
    if advantage is None and intervention is None and explicit is None:
        return None

    base = explicit.to(dtype=torch.float32) if explicit is not None else torch.ones_like(advantage, dtype=torch.float32)
    if advantage is not None:
        advantage = advantage.to(dtype=torch.float32)
        base = torch.where(advantage > 0, base * config.positive_weight, base * config.negative_weight)
    if intervention is not None:
        base = torch.where(intervention.to(dtype=torch.bool), base * config.intervention_weight, base)
    return base.clamp(min=config.min_weight, max=config.max_weight)


def apply_loss_weights(losses: torch.Tensor, weights: torch.Tensor | None) -> torch.Tensor:
    """把 batch 级权重广播到 flow-matching loss 张量。"""

    if weights is None:
        return losses
    while weights.ndim < losses.ndim:
        weights = weights[..., None]
    return losses * weights.to(device=losses.device, dtype=losses.dtype)


def apply_knowledge_insulation(model: torch.nn.Module, config: KnowledgeInsulationConfig) -> dict[str, int]:
    """按模块名冻结 VLM/动作专家参数，返回参数统计。"""

    if not config.enabled:
        return {"trainable": sum(p.numel() for p in model.parameters() if p.requires_grad), "frozen": 0}

    trainable = 0
    frozen = 0
    for name, param in model.named_parameters():
        allow_train = False
        if config.train_action_expert and ("gemma_expert" in name or "time_mlp" in name):
            allow_train = True
        if config.train_action_projections and ("action_in_proj" in name or "action_out_proj" in name):
            allow_train = True
        if not config.freeze_vlm:
            allow_train = True
        param.requires_grad_(allow_train)
        if allow_train:
            trainable += param.numel()
        else:
            frozen += param.numel()
    return {"trainable": trainable, "frozen": frozen}


def _standardize_record(record: Mapping[str, Any], *, include_text: bool = True) -> dict[str, Any]:
    output: dict[str, Any] = {}
    output[ADVANTAGE_KEY] = float(record.get("advantage_indicator", record.get("advantage", 0.0)))
    output[USE_ADVANTAGE_KEY] = float(record.get("use_advantage", bool(output[ADVANTAGE_KEY])))
    output[INTERVENTION_KEY] = float(record.get("is_human_intervention", record.get("human_intervention", 0.0)))
    output[RL_WEIGHT_KEY] = float(record.get("rl_token_weight", 1.0))
    if include_text and "memory" in record:
        output[MEMORY_KEY] = str(record["memory"])
    if include_text and "next_memory" in record:
        output[NEXT_MEMORY_KEY] = str(record["next_memory"])
    return output


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(np.asarray(value).item())
    except (TypeError, ValueError):
        return None


def write_jsonl(records: Sequence[Mapping[str, Any]], path: str | pathlib.Path) -> None:
    """写出 JSONL sidecar，便于后续人工补标或脚本改写。"""

    sidecar_path = pathlib.Path(path)
    sidecar_path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(json.dumps(dict(record), ensure_ascii=False, sort_keys=True) for record in records)
    sidecar_path.write_text(text + ("\n" if text else ""))
