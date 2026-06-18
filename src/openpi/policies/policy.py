from collections.abc import Sequence
import logging
import pathlib
import time
from typing import Any, TypeAlias

import flax
import flax.traverse_util
import jax
import jax.numpy as jnp
import numpy as np
from openpi_client import base_policy as _base_policy
import torch
from typing_extensions import override

from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.shared import array_typing as at
from openpi.shared import nnx_utils

BasePolicy: TypeAlias = _base_policy.BasePolicy


class Policy(BasePolicy):
    def __init__(
        self,
        model: _model.BaseModel,
        *,
        rng: at.KeyArrayLike | None = None,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
        sample_kwargs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        pytorch_device: str = "cpu",
        is_pytorch: bool = False,
    ):
        """Initialize the Policy.

        Args:
            model: The model to use for action sampling.
            rng: Random number generator key for JAX models. Ignored for PyTorch models.
            transforms: Input data transformations to apply before inference.
            output_transforms: Output data transformations to apply after inference.
            sample_kwargs: Additional keyword arguments to pass to model.sample_actions.
            metadata: Additional metadata to store with the policy.
            pytorch_device: Device to use for PyTorch models (e.g., "cpu", "cuda:0").
                          Only relevant when is_pytorch=True.
            is_pytorch: Whether the model is a PyTorch model. If False, assumes JAX model.
        """
        self._model = model
        self._input_transform = _transforms.compose(transforms)
        self._output_transform = _transforms.compose(output_transforms)
        self._sample_kwargs = sample_kwargs or {}
        self._metadata = metadata or {}
        self._is_pytorch_model = is_pytorch
        self._pytorch_device = pytorch_device

        if self._is_pytorch_model:
            self._model = self._model.to(pytorch_device)
            self._model.eval()
            self._sample_actions = model.sample_actions
        else:
            # JAX model setup
            self._sample_actions = nnx_utils.module_jit(model.sample_actions)
            self._rng = rng or jax.random.key(0)

    @override
    def infer(self, obs: dict, *, noise: np.ndarray | None = None) -> dict:  # type: ignore[misc]
        rtc_guidance = obs.get("rtc_guidance")
        rtc_prefix = obs.get("rtc_prefix")
        obs_without_guidance = dict(obs)
        obs_without_guidance.pop("rtc_guidance", None)
        obs_without_guidance.pop("rtc_prefix", None)

        # Make a copy since transformations may modify the inputs in place.
        inputs = jax.tree.map(lambda x: x, obs_without_guidance)
        inputs = self._input_transform(inputs)

        rtc_sample_kwargs = self._prepare_rtc_guidance(obs_without_guidance, rtc_guidance)
        rtc_sample_kwargs.update(self._prepare_rtc_prefix(obs_without_guidance, rtc_prefix))

        if not self._is_pytorch_model:
            # Make a batch and convert to jax.Array.
            inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
            self._rng, sample_rng_or_pytorch_device = jax.random.split(self._rng)
        else:
            # Convert inputs to PyTorch tensors and move to correct device
            inputs = jax.tree.map(lambda x: torch.from_numpy(np.array(x)).to(self._pytorch_device)[None, ...], inputs)
            sample_rng_or_pytorch_device = self._pytorch_device

        # Prepare kwargs for sample_actions
        sample_kwargs = dict(self._sample_kwargs)
        sample_kwargs.update(rtc_sample_kwargs)
        if noise is not None:
            noise = torch.from_numpy(noise).to(self._pytorch_device) if self._is_pytorch_model else jnp.asarray(noise)

            if noise.ndim == 2:  # If noise is (action_horizon, action_dim), add batch dimension
                noise = noise[None, ...]  # Make it (1, action_horizon, action_dim)
            sample_kwargs["noise"] = noise

        observation = _model.Observation.from_dict(inputs)
        start_time = time.monotonic()
        outputs = {
            "state": inputs["state"],
            "actions": self._sample_actions(sample_rng_or_pytorch_device, observation, **sample_kwargs),
        }
        model_time = time.monotonic() - start_time
        if self._is_pytorch_model:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...].detach().cpu()), outputs)
        else:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)

        outputs = self._output_transform(outputs)
        outputs["policy_timing"] = {
            "infer_ms": model_time * 1000,
        }
        return outputs

    def _prepare_rtc_prefix(self, obs: dict, rtc_prefix: dict | None) -> dict[str, Any]:
        """把训练时 RTC hard-prefix 动作转换到模型采样空间。"""
        if rtc_prefix is None:
            return {}
        if not self._is_pytorch_model:
            raise ValueError("当前 RTC hard-prefix inference 只实现了 PyTorch 模型路径。")

        action_prefix = np.asarray(
            rtc_prefix.get("action_prefix", rtc_prefix.get("target_actions")), dtype=np.float32
        )
        if action_prefix.ndim != 2:
            raise ValueError(f"rtc_prefix action_prefix 必须是二维数组，当前 shape={action_prefix.shape}")

        prefix_obs = jax.tree.map(lambda x: x, obs)
        prefix_obs["actions"] = action_prefix
        transformed = self._input_transform(prefix_obs)
        model_prefix = np.asarray(transformed["actions"], dtype=np.float32)

        model_config = getattr(self._model, "config", None)
        model_horizon = int(getattr(model_config, "action_horizon", model_prefix.shape[0]))
        model_action_dim = int(getattr(model_config, "action_dim", model_prefix.shape[-1]))
        model_prefix = self._resize_rtc_array(
            model_prefix,
            target_horizon=model_horizon,
            target_dim=model_action_dim,
            fill_value=0.0,
        )

        delay = int(rtc_prefix.get("delay", rtc_prefix.get("prefix_steps", 0)))
        return {
            "rtc_prefix": {
                "action_prefix": torch.from_numpy(model_prefix[None, ...]).to(self._pytorch_device),
                "delay": delay,
            }
        }

    def _prepare_rtc_guidance(self, obs: dict, rtc_guidance: dict | None) -> dict[str, Any]:
        """把部署侧 RTC 目标动作转换到模型采样空间。"""
        if rtc_guidance is None:
            return {}
        if not self._is_pytorch_model:
            raise ValueError("当前 RTC guided inference 只实现了 PyTorch 模型路径。")

        target_actions = np.asarray(rtc_guidance["target_actions"], dtype=np.float32)
        weights = np.asarray(rtc_guidance["weights"], dtype=np.float32)
        if target_actions.ndim != 2:
            raise ValueError(f"rtc target_actions 必须是二维数组，当前 shape={target_actions.shape}")
        if weights.ndim == 1:
            weights = np.repeat(weights[:, None], target_actions.shape[-1], axis=-1)
        if weights.shape != target_actions.shape:
            raise ValueError(f"rtc weights shape {weights.shape} 必须等于 target_actions shape {target_actions.shape}")

        guidance_obs = jax.tree.map(lambda x: x, obs)
        guidance_obs["actions"] = target_actions
        transformed = self._input_transform(guidance_obs)
        model_targets = np.asarray(transformed["actions"], dtype=np.float32)

        model_config = getattr(self._model, "config", None)
        model_horizon = int(getattr(model_config, "action_horizon", model_targets.shape[0]))
        model_action_dim = int(getattr(model_config, "action_dim", model_targets.shape[-1]))
        model_targets = self._resize_rtc_array(
            model_targets,
            target_horizon=model_horizon,
            target_dim=model_action_dim,
            fill_value=0.0,
        )
        weights = self._resize_rtc_array(
            weights,
            target_horizon=model_horizon,
            target_dim=model_action_dim,
            fill_value=0.0,
        )

        return {
            "rtc_guidance": {
                "target_actions": torch.from_numpy(model_targets[None, ...]).to(self._pytorch_device),
                "weights": torch.from_numpy(weights[None, ...]).to(self._pytorch_device),
                "beta": float(rtc_guidance.get("beta", 0.0)),
                "eps": float(rtc_guidance.get("eps", 1e-4)),
            }
        }

    @staticmethod
    def _resize_rtc_array(
        value: np.ndarray,
        *,
        target_horizon: int,
        target_dim: int,
        fill_value: float,
    ) -> np.ndarray:
        """把 RTC guidance 数组补齐或截断到模型动作形状。"""
        value = np.asarray(value, dtype=np.float32)
        resized = np.full((target_horizon, target_dim), fill_value, dtype=np.float32)
        copy_horizon = min(value.shape[0], target_horizon)
        copy_dim = min(value.shape[-1], target_dim)
        resized[:copy_horizon, :copy_dim] = value[:copy_horizon, :copy_dim]
        return resized

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata


class PolicyRecorder(_base_policy.BasePolicy):
    """Records the policy's behavior to disk."""

    def __init__(self, policy: _base_policy.BasePolicy, record_dir: str):
        self._policy = policy

        logging.info(f"Dumping policy records to: {record_dir}")
        self._record_dir = pathlib.Path(record_dir)
        self._record_dir.mkdir(parents=True, exist_ok=True)
        self._record_step = 0

    @override
    def infer(self, obs: dict) -> dict:  # type: ignore[misc]
        results = self._policy.infer(obs)

        data = {"inputs": obs, "outputs": results}
        data = flax.traverse_util.flatten_dict(data, sep="/")

        output_path = self._record_dir / f"step_{self._record_step}"
        self._record_step += 1

        np.save(output_path, np.asarray(data))
        return results
