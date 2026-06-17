import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def make_tienkung_example() -> dict:
    """创建一条天工双手策略的随机输入样例。"""
    return {
        # 状态顺序与真实采集数据一致：左臂7 + 左手6 + 右臂7 + 右手6，共 26 维。
        "observation/state": np.random.rand(26),
        "observation/base_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/left_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/right_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "prompt": "Pick up the black box on the table with both hands, hold it briefly, then put the box down.",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class TienkungInputs(transforms.DataTransformFn):
    """将天工双手数据转换为 pi0/pi0.5 模型需要的输入格式。"""

    # 用于区分 pi0、pi0.5 等模型类型。
    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        # LeRobot 中图像可能是 float32 的 CHW 格式，这里统一转为 uint8 的 HWC 格式。
        base_image = _parse_image(data["observation/base_image"])
        left_image = _parse_image(data["observation/left_image"])
        right_image = _parse_image(data["observation/right_image"])

        # 下面这些键名是 pi0/pi0.5 模型固定读取的相机槽位。
        inputs = {
            "state": data["observation/state"],
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": left_image,
                "right_wrist_0_rgb": right_image,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_,
            },
        }

        # actions 只在训练时存在，后续 model transform 会自动补齐到模型动作维度。
        if "actions" in data:
            inputs["actions"] = data["actions"]

        # prompt 是语言指令，LeRobot 有时会以 bytes 形式读出。
        if "prompt" in data:
            if isinstance(data["prompt"], bytes):
                data["prompt"] = data["prompt"].decode("utf-8")
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class TienkungOutputs(transforms.DataTransformFn):
    """将模型输出裁剪回天工双手 26 维动作。"""

    def __call__(self, data: dict) -> dict:
        # 模型动作维度为 32，天工双手实际控制前 26 维。
        return {"actions": np.asarray(data["actions"][:, :26])}
