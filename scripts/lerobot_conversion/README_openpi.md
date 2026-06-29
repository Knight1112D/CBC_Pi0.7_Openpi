# OpenPI LeRobot v3 转 v2.1 工具

这个目录集成了 NVIDIA Isaac-GR00T 的 `convert_v3_to_v2.py`，用于把 LeRobot v3.0 数据转换成当前 OpenPI 训练代码可读取的 v2.1 布局。

## 本地 v3 数据转换

先准备一个 LeRobot v3 数据集目录，然后执行：

```bash
OPENPI_DIR=/path/to/openpi \
RAW_DATASET=/path/to/lerobot_v3_dataset \
LEROBOT_ROOT=/path/to/lerobot_root \
REPO_ID=owner/dataset_name \
./scripts/lerobot_conversion/convert_zhuji_v3_to_v2.sh
```

其中：

```text
OPENPI_DIR   OpenPI 项目根目录
RAW_DATASET  原始 LeRobot v3 数据集目录
LEROBOT_ROOT LeRobot 本地根目录，转换脚本会在这里创建 owner/dataset_name
REPO_ID      LeRobot repo id，例如 owner/dataset_name
```

脚本会先创建软链接，再调用 NVIDIA 转换脚本：

```bash
python convert_v3_to_v2.py --repo-id "${REPO_ID}" --root "${LEROBOT_ROOT}"
```

当前集成版默认使用 OpenPI 主环境：

```text
${OPENPI_DIR}/.venv/bin/python
```

`convert_v3_to_v2.py` 内部保留 NVIDIA 的转换逻辑，并补了少量本地兼容函数；因此不需要安装 NVIDIA/Isaac-GR00T，也不需要额外拉取完整新版 LeRobot 或 torch/CUDA 依赖。
