# CBC_Pi0.7_Openpi 项目进度

## 项目目标

当前 Physical Intelligence 官方 `openpi` 主要开源了 `pi0`、`pi0-FAST` 和 `pi0.5` 的可用训练/推理路径；更后续的 RTC、pi0.6、RECAP、MEM、pi0.7 世界模型结合等完整工程实现尚未作为统一 openpi 项目开源。

本项目基于本地 `openpi` 代码镜像，尝试逐步复现和工程化这些能力，并适配 CBC 的天工机器人 VLA 训练与部署流程。

## 当前已完成

- 已在 A100_8 上部署 openpi 项目。
- 已完成天工机器人数据到 LeRobot 格式的训练链路。
- 已基于 pi0.5 训练出天工 VLA policy。
- 已准备本地代码镜像 `/data/Openpi_CBC_Upgrade/openpi`，用于快速检查和同步。
- 已规划本地/A100_8 双向同步 skill 与 GitHub 发布 skill。

## 后续方向

- pi0.5 高级语义与 knowledge insulation 思路的工程化实验。
- RTC/action chunking：异步推理、延迟估计、chunk suffix 融合和模型侧 guidance。
- pi0.6 RECAP/RL：success/failure、人类干预、value proxy、advantage 标签和条件化训练。
- MEM 相关记忆/任务上下文机制。
- pi0.7-style 世界模型、视觉子目标、交互式上下文和长程任务规划的结合。

## 记录约定

- 每个阶段记录 config、checkpoint、数据路径、GPU、命令、指标和结论。
- 不提交模型权重、训练数据、虚拟环境、日志或评估大文件。
