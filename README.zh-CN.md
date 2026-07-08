# CBC_Pi0.7_Openpi

语言：[English](README.md) | 简体中文

## 概览

`CBC_Pi0.7_Openpi` 是一个非官方的个人研究和工程尝试，目标是为人形机器人 VLA 研究构建一个更完整的 OpenPI 项目。

本项目基于 Physical Intelligence 开源的 [`openpi`](https://github.com/Physical-Intelligence/openpi) 仓库构建。感谢 Physical Intelligence 团队发布 OpenPI 代码库、模型定义、训练和推理示例、远程 policy-server 工作流，以及公开的 `pi0`、`pi0-FAST` 和 `pi0.5` 资产。没有这些基础，本仓库不可能存在。

这是一个独立项目，并且对上游 OpenPI 工作保持充分尊重。它的目的，是复现、集成并扩展公开材料中讨论过、但尚未作为完整 OpenPI 实现发布的 OpenPI 风格工程组件。

本项目也感谢 OpenTau 项目在后续 OpenPI 风格训练方向上的公开工程参考价值，包括 memory-conditioned policies、value/advantage labeling 和 policy-family organization。本仓库会把相关思路放在显式 OpenPI 兼容开关之后适配，默认上游训练路径保持不变。

原始上游 OpenPI README 保留在这里：

- [UPSTREAM_OPENPI_README.md](UPSTREAM_OPENPI_README.md)

## 上游致谢

本仓库继承并构建在以下上游 OpenPI 贡献之上：

- `pi0`，一个基于 flow 的视觉-语言-动作模型。
- `pi0-FAST`，一个基于 FAST 动作 tokenizer 的自回归 VLA。
- `pi0.5`，包括公开发布的 flow-matching 训练和推理路径。
- 用于将 OpenPI 适配到新机器人本体的 policy transform、LeRobot 数据配置、训练配置、归一化统计和 policy-server 模式。
- JAX 和 PyTorch 模型/训练结构。
- 远程 websocket policy 服务。
- 面向 DROID、ALOHA、LIBERO、UR5 以及相关工作流的参考示例。

主要上游参考：

- OpenPI GitHub：<https://github.com/Physical-Intelligence/openpi>
- Physical Intelligence：<https://www.physicalintelligence.company/>
- pi0：<https://www.physicalintelligence.company/blog/pi0>
- pi0-FAST：<https://www.physicalintelligence.company/research/fast>
- pi0.5：<https://www.physicalintelligence.company/blog/pi05>
- Knowledge Insulation：<https://www.physicalintelligence.company/research/knowledge_insulation>

## 项目目标

OpenPI 当前为 `pi0`、`pi0-FAST` 和 `pi0.5` 提供了强大的开源基础。公开材料中讨论的一些后续能力，目前还没有作为完整的开源 OpenPI 工程栈提供。

本项目旨在将这些公开想法转化为一个更完整、可运行的 OpenPI 风格工程栈：

- OpenPI 代码库内的人形机器人 VLA 训练、推理和评估。
- `pi0.5` 风格的更高层语义泛化，以及 KI（Knowledge Insulation / Knowledge Isolation）实验。
- 面向远程 VLA 推理的 RTC / 实时动作分块。
- `pi0.6` 风格的 RECAP、RL、人类干预标签、价值代理和 advantage-conditioned policy。
- MEM 风格的记忆、任务上下文和恢复机制。
- 受 `pi0.7` 启发的世界模型、视觉子目标、交互式上下文和长时程规划想法。

目标机器人平台有意不固定。仓库首先围绕补全 OpenPI 工程栈组织，然后再在人形机器人平台上验证。

## 相关项目

如需一个具体的人形机器人 VLA 数据采集流水线，请参见我的另一个项目：

- [Knight1112D/Tienkung_vla_collect_data](https://github.com/Knight1112D/Tienkung_vla_collect_data)

该仓库是一个数据采集参考。本仓库专注于构建 OpenPI 侧的模型、数据、训练、推理、实时执行和更高层能力栈。

## 路线图

### 1. 人形机器人 VLA 部署

- [ ] 在 OpenPI 数据和 policy transform 栈中定义人形机器人观测/动作接口。
- [ ] 添加用于本地验证的 dry-run 和 replay-mode policy clients。
- [ ] 添加针对 policy-server 延迟、动作 chunk 和动作归一化的离线检查。
- [ ] 添加机器人侧安全约束，包括关节顺序、限制、频率、插值和停止行为。
- [ ] 将机器人特定 bridge 与核心 OpenPI 复现工作隔离。

### 2. pi0.5 风格语义 / KI

- [ ] 对比公开 `pi0.5` 实现与更高层语义泛化的公开描述。
- [ ] 在人形机器人操作数据集上测试 prompt 敏感性、任务上下文和视觉语义变化。
- [ ] 探索用于语义泛化的多任务或多场景混合。
- [ ] 在 OpenPI PyTorch pi0.5 路径上添加 KI-lite 实验：用 FAST action-token CE 训练 VLM 的动作表示，用 flow loss 训练 action expert，并阻断 flow/action-expert loss 回传到 VLM backbone。
- [ ] 将 `stop-gradient only` 和真正的 KI-lite 区分开；只有 FAST CE 和 flow loss 都进入实验时，才称为 KI。
- [ ] 将 KI 与 RTC 作为可组合开关测试，同时不破坏 training-time delay sampling、hard-prefix sampling 或 masked postfix loss。
- [ ] 尝试受控冻结、adapter、数据加权和 KI 启发的训练配方。

### 3. RTC / 实时分块

RTC 是当前用于提升远程推理稳定性的优先方向。在异步人形机器人 policy-server 设置中，机器人控制循环会以固定频率持续发布动作，同时模型在后台生成下一个动作 chunk。已经执行过的动作，或者会在下一个 chunk 到达前被执行的动作，会成为新 chunk 的 hard prefix。RTC 的目标是在存在非零推理延迟时，让新生成的 chunk 与这些已经确定的动作保持连续。

当前实现已经从早期的 inference-time VJP/Jacobian-guidance 想法转向 training-time RTC。训练会随机模拟推理延迟，并将 prefix tokens 作为干净动作条件输入。部署时只传递 `rtc_prefix`；flow-matching 采样循环会在每个去噪步骤 hard-overwrite prefix，不再执行额外的 inference-time backpropagation。详细设计说明见 [`docs/cbc/training_time_rtc.md`](docs/cbc/training_time_rtc.md)。

已完成：

- [x] 添加 `RTCTrainingConfig`，并接入 `pi05_tienkung_finetune_rtc`。
- [x] 在 PyTorch pi0.5 flow-matching 路径中添加 training-time delay sampling：prefix tokens 使用干净动作，postfix tokens 保持 noisy，并且 loss 只在有效 postfix steps 上计算。
- [x] 在 `PI0Pytorch.sample_actions()` 中添加 `rtc_prefix` hard-prefix 采样支持，在当前 OpenPI flow 约定下 prefix tokens 使用干净 endpoint。
- [x] 更新 policy 接口和异步 client 示例，使部署根据观测到的 delay 跳过已经执行的 prefix steps，并只执行新的 postfix。
- [x] 添加 `examples/tienkung/rtc_chunker_test.py` 和 `examples/tienkung/simulate_rtc_replay.py`，用于 chunk suffix 合并和 hard-prefix replay smoke tests。
- [x] 验证 real-batch forward/backward、hard-prefix `sample_actions`、2-step training smoke、example compilation 和 replay smoke。

后续实验：

- [ ] 实现系统性的离线延迟模拟和 replay 评估。
- [ ] 对比 async baseline、fixed-horizon RTC、delay-adaptive RTC、suffix soft blending 和 training-time RTC hard-prefix variants。
- [ ] 添加延迟分布选项：`uniform`、`exp` 和 `empirical`。
- [ ] 从真实推理延迟日志中构建经验延迟直方图，并在 `d=0..25, s=25` 下运行离线 delay sweep。
- [ ] 报告 latency、replan frequency、action smoothness、prefix/postfix discontinuity 和 rollout stability。
- [ ] 如果重新考虑 model-side inference-time guidance，将其保留为实验开关和对比 baseline，而不是默认部署路径。

### 4. pi0.6 风格 RECAP / RL

- [x] 为 success、failure、evaluation episodes 和 human intervention 定义通用 episode metadata。
- [x] 为 `advantage_indicator`、`use_advantage` 和 `is_human_intervention` 生成 sidecar labels。
- [x] 添加 code-only sidecar 生成脚本，用于离线 advantage/RL-token labels。
- [x] 在不修改原始数据集的情况下，将 RECAP sidecar fields 合并进 dataloader。
- [x] 为 PyTorch flow-matching loss 添加 RL-token 样本级加权。
- [x] 添加用于 1-2 step smoke tests 的小型 debug config。
- [ ] 对比标准 SFT 和 RECAP 风格 fine-tuning。

### 5. MEM / 记忆

- [x] 调研公开 MEM 相关材料和可复现实现线索。
- [x] 为 recent history、task state、failures 和 recovery hints 设计可选 memory/context fields。
- [x] 保持 memory inputs 可选，使现有 OpenPI policies 仍然兼容。
- [x] 添加可选 sidecar memory-to-prompt augmentation，用于 debug runs。
- [ ] 评估有无 memory context 时的行为差异。

### 6. 受 pi0.7 启发的世界模型

- [ ] 总结公开 pi0.7 paper/blog 概念，并将它们与项目假设区分开。
- [ ] 设计世界模型风格的中间表示或视觉子目标输入。
- [ ] 探索 keyframes、subgoals 和 language decomposition 作为 policy context。
- [ ] 构建长时程人形机器人任务示例，并追踪 success、failure modes 和 recovery。

## 仓库策略

- 不包含模型权重。
- 不包含私有数据集。
- 不包含虚拟环境。
- 不包含机器特定日志或实验 artifacts。
- 保持上游 OpenPI attribution 可见。
- 清楚标注非官方复现代码和推测性工程实验。

## 许可证

本仓库保留上游 OpenPI 许可证文件和第三方声明。新增代码和文档应保持与上游许可约束兼容。任何第三方复现工作或受论文启发的实现，都应在相关文件中包含来源说明。
