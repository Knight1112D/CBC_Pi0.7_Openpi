# CBC_Pi0.7_Openpi

## 项目说明

本项目基于 Physical Intelligence 团队开源的 [`openpi`](https://github.com/Physical-Intelligence/openpi) 项目开展二次研究和工程复现。首先感谢 Physical Intelligence 团队公开 `openpi` 代码、训练/推理示例、模型配置、远程推理流程，以及 `pi0`、`pi0-FAST`、`pi0.5` 相关基础能力；没有这些基础工作，本项目无法快速在天工机器人上建立 VLA 训练与部署链路。

本仓库不是 Physical Intelligence 官方项目，也不代表官方实现。本项目会尽量保留 upstream 贡献、许可证和引用关系，并在自己的实验代码中明确区分：

- 官方 `openpi` 已开源的能力。
- 论文、博客或公开材料中描述但尚未在 `openpi` 中完整开源的能力。
- 本项目基于公开资料、第三方复现和本地实验做出的工程尝试。

原始 `openpi` README 已保存在 [UPSTREAM_OPENPI_README.md](UPSTREAM_OPENPI_README.md)，便于查看官方安装、训练、推理和模型说明。

## 上游项目贡献

本项目主要继承并使用了 `openpi` 的以下贡献：

- `pi0` flow-based VLA 模型框架。
- `pi0-FAST` autoregressive VLA 与 FAST action tokenizer 相关工程入口。
- `pi0.5` flow matching head 的训练与推理路径。
- 自定义机器人数据接入范式：policy transform、LeRobot data config、train config、norm stats、policy server。
- PyTorch/JAX 训练与推理代码结构。
- 远程 websocket policy server/client 方案。
- DROID、ALOHA、LIBERO、UR5 等示例工程。

引用与致谢：

- OpenPI GitHub: <https://github.com/Physical-Intelligence/openpi>
- Physical Intelligence: <https://www.physicalintelligence.company/>
- pi0: <https://www.physicalintelligence.company/blog/pi0>
- pi0-FAST: <https://www.physicalintelligence.company/research/fast>
- pi0.5: <https://www.physicalintelligence.company/blog/pi05>
- Knowledge Insulation: <https://www.physicalintelligence.company/research/knowledge_insulation>

## 本项目目标

当前官方 `openpi` 已经提供 `pi0`、`pi0-FAST` 和 `pi0.5` 的公开训练/推理基础，但更后续的一些高级能力还没有以完整工程形式开源到 `openpi` 中。本项目希望在官方 `openpi` 基础上，围绕天工机器人和通用 VLA 能力，逐步复现与工程化以下方向：

- `pi0.5` 高级语义和 knowledge insulation 相关能力在自定义机器人上的可用训练流程。
- RTC/action chunking：异步推理、延迟估计、chunk suffix 融合、模型侧 guidance、训练期 delay conditioning。
- `pi0.6` 风格的 RECAP/RL：success/failure、human intervention、value proxy、advantage labeling、advantage-conditioned policy。
- MEM 相关的记忆、上下文和任务恢复能力。
- `pi0.7` 论文/公开材料中世界模型、视觉子目标、交互式上下文和长程任务规划相关思想的工程尝试。

## 当前进度

- 已在远程服务器 `A100_8:/data/caobochun/openpi` 部署 openpi。
- 已把天工机器人双手数据转换为 LeRobot 训练格式。
- 已基于 `pi0.5` 训练出天工机器人 VLA policy。
- 已实现天工 policy transform、训练 config、数据转换脚本、离线评估脚本和异步/RTC 部署客户端雏形。
- 已建立本地代码镜像 `/data/Openpi_CBC_Upgrade/openpi`，用于快速检查、编辑和同步到 A100_8 验证。
- 已建立同步脚本和 Codex skill，用于本地与 A100_8 双向同步。

## Todo List

### 1. 项目基线

- [x] 保留并引用 upstream `openpi` 项目。
- [x] 建立天工机器人 `pi0.5` 训练基线。
- [x] 建立本地代码镜像与 A100_8 同步流程。
- [x] 建立 GitHub 项目说明和进度记录。
- [ ] 整理天工训练/评估最小复现命令。
- [ ] 将当前实验配置、数据路径、checkpoint 约定整理成独立文档。

### 2. pi0.5 高级语义与 knowledge insulation

- [ ] 梳理官方 `pi0.5` 已开源实现与论文/博客描述之间的差距。
- [ ] 在天工数据上验证 prompt、task context、视觉语义变化对动作输出的影响。
- [ ] 设计多任务或多场景数据混合实验，观察语义泛化能力。
- [ ] 尝试可控的冻结/解冻、adapter、数据重加权或 knowledge-insulation-style 实验。

### 3. RTC / Real-Time Chunking

- [x] 建立异步 policy client 与 action chunk 执行雏形。
- [x] 建立 RTC chunker 的初步实现。
- [ ] 增加离线 replay 模式，不发布机器人动作即可测试延迟和 chunk 融合。
- [ ] 对比 async baseline、固定 horizon RTC、delay-adaptive RTC、suffix soft blend。
- [ ] 验证模型侧 RTC guidance 的 shape、归一化和采样稳定性。
- [ ] 尝试训练期 simulated delay conditioning。

### 4. pi0.6 RECAP / RL

- [ ] 参考公开 RECAP 复现项目，设计天工 episode metadata 和 sidecar label 格式。
- [ ] 增加 success/failure、human intervention、eval episode 标注。
- [ ] 实现 value proxy 或 progress proxy，生成 frame-level advantage label。
- [ ] 在 dataloader 中合并 `advantage_indicator`、`use_advantage`、`is_human_intervention`。
- [ ] 增加 advantage-conditioned prompt/token/embedding 路径。
- [ ] 增加 RECAP debug config，先跑 1-2 step smoke test。
- [ ] 对比普通 SFT 与 RECAP-style finetune 的离线指标。

### 5. MEM / 记忆机制

- [ ] 调研 MEM 公开材料和可复现工程线索。
- [ ] 设计短期历史、任务上下文、失败原因和恢复提示的数据结构。
- [ ] 实现不破坏旧 policy 输入的可选 context 字段。
- [ ] 在离线评估中测试有无记忆上下文的输出差异。

### 6. pi0.7 世界模型与长程交互

- [ ] 梳理 pi0.7 论文/博客/公开材料中的核心模块。
- [ ] 设计 world-model-style 中间表示或 visual subgoal 输入格式。
- [ ] 尝试用关键帧、子目标或语言分解增强 VLA policy 调用。
- [ ] 建立长程任务评估样例，记录成功率、失败模式和恢复过程。
- [ ] 明确哪些内容是论文思想复现，哪些是本项目工程假设。

### 7. 天工真机验证

- [ ] 真机前重新确认 26 维动作顺序、限幅、频率、插值和急停。
- [ ] 增加 dry-run/replay/只读观测模式。
- [ ] 小范围验证 policy server 延迟、图像状态同步和动作曲线。
- [ ] 记录失败样本，反哺 RECAP/RL 数据闭环。

## 开发约定

- 本仓库只提交代码、配置、文档和小型脚本。
- 不提交模型权重、checkpoint、训练数据、虚拟环境、日志、wandb 或评估大文件。
- 远端训练与验证默认在 `A100_8:/data/caobochun/openpi` 进行。
- 本地快速检查默认在 `/data/Openpi_CBC_Upgrade/openpi` 进行。

## License

本项目保留 upstream `openpi` 的许可证文件和第三方依赖声明。新增代码和文档在不冲突的前提下遵循原项目许可证约束；涉及第三方复现项目或论文思想时，会在对应文件中补充来源说明。
