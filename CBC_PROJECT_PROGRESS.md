# CBC_Pi0.7_Openpi 项目进度

## 项目目标

本项目是基于 OpenPI 的非官方个人复现和扩展工程，目标是在官方 OpenPI 基础上逐步补齐更完整的 VLA 工程栈能力，并面向人形机器人训练、异步部署和长时程任务验证做实验性探索。

本项目感谢并基于 Physical Intelligence 的 [`openpi`](https://github.com/Physical-Intelligence/openpi)。这是一个独立个人版本，主要实验方向包括：

- pi0.5 风格的高级语义训练和部署增强。
- RTC / real-time action chunking。
- pi0.6 风格的 RECAP、RL 和 MEM。
- pi0.7-inspired 世界模型和长时程交互方向。

## 当前状态

- 已保留上游 OpenPI 的署名、许可证和引用信息。
- 上游原始 README 已保存为 `UPSTREAM_OPENPI_README.md`。
- 本仓库保持 code-only：不纳入模型权重、数据集、虚拟环境、日志、wandb 输出或私有实验产物。
- 已建立项目 README、复现路线和 CBC 专项文档入口。

## 2026-06-18 进展：Training-Time RTC

本轮完成了 training-time RTC 的核心闭环，把早期推理期 VJP/Jacobian guidance 路线调整为训练期模拟延迟、部署期 hard-prefix 约束的方案。详细设计记录见 `docs/cbc/training_time_rtc.md`。

已完成内容：

- 新增 `RTCTrainingConfig` 配置，并接入 `pi05_tienkung_finetune_rtc` 训练配置。
- 在 PyTorch pi0.5 训练路径中按样本随机采样 prefix delay，将 prefix token 作为无噪动作条件输入，postfix token 继续执行 flow matching 加噪和 loss。
- 在采样路径中支持 `rtc_prefix`，每个 denoise step 都 hard overwrite prefix，并让 prefix token 使用 OpenPI 当前 flow 公式下的 clean endpoint。
- 更新策略接口和异步客户端示例，移除旧推理期 RTC guidance 字段，保留部署侧按真实 observed delay 跳过 prefix、执行 postfix 的路径。
- 新增 `examples/tienkung/rtc_chunker_test.py` 和 `examples/tienkung/simulate_rtc_replay.py`，用于 chunk suffix 融合和 hard-prefix replay smoke。
- 补充 `docs/cbc/README.md` 和 `docs/cbc/training_time_rtc.md`，记录参数选择、训练逻辑、部署逻辑、验证结果和后续实验建议。

已验证内容：

- 真实 batch forward/backward 通过，loss finite。
- hard-prefix `sample_actions` 通过，输出 shape 为 `(1, 50, 32)`。
- `train_pytorch` 2 step smoke 通过。
- `examples/tienkung` 编译通过。
- `rtc_chunker_test.py` 通过。
- `simulate_rtc_replay.py` hard-prefix smoke 通过。

## 待办列表

- [x] 实现基础人形机器人 observation/action 接口和天工双手示例路径。
- [x] 增加 RTC 相关 code-only smoke 测试和 replay 验证脚本。
- [x] 实现 training-time RTC hard-prefix 训练和采样闭环。
- [ ] 增加 RTC delay 分布配置：`uniform`、`exp`、`empirical`。
- [ ] 基于真实部署延迟日志生成 empirical delay histogram。
- [ ] 做离线 delay sweep：`d=0..25, s=25`。
- [ ] 对比普通 async、fixed-horizon RTC、delay-adaptive RTC、suffix soft blending、training-time RTC hard-prefix 和历史 inference-time guidance。
- [ ] 设计 RECAP metadata 和 sidecar label 格式。
- [ ] 增加 RECAP debug config 和 1-2 step smoke test。
- [ ] 设计可选 MEM/context 输入字段。
- [ ] 总结 pi0.7 公开概念，并清楚标注本项目的工程假设。
- [ ] 继续保持平台专用机器人桥接代码与 OpenPI 核心复现代码解耦。

## 相关项目

人形机器人 VLA 数据采集流水线见独立项目：

- [Knight1112D/Tienkung_vla_collect_data](https://github.com/Knight1112D/Tienkung_vla_collect_data)
