# CBC_Pi0.7_Openpi

## Overview

`CBC_Pi0.7_Openpi` is an unofficial personal research and engineering attempt to build a more complete OpenPI project for humanoid robot VLA research.

The project is built on top of Physical Intelligence's open-source [`openpi`](https://github.com/Physical-Intelligence/openpi) repository. I am grateful to the Physical Intelligence team for releasing the OpenPI codebase, model definitions, training and inference examples, remote policy-server workflow, and public `pi0`, `pi0-FAST`, and `pi0.5` assets. This repository would not be possible without that foundation.

This is an independent project built with deep respect for the upstream OpenPI work. Its purpose is to reproduce, integrate, and extend OpenPI-style engineering pieces that have been discussed in public materials but have not yet been released as a complete OpenPI implementation.

The original upstream OpenPI README is preserved here:

- [UPSTREAM_OPENPI_README.md](UPSTREAM_OPENPI_README.md)

## Upstream Credit

This repository inherits and builds on the following upstream OpenPI contributions:

- `pi0`, a flow-based vision-language-action model.
- `pi0-FAST`, an autoregressive VLA based on the FAST action tokenizer.
- `pi0.5`, including the publicly released flow-matching training and inference path.
- The policy transform, LeRobot data config, training config, normalization-statistics, and policy-server patterns used to adapt OpenPI to new robot embodiments.
- JAX and PyTorch model/training structure.
- Remote websocket policy serving.
- Reference examples for DROID, ALOHA, LIBERO, UR5, and related workflows.

Primary upstream references:

- OpenPI GitHub: <https://github.com/Physical-Intelligence/openpi>
- Physical Intelligence: <https://www.physicalintelligence.company/>
- pi0: <https://www.physicalintelligence.company/blog/pi0>
- pi0-FAST: <https://www.physicalintelligence.company/research/fast>
- pi0.5: <https://www.physicalintelligence.company/blog/pi05>
- Knowledge Insulation: <https://www.physicalintelligence.company/research/knowledge_insulation>

## Project Goal

OpenPI currently provides a strong open-source base for `pi0`, `pi0-FAST`, and `pi0.5`. Several later capabilities discussed in public materials are not yet available as a full open-source OpenPI engineering stack.

This project aims to turn those public ideas into a more complete, runnable OpenPI-style engineering stack:

- Humanoid robot VLA training, inference, and evaluation inside the OpenPI codebase.
- `pi0.5`-style higher-level semantic generalization and knowledge-insulation-inspired experiments.
- RTC / real-time action chunking for remote VLA inference.
- `pi0.6`-style RECAP, RL, human intervention labels, value proxies, and advantage-conditioned policies.
- MEM-style memory, task context, and recovery mechanisms.
- `pi0.7`-inspired world-model, visual-subgoal, interactive-context, and long-horizon planning ideas.

The target robot platform is intentionally not fixed. The repository is organized around completing the OpenPI engineering stack first, then validating it on humanoid robot platforms.

## Related Project

For one concrete humanoid VLA data-collection pipeline, see my separate project:

- [Knight1112D/Tienkung_vla_collect_data](https://github.com/Knight1112D/Tienkung_vla_collect_data)

That repository is a data-collection reference. This repository is focused on building the OpenPI-side model, data, training, inference, real-time execution, and higher-level capability stack.

## Roadmap

### 1. OpenPI Foundation

- [x] Preserve upstream OpenPI credit, license files, and README reference.
- [x] Keep the project as a code-only mirror with no model weights, datasets, virtual environments, or training logs.
- [x] Add a public project README explaining the unofficial scope.
- [ ] Keep the upstream OpenPI baseline reproducible after each project change.
- [ ] Add project smoke tests for new model, data, and policy-server paths.

### 2. Humanoid VLA Deployment

- [ ] Define a humanoid observation/action interface inside the OpenPI data and policy transform stack.
- [ ] Add dry-run and replay-mode policy clients for local validation.
- [ ] Add offline checks for policy-server latency, action chunks, and action normalization.
- [ ] Add robot-side safety contracts for joint ordering, limits, frequency, interpolation, and stop behavior.
- [ ] Keep robot-specific bridges isolated from the core OpenPI reproduction work.

### 3. pi0.5-Style Semantics

- [ ] Compare the public `pi0.5` implementation with the public descriptions of higher-level semantic generalization.
- [ ] Test prompt sensitivity, task context, and visual semantic changes on humanoid manipulation datasets.
- [ ] Explore multi-task or multi-scene mixtures for semantic generalization.
- [ ] Try controlled freezing, adapters, data weighting, or knowledge-insulation-inspired training recipes.

### 4. RTC / Real-Time Chunking

RTC 是当前项目优先推进的远程推理稳定性方向。人形机器人异步部署实验通常采用 policy server：控制循环保持固定频率发布动作，模型在后台生成下一段 action chunk。推理期间已经执行或即将执行的旧 chunk 前缀会成为新 chunk 的 hard prefix，因此 RTC 的关键目标是让新 chunk 在延迟存在时仍能和已承诺动作连续衔接。

当前路线已经从早期 inference-time VJP/Jacobian guidance 调整为 training-time RTC：训练时随机模拟推理延迟，把 prefix token 作为无噪动作条件输入；部署时只传 `rtc_prefix`，采样循环每个 denoise step 都 hard overwrite prefix，不再在推理期做额外反传修正。详细设计见 [`docs/cbc/training_time_rtc.md`](docs/cbc/training_time_rtc.md)。

已完成：

- [x] 新增 `RTCTrainingConfig`，并接入 `pi05_tienkung_finetune_rtc`。
- [x] 在 PyTorch pi0.5 flow-matching 训练中加入 training-time delay sampling：prefix 使用 clean action，postfix 继续加噪并只对有效 postfix 计算 loss。
- [x] 在 `PI0Pytorch.sample_actions()` 中支持 `rtc_prefix` hard-prefix 采样，prefix token 使用当前 OpenPI flow 公式下的 clean endpoint。
- [x] 更新 policy 接口和异步客户端示例，部署侧按 observed delay 跳过已执行 prefix，只执行新 chunk 的 postfix。
- [x] 增加 `examples/tienkung/rtc_chunker_test.py` 和 `examples/tienkung/simulate_rtc_replay.py`，用于 chunk suffix 融合和 hard-prefix replay smoke。
- [x] 完成真实 batch forward/backward、hard-prefix `sample_actions`、2 step training smoke、示例编译和 replay smoke。

后续实验：

- [ ] 实现系统化 offline delay simulation 和 replay evaluation。
- [ ] 对比 async baseline、fixed-horizon RTC、delay-adaptive RTC、suffix soft blending 和 training-time RTC hard-prefix。
- [ ] 补充 delay 分布配置：`uniform`、`exp`、`empirical`。
- [ ] 用真实推理延迟日志生成 empirical delay histogram，并做 `d=0..25, s=25` 的离线 delay sweep。
- [ ] 记录并汇报 latency、replan frequency、action smoothness、prefix/postfix discontinuity、rollout stability 等指标。
- [ ] 如果后续重新验证 model-side inference-time guidance，需要作为实验性开关和 training-time RTC 对照项，而不是默认部署路径。

### 5. pi0.6-Style RECAP / RL

- [ ] Define generic episode metadata for success, failure, evaluation episodes, and human intervention.
- [ ] Generate sidecar labels for `advantage_indicator`, `use_advantage`, and `is_human_intervention`.
- [ ] Add value-proxy or progress-proxy tools for offline advantage labeling.
- [ ] Merge RECAP sidecar fields into the dataloader without modifying the original dataset.
- [ ] Add advantage-conditioned prompt, token, or embedding paths.
- [ ] Add a small debug config for 1-2 step smoke tests.
- [ ] Compare standard SFT and RECAP-style fine-tuning.

### 6. MEM / Memory

- [ ] Survey public MEM-related materials and reproducible implementation clues.
- [ ] Design optional memory/context fields for recent history, task state, failures, and recovery hints.
- [ ] Keep memory inputs optional so existing OpenPI policies remain compatible.
- [ ] Evaluate behavior with and without memory context.

### 7. pi0.7-Inspired World Model

- [ ] Summarize public pi0.7 paper/blog concepts and separate them from project assumptions.
- [ ] Design world-model-style intermediate representations or visual-subgoal inputs.
- [ ] Explore keyframes, subgoals, and language decomposition as policy context.
- [ ] Build long-horizon humanoid task examples and track success, failure modes, and recovery.

## Repository Policy

- No model weights.
- No private datasets.
- No virtual environments.
- No machine-specific logs or experiment artifacts.
- Keep upstream OpenPI attribution visible.
- Clearly label unofficial reproduction code and speculative engineering experiments.

## License

This repository preserves the upstream OpenPI license files and third-party notices. New code and documentation should remain compatible with upstream licensing constraints. Any third-party reproduction work or paper-inspired implementation should include source notes in the relevant files.
