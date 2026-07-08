# CBC_Pi0.7_Openpi

Language: English | [简体中文](README.zh-CN.md)

## Overview

`CBC_Pi0.7_Openpi` is an unofficial personal research and engineering attempt to build a more complete OpenPI project for humanoid robot VLA research.

The project is built on top of Physical Intelligence's open-source [`openpi`](https://github.com/Physical-Intelligence/openpi) repository. I am grateful to the Physical Intelligence team for releasing the OpenPI codebase, model definitions, training and inference examples, remote policy-server workflow, and public `pi0`, `pi0-FAST`, and `pi0.5` assets. This repository would not be possible without that foundation.

This is an independent project built with deep respect for the upstream OpenPI work. Its purpose is to reproduce, integrate, and extend OpenPI-style engineering pieces that have been discussed in public materials but have not yet been released as a complete OpenPI implementation.

This project also thanks the OpenTau project for useful public engineering references around later OpenPI-style training directions, including memory-conditioned policies, value/advantage labeling, and policy-family organization. In this repository those ideas are adapted behind explicit OpenPI-compatible switches so the default upstream training path remains unchanged.

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
- `pi0.5`-style higher-level semantic generalization and KI (Knowledge Insulation / Knowledge Isolation) experiments.
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

### 1. Humanoid VLA Deployment

- [ ] Define a humanoid observation/action interface inside the OpenPI data and policy transform stack.
- [ ] Add dry-run and replay-mode policy clients for local validation.
- [ ] Add offline checks for policy-server latency, action chunks, and action normalization.
- [ ] Add robot-side safety contracts for joint ordering, limits, frequency, interpolation, and stop behavior.
- [ ] Keep robot-specific bridges isolated from the core OpenPI reproduction work.

### 2. pi0.5-Style Semantics / KI

- [ ] Compare the public `pi0.5` implementation with the public descriptions of higher-level semantic generalization.
- [ ] Test prompt sensitivity, task context, and visual semantic changes on humanoid manipulation datasets.
- [ ] Explore multi-task or multi-scene mixtures for semantic generalization.
- [ ] Add KI-lite experiments on the OpenPI PyTorch pi0.5 path: train VLM motor representations with FAST action-token CE, train the action expert with flow loss, and stop flow/action-expert gradients from flowing back into the VLM backbone.
- [ ] Keep `stop-gradient only` separate from true KI-lite; do not call the work KI unless FAST CE and flow loss are both part of the experiment.
- [ ] Test KI with RTC as composable switches, without breaking training-time delay sampling, hard-prefix sampling, or masked postfix loss.
- [ ] Try controlled freezing, adapters, data weighting, and KI-inspired training recipes.

### 3. RTC / Real-Time Chunking

RTC is the current priority for improving remote-inference stability. In an asynchronous humanoid policy-server setup, the robot control loop keeps publishing actions at a fixed rate while the model generates the next action chunk in the background. Actions that have already been executed, or will be executed before the next chunk arrives, become the hard prefix of the new chunk. The RTC goal is to make the newly generated chunk remain continuous with those committed actions under nonzero inference latency.

The current implementation has moved from an early inference-time VJP/Jacobian-guidance idea to training-time RTC. Training randomly simulates inference delay and feeds prefix tokens as clean action conditions. Deployment only passes `rtc_prefix`; the flow-matching sampling loop hard-overwrites the prefix at every denoising step and no longer performs extra inference-time backpropagation. See [`docs/cbc/training_time_rtc.md`](docs/cbc/training_time_rtc.md) for the detailed design notes.

Completed:

- [x] Added `RTCTrainingConfig` and wired it into `pi05_tienkung_finetune_rtc`.
- [x] Added training-time delay sampling to the PyTorch pi0.5 flow-matching path: prefix tokens use clean actions, postfix tokens stay noisy, and loss is computed only on valid postfix steps.
- [x] Added `rtc_prefix` hard-prefix sampling support in `PI0Pytorch.sample_actions()`, with prefix tokens using the clean endpoint under the current OpenPI flow convention.
- [x] Updated the policy interface and asynchronous client example so deployment skips already executed prefix steps according to observed delay and executes only the new postfix.
- [x] Added RTC smoke tests and replay tools under `examples/tienkung/rtc/` and `examples/tienkung/eval/`.
- [x] Verified real-batch forward/backward, hard-prefix `sample_actions`, 2-step training smoke, example compilation, and replay smoke.

Next experiments:

- [ ] Implement systematic offline delay simulation and replay evaluation.
- [ ] Compare the async baseline, fixed-horizon RTC, delay-adaptive RTC, suffix soft blending, and training-time RTC hard-prefix variants.
- [ ] Add delay distribution options: `uniform`, `exp`, and `empirical`.
- [ ] Build an empirical delay histogram from real inference-latency logs and run an offline delay sweep with `d=0..25, s=25`.
- [ ] Report latency, replan frequency, action smoothness, prefix/postfix discontinuity, and rollout stability.
- [ ] If model-side inference-time guidance is revisited, keep it as an experimental switch and comparison baseline rather than the default deployment path.

### 4. pi0.6-Style RECAP / RL

- [x] Define generic episode metadata for success, failure, evaluation episodes, and human intervention.
- [x] Generate sidecar labels for `advantage_indicator`, `use_advantage`, and `is_human_intervention`.
- [x] Add a code-only sidecar generator for offline advantage/RL-token labels.
- [x] Merge RECAP sidecar fields into the dataloader without modifying the original dataset.
- [x] Add RL-token sample weighting for PyTorch flow-matching loss.
- [x] Add a small debug config for 1-2 step smoke tests.
- [ ] Compare standard SFT and RECAP-style fine-tuning.

### 5. MEM / Memory

- [x] Survey public MEM-related materials and reproducible implementation clues.
- [x] Design optional memory/context fields for recent history, task state, failures, and recovery hints.
- [x] Keep memory inputs optional so existing OpenPI policies remain compatible.
- [x] Add optional sidecar memory-to-prompt augmentation for debug runs.
- [ ] Evaluate behavior with and without memory context.

### 6. pi0.7-Inspired World Model

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
