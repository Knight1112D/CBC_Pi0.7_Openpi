# CBC_Pi0.7_Openpi

## Overview

`CBC_Pi0.7_Openpi` is an unofficial, personal research and engineering attempt to build a more complete OpenPI-style VLA project for humanoid robots.

The project is built on top of Physical Intelligence's open-source [`openpi`](https://github.com/Physical-Intelligence/openpi) repository. I am grateful to the Physical Intelligence team for releasing the OpenPI codebase, model definitions, training and inference examples, remote policy-server workflow, and public `pi0`, `pi0-FAST`, and `pi0.5` assets. This repository would not be possible without that foundation.

This repository is not affiliated with or endorsed by Physical Intelligence. It is a community-oriented experiment intended to help others learn how to adapt open-source VLAs to humanoid robots, while also exploring research ideas that have not yet been released as a complete OpenPI implementation.

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

This project aims to explore, reproduce, and teach the engineering path toward those capabilities on humanoid robots:

- Open-source VLA deployment on humanoid robots.
- `pi0.5`-style higher-level semantic generalization and knowledge-insulation-inspired experiments.
- RTC / real-time action chunking for remote VLA inference.
- `pi0.6`-style RECAP, RL, human intervention labels, value proxies, and advantage-conditioned policies.
- MEM-style memory, task context, and recovery mechanisms.
- `pi0.7`-inspired world-model, visual-subgoal, interactive-context, and long-horizon planning ideas.

The target robot platform is intentionally not fixed. The project is meant to be useful for humanoid robots in general, and future experiments may use different humanoid platforms.

## Related Project

For one concrete humanoid VLA data-collection pipeline, see my separate project:

- [Knight1112D/Tienkung_vla_collect_data](https://github.com/Knight1112D/Tienkung_vla_collect_data)

That repository is a data-collection reference. This repository is focused on adapting and extending OpenPI-style VLA training, inference, real-time execution, and higher-level model capabilities.

## Roadmap

### 1. OpenPI Foundation

- [x] Preserve upstream OpenPI credit, license files, and README reference.
- [x] Keep the project as a code-only mirror with no model weights, datasets, virtual environments, or training logs.
- [x] Add a public project README explaining the unofficial scope.
- [ ] Provide minimal reproducible commands for code-only setup and smoke tests.
- [ ] Document how to adapt a new humanoid robot dataset to OpenPI policy transforms and LeRobot data configs.

### 2. Humanoid VLA Deployment

- [ ] Define a generic humanoid observation/action adapter interface.
- [ ] Provide a safe dry-run client that never publishes robot actions.
- [ ] Provide replay-mode inference for testing policy-server latency and action chunks offline.
- [ ] Document action normalization, joint ordering, limits, frequency, interpolation, and emergency-stop checks.
- [ ] Add examples for swapping robot-specific ROS or middleware bridges.

### 3. pi0.5-Style Semantics

- [ ] Compare the public `pi0.5` implementation with the public descriptions of higher-level semantic generalization.
- [ ] Test prompt sensitivity, task context, and visual semantic changes on humanoid manipulation datasets.
- [ ] Explore multi-task or multi-scene mixtures for semantic generalization.
- [ ] Try controlled freezing, adapters, data weighting, or knowledge-insulation-inspired training recipes.

### 4. RTC / Real-Time Chunking

- [ ] Implement offline delay simulation and replay evaluation.
- [ ] Compare async baseline, fixed-horizon RTC, delay-adaptive RTC, and suffix soft blending.
- [ ] Validate model-side RTC guidance in the flow-matching sampling loop.
- [ ] Explore training-time delay conditioning.
- [ ] Report metrics such as latency, replan frequency, action smoothness, and rollout stability.

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
