# CBC_Pi0.7_Openpi Progress

## Goal

This is an unofficial OpenPI-based project for learning, reproducing, and extending open-source VLA deployment on humanoid robots.

The project credits and builds on Physical Intelligence's [`openpi`](https://github.com/Physical-Intelligence/openpi). The goal is not to present an official PI implementation, but to explore a community-readable path toward capabilities such as:

- pi0.5-style higher-level semantics.
- RTC / real-time action chunking.
- pi0.6-style RECAP, RL, and MEM.
- pi0.7-inspired world-model and long-horizon interaction ideas.

## Current Status

- Upstream OpenPI attribution and license files are preserved.
- The original upstream README is saved as `UPSTREAM_OPENPI_README.md`.
- This repository is kept code-only: no model weights, datasets, virtual environments, logs, or private experiment artifacts.
- The first project-specific README and roadmap are in place.

## Todo List

- [ ] Add generic humanoid robot adapter documentation.
- [ ] Add code-only setup and smoke-test instructions.
- [ ] Add safe dry-run and replay-mode policy-client examples.
- [ ] Implement RTC offline replay and delay sweep.
- [ ] Design RECAP metadata and sidecar label format.
- [ ] Add RECAP debug config and 1-2 step smoke test.
- [ ] Design optional MEM/context input fields.
- [ ] Summarize pi0.7 public concepts and mark project assumptions clearly.
- [ ] Keep all platform-specific deployment notes separate from the generic OpenPI extension path.

## Related Project

For a concrete humanoid VLA data-collection pipeline, see:

- [Knight1112D/Tienkung_vla_collect_data](https://github.com/Knight1112D/Tienkung_vla_collect_data)
