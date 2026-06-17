# CBC_Pi0.7_Openpi Progress

## Goal

This is an unofficial OpenPI-based project for reproducing, integrating, and extending a more complete OpenPI-style VLA engineering stack for humanoid robots.

The project credits and builds on Physical Intelligence's [`openpi`](https://github.com/Physical-Intelligence/openpi). It is an independent personal version that experiments with capabilities such as:

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

- [ ] Implement a humanoid observation/action interface in the OpenPI transform stack.
- [ ] Add code-only smoke tests for new model, data, and policy-server paths.
- [ ] Add dry-run and replay-mode policy clients for validation.
- [ ] Implement RTC offline replay and delay sweep.
- [ ] Design RECAP metadata and sidecar label format.
- [ ] Add RECAP debug config and 1-2 step smoke test.
- [ ] Design optional MEM/context input fields.
- [ ] Summarize pi0.7 public concepts and mark project assumptions clearly.
- [ ] Keep platform-specific robot bridges separate from the core OpenPI reproduction work.

## Related Project

For a concrete humanoid VLA data-collection pipeline, see:

- [Knight1112D/Tienkung_vla_collect_data](https://github.com/Knight1112D/Tienkung_vla_collect_data)
