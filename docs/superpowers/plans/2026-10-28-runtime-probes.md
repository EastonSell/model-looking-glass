# Runtime Probes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build reproducible runtime observation experiments comparing direct and gradient capture.
**Architecture:** One shared recorder, an instrumented deterministic residual model, and a finite evaluation CLI. Tests independently compare copied model states.
**Tech Stack:** Python 3.10+, PyTorch, standard-library unittest.
**Spec:** docs/superpowers/specs/2026-10-28-runtime-probes-design.md

## Global Constraints
- No remote telemetry or network access at runtime.
- Never mutate or replace main-path activation values or gradients.
- Report executed results only; distinguish unavailable dependencies.
- Float32 CPU eager is the initial evidence domain.
- Explicit branch metadata is not inferred from tensor values.

## Review Focus
- Inference mode must allow isolated gradients without changing the model graph.
- Repeated module calls must have distinct occurrences.
- Exception/overflow paths must remove hooks and not leak state.
- Non-finite values must fail explicitly rather than claim exact equality.
- Timing must measure complete capture and not mistake logged bytes for peak memory.

### Task 1: Capture channels and tests
Files: looking_glass/probes.py, tests/test_probes.py.
Interface: TraceSession(model, mode, placement, max_events) context manager; events list; TapPoint(node_id, group); gradient_readout(tensor).
- [ ] Write failing tests for exact detached capture, inference contexts, unchanged gradients, repeated calls, bound validation and cleanup.
- [ ] Run and retain the red result or dependency blocker.
- [ ] Implement direct, gradient and ID channels.
- [ ] Run full unittest suite and commit.

### Task 2: Model and numerical checks
Files: looking_glass/model.py, tests/test_model.py.
Interface: TinyResidual(width, depth=4, identical_branches=False); forward(x, route=0, perturb=None).
- [ ] Write failing tests comparing output, task loss, input/model gradients and SGD update for all channels.
- [ ] Implement named tap points and separately labeled identical branches.
- [ ] Run tests and commit.

### Task 3: Frozen evaluation and reporting
Files: looking_glass/experiment.py, tests/test_experiment.py, PROTOCOL.md.
Interface: run_experiment(output_dir, quick=False), CLI python -m looking_glass.experiment.
- [ ] Freeze protocol before the first complete run.
- [ ] Test no overwrite, finite configuration, equality metrics and deterministic output fields.
- [ ] Implement timing, trace comparison, metadata-only ambiguity example and JSON/CSV evidence.
- [ ] Run available tests and the full finite protocol; record environment and results honestly.

### Task 4: Package and review
Files: README.md, pyproject.toml, .github/workflows/tests.yml.
- [ ] Document usage, mathematical identity, tradeoffs, reproduction and limitations.
- [ ] Independently review full code and evidence.
- [ ] Fix material findings with regression tests, verify full suite and archive.
- [ ] Publish only through authorized available route and verify remote commit and CI.
