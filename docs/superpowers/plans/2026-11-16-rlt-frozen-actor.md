# RLT Frozen Actor Implementation Plan

> Execute inline using superpowers:executing-plans; final review by a fresh reviewer.

**Goal:** Prevent unconstrained offline Actor updates on failed episodes.
**Architecture:** Persist a frozen warmup Actor in the existing learner checkpoint;
use it as the failed-state behavior target and expose divergence diagnostics.
**Tech Stack:** Existing Python/PyTorch, unittest, Docker runtime.
**Spec:** ../specs/2026-11-16-rlt-frozen-actor-design.md

## Constraints and review focus

- Preserve existing uncommitted changes and model artifacts; no new dependencies.
- Anchor must survive repeated begin_phase, bundle resume, and async snapshots.
- A missing anchor in an already-trained RL state must not be recreated silently.
- Successful labels, active-joint mask, and reference dropout semantics must hold.
- Alarm failure must precede successful publication; thresholds are diagnostics.
- Real data tests use immutable Critic warmup and separate diagnostic output.

## Tasks

- [x] Add failing tests for failed-state penalty, frozen parameters and exact
  checkpoint resume, old warmup compatibility and old RL rejection.
- [x] Implement anchor lifecycle/state and objective in rlt/stage2.py.
- [x] Add failing CLI tests for metrics, request provenance, and finite divergence.
- [x] Implement update metrics and CLI diagnostics/alarms.
- [x] Run RLT core/runtime suites and a 10,000-step real-data replay check.
- [x] Review the complete diff, resolve important findings, report checks.

## Execution notes

- Approved approach is recorded in the preceding user conversation. No further
  design approval is needed. Work in place because the running environment mounts
  this checkout and relies on existing uncommitted normalization changes.
- Frozen-anchor objective, full-reference dropout behavior, persistence, and old
  RL rejection were verified RED -> GREEN. CLI integration verifies failure
  before artifact publication.
- Fresh reviewer identified async guard bypass and pre-optimizer-only metrics.
  Both plus split-path provenance checking now have RED -> GREEN regression tests.
  Shared validation runs before async callbacks; metrics include updated policy.
- Final relevant suite: 149 passed, 5 skipped (showroom artifacts unavailable).
  An initial unittest run could not import pytest-based data-validation tests;
  reran all RLT tests with temporary pytest tools, without changing dependencies.
- Real replay: 10,000 updates with a state restore at 5,000 preserved the anchor
  exactly; final Critic loss 0.00024750118609517813.
- Final CLI verification with OMP_NUM_THREADS=2 completed 10,000 updates and
  saved `/tmp/rlt_frozen_actor_cli_validation_final` in the groot container.
  Final Critic loss 0.00024750118609517813, policy Q absolute maximum
  0.8048294186592102, target absolute maximum 0.8016306161880493, deviation
  maximum 0.09380173683166504. Bundle resume confirmed unchanged frozen anchor
  equal to the original warmup Actor and a different trained Actor.
- Verification outputs on host: `/tmp/rlt_final_tests.log`,
  `/tmp/rlt_final_cli_verified.log`, `/tmp/rlt_implemented_verify.jsonl`.
  Earlier CLI validation with the inference container's default CPU thread
  count was stopped and replaced by the service's two-thread configuration.
  No production checkpoints were replaced. Working-tree changes are left for
  the user; pre-existing changes are preserved.
