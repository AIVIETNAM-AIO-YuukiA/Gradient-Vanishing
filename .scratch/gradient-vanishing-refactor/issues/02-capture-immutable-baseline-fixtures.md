# 02: Capture immutable baseline fixtures

Status: claimed
Type: task
Blocked by: 01

**Approved specification:** [Refactoring specification](../../../docs/refactoring_spec.md)  
**Architecture:** [Architecture proposal](../../../docs/architecture_proposal.md)  
**Source notebook:** [GradientVanishingAnalysis.ipynb](../../../Notebooks/GradientVanishingAnalysis.ipynb)

**Shared preservation:** Keep the source notebook unchanged at SHA-256 `71ec17cdb7fffcc39fef12bfe4977f75117ceedaa90c03109e79f51d98059f44`; do not silently fix documented bugs; treat its saved outputs as unverified observations.

## What to build

A reviewer has baseline numerical evidence before any extraction can change the implementation.

## Objective

A reviewer has baseline numerical evidence before any extraction can change the implementation.

## Scope

Run small, deterministic CPU cases directly from the untouched notebook definitions in the pinned environment; record initial parameters, one-step predictions, short loss traces, validation/test MSE, and hidden-gradient norms. Record Python and dependency versions, CPU/device details, deterministic settings, source revision, notebook hash, invocation, cell provenance, and fixture checksums before extraction. Predeclare comparison tolerances.

## Dependencies

01.

## Files or components affected

Baseline fixture generator or capture script, fixture data, fixture metadata, fixture verification tests.

## Acceptance criteria

- [ ] Fixtures cover Vanilla RNN, LSTM, and custom GRU at short and longer sequence lengths; parameter arrays, seed/config metadata, complete baseline environment, and fixture provenance are stored before extraction; saved notebook output cells are labeled historical observations and are not used as proof of reproduction.

## Required tests

Re-run the capture on the same pinned CPU environment and compare initial parameters exactly and computed float32/float64 quantities at the tolerances in the approved specification; verify fixture source hash and immutability policy.

## Behavior-preservation checks

Execute definitions without changing notebook cells or RNG call order. Capture the original evaluation-mode side effect and full BPTT behavior rather than correcting them.

## Definition of Done

- [ ] A reviewer can reproduce and inspect the small-case fixtures and baseline environment; notebook SHA-256 still matches. This ticket completes before any implementation extraction in tickets 03–06.
