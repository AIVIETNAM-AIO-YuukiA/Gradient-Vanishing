# 08: Reproduce the default-versus-tuned main comparison

Status: resolved
Type: task
Blocked by: 06, 07

**Approved specification:** [Refactoring specification](../../../docs/refactoring_spec.md)
**Architecture:** [Architecture proposal](../../../docs/architecture_proposal.md)
**Source notebook:** [GradientVanishingAnalysis.ipynb](../../../Notebooks/GradientVanishingAnalysis.ipynb)

**Shared preservation:** Keep the source notebook unchanged at SHA-256 `71ec17cdb7fffcc39fef12bfe4977f75117ceedaa90c03109e79f51d98059f44`; do not silently fix documented bugs; treat its saved outputs as unverified observations.

## What to build

A researcher can compare Vanilla RNN, LSTM, and GRU at configured sequence lengths with default and tuned hyperparameters over the same seeds.

## Objective

A researcher can compare Vanilla RNN, LSTM, and GRU at configured sequence lengths with default and tuned hyperparameters over the same seeds.

## Scope

Orchestrate the three-length, three-model, two-condition, five-seed grid; invoke tuning per `(length, model)`; collect final test MSE and histories, including selected training gradients. Save per-run model, length, condition, seed, effective parameters, validation/test split seeds, and tuning-study reference. Use reduced config in ticket tests.

## Dependencies

06 and 07.

## Files or components affected

Main experiment orchestration, raw comparison result record, grid tests.

## Acceptance criteria

- [x] Default config schedules nine studies and 90 final runs; both conditions use retention-bias mode `one`; each run uses final weights and the fixed test set for its length; raw per-seed values, selected tuned parameters, and per-run reproduction metadata are available for reporting.

## Required tests

Reduced grid cardinality and ordering, exact seeds, same fixed test set per length, test exclusion from tuning, final test MSE, captured gradient indices, sample mean/std (`ddof=1`).

## Behavior-preservation checks

Keep the 150-versus-300-update distinction and default-versus-tuned semantics. Do not claim saved notebook numbers are reproduced by reduced runs.

## Definition of Done

- [x] A reduced grid yields inspectable raw results and passes the baseline-compatible orchestration tests; full grid remains ticket 14.

## Answer

`run_main_comparison` in `src/gradient_vanishing/experiments.py` follows notebook cell 17's length, model, condition, and seed order. It reuses the existing tuning, training, data, evaluation, and gradient implementations, and retains partial results on failure. `docs/main_comparison.md` documents the notebook audit and output schema. The reduced direct-notebook check used `T=10`, Vanilla RNN, two seeds, two tuning trials of two updates, and three final training updates. Selected parameters, losses, gradients, metrics, and final Torch RNG digest matched; all 109 tests passed. The complete nine-study, 90-run matrix remains for Ticket 14.
