# 10: Reproduce the numeric retention-bias sweep

Status: resolved
Type: task
Blocked by: 06

**Approved specification:** [Refactoring specification](../../../docs/refactoring_spec.md)
**Architecture:** [Architecture proposal](../../../docs/architecture_proposal.md)
**Source notebook:** [GradientVanishingAnalysis.ipynb](../../../Notebooks/GradientVanishingAnalysis.ipynb)

**Shared preservation:** Keep the source notebook unchanged at SHA-256 `71ec17cdb7fffcc39fef12bfe4977f75117ceedaa90c03109e79f51d98059f44`; do not silently fix documented bugs; treat its saved outputs as unverified observations.

## What to build

A researcher can measure how initial gradient decay varies with the abstract retention bias for LSTM and custom GRU.

## Objective

A researcher can measure how initial gradient decay varies with the abstract retention bias for LSTM and custom GRU.

## Scope

Sweep 17 values from `-4` through `4`, five initialization seeds per point, with theoretical sigmoid retention values and measured decay factors. Return numeric plot inputs plus model, length, exact bias grid, seeds, dtype, initialization, and fit settings.

## Dependencies

06. It can proceed alongside 09.

## Files or components affected

Sweep orchestration, raw sweep results, sweep tests.

## Acceptance criteria

- [x] Default sweep uses `T=100`, both gated models, 17 evenly spaced biases, and five seed curves per bias; LSTM uses `+b` at forget bias, GRU uses `-b` at Update Gate bias; results include measured decay, theoretical retention, and enough metadata to repeat each measurement.

## Required tests

Sweep grid/count, endpoint and spacing checks, gate sign, reduced sweep reproducibility, decay calculation, no training/test-set access.

## Behavior-preservation checks

Keep model construction and RNG order per bias/seed, float64 gradient measurement, and the notebook's custom GRU equation. Do not reinterpret the sweep as a Reset Gate intervention.

## Definition of Done

- [x] A reduced sweep independently yields numeric arrays needed for the reference plot.

## Answer

`run_retention_bias_sweep` in `src/gradient_vanishing/experiments.py` reproduces the notebook's LSTM/GRU numeric gradient sweep and saves per-bias float64 initial-gradient curves, measured decay, theoretical sigmoid retention, assigned gate bias, and provenance. `docs/retention_bias_sweep.md` documents the notebook audit and measurement definition. A reduced `T=5`, two-seed run compared all 17 original biases for both models directly with original notebook gradient and decay definitions; gate vectors, predictions, initial MSE, curves, decay statistics, and final Torch RNG matched. The full `T=100`, five-seed sweep remains unexecuted. The source sweep does not train models or produce validation/test MSE.
