# 06: Reproduce hidden-gradient diagnostics and full BPTT checks

Status: resolved
Type: task
Blocked by: 05

**Approved specification:** [Refactoring specification](../../../docs/refactoring_spec.md)
**Architecture:** [Architecture proposal](../../../docs/architecture_proposal.md)
**Source notebook:** [GradientVanishingAnalysis.ipynb](../../../Notebooks/GradientVanishingAnalysis.ipynb)

**Shared preservation:** Keep the source notebook unchanged at SHA-256 `71ec17cdb7fffcc39fef12bfe4977f75117ceedaa90c03109e79f51d98059f44`; do not silently fix documented bugs; treat its saved outputs as unverified observations.

## Objective

A researcher can inspect hidden-state gradient flow at initialization and during training with the notebook's metric.

## Scope

Extract gradient-norm reading, float64 ten-seed initial-gradient curves, geometric aggregation, decay-factor fit, and selected pre-clipping training captures. Return numeric arrays and the seed list, model/configuration, sequence length, dtype, device, and fit/capture settings needed to reproduce them.

## Dependencies

05.

## Files or components affected

Experiment analysis functions, gradient result records, gradient/BPTT tests.

## Acceptance criteria

- [x] Intermediate hidden states remain connected across the full sequence; `dL/dh_t` uses mean batch L2 norm; initial curves divide by last-step norm; decay uses the notebook fit window; default capture indices are 0, 50, 150, 299, with valid reduced-run indices supplied only by smoke config. Saved metadata identifies the seeds, dtype, model/configuration, sequence length, and fit/capture settings.

## Required tests

Independent short-sequence gradient calculation, first-step gradient connection, norm ordering/length, float64 ratio and geometric mean, decay fit, capture before clipping and optimizer step, zero/non-finite ratio reporting.

## Behavior-preservation checks

Do not truncate BPTT, detach live hidden/cell states, substitute parameter gradients, or claim hidden-only metrics measure all LSTM cell paths. Preserve notebook clipping and normalization rules.

## Definition of Done

- [x] Reduced gradient run matches baseline small-case fixtures, emits arrays needed for both plots, and passes full-BPTT tests.

## Answer

Implemented `src/gradient_vanishing/gradients.py` for retained hidden-gradient measurement, float64 initial curves and decay fitting, and pre-clipping float32 training captures. Structured JSON results contain raw and transformed arrays plus provenance. `docs/gradient_diagnostics.md` records the exact metric, timestep and fit conventions, RNG and mode dependencies, and limitations. Tests compare all 12 stored model-gradient cases and three training capture cases with the immutable fixture; three reduced initial-curve cases are compared with direct execution of the original notebook definitions. Independent tests check the chain rule, batch-mean norm, full BPTT, capture order, and nonfinite reporting. The full suite passed: 96 tests, 0 failures. No full `T=100` empirical gradient profile was run.
