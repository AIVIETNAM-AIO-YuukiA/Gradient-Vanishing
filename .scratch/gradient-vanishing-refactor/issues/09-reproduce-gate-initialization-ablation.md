# 09: Reproduce the gate-initialization ablation

Status: resolved
Type: task
Blocked by: 06

**Approved specification:** [Refactoring specification](../../../docs/refactoring_spec.md)
**Architecture:** [Architecture proposal](../../../docs/architecture_proposal.md)
**Source notebook:** [GradientVanishingAnalysis.ipynb](../../../Notebooks/GradientVanishingAnalysis.ipynb)

**Shared preservation:** Keep the source notebook unchanged at SHA-256 `71ec17cdb7fffcc39fef12bfe4977f75117ceedaa90c03109e79f51d98059f44`; do not silently fix documented bugs; treat its saved outputs as unverified observations.

## What to build

A researcher can compare zero, one, and chrono retention bias against the Vanilla RNN reference at `T=100`.

## Objective

A researcher can compare zero, one, and chrono retention bias against the Vanilla RNN reference at `T=100`.

## Scope

Run the seven notebook conditions with default hyperparameters, five seeds, final test MSE, first checked validation MSE below `0.01`, and ten-seed initial-gradient decay. Save each condition's model, gate mode, length, seeds, effective config, threshold, and gradient settings. Allow smaller validation runs.

## Dependencies

06.

## Files or components affected

Ablation orchestration, raw ablation results, ablation tests.

## Acceptance criteria

- [x] Default config schedules 35 training runs; Vanilla RNN has no gate intervention; LSTM and GRU each use zero/one/chrono; threshold speed is based only on validation checks and reports success count; decay uses the same initial-gradient metric. Saved metadata identifies every condition and its run/measurement settings.

## Required tests

Condition matrix, gate-parameter signs, threshold resolution and non-achievement, mean/std formatting inputs, test isolation, reduced ablation result schema.

## Behavior-preservation checks

`W_z.bias` is the GRU intervention, not `W_r`; chrono's RNG draw and LSTM input-bias pairing stay intact; no gate bias enters Optuna.

## Definition of Done

- [x] A reduced ablation runs independently of the main grid and produces raw metrics for later table generation.

## Answer

`run_gate_ablation` in `src/gradient_vanishing/experiments.py` reproduces notebook cell 27's seven ordered conditions and saves raw per-seed metrics, threshold outcomes, float64 initial-gradient curves, decay values, and provenance. The training metadata records each initialized gate-bias vector before updates, including the GRU sign and chrono LSTM input pairing. `docs/gate_ablation.md` documents the notebook audit and GRU retention-bias sign. The reduced `T=5`, two-seed, two-update comparison matched direct notebook execution for initial gate parameters, predictions, losses, test metrics, gradient curves, decay, ordering, and final Torch RNG. The full 35-run experiment remains unexecuted.
