# 05: Reproduce one seeded training and evaluation run

Status: resolved
Blocked by: 03, 04

**Approved specification:** [Refactoring specification](../../../docs/refactoring_spec.md)
**Architecture:** [Architecture proposal](../../../docs/architecture_proposal.md)
**Source notebook:** [GradientVanishingAnalysis.ipynb](../../../Notebooks/GradientVanishingAnalysis.ipynb)

**Shared preservation:** Keep the source notebook unchanged at SHA-256 `71ec17cdb7fffcc39fef12bfe4977f75117ceedaa90c03109e79f51d98059f44`; do not silently fix documented bugs; treat its saved outputs as unverified observations.

## What to build

A researcher can train one selected model/length/configuration and obtain a history plus final-model validation/test MSE.

## Objective

A researcher can train one selected model/length/configuration and obtain a history plus final-model validation/test MSE.

## Scope

Extract run seeding, fresh-batch training, Adam/MSE, gradient clipping, validation schedule, pruning hooks, and evaluation. Expose the final model state and a minimal per-run metadata record containing effective configuration, random seed, dependency versions, effective hyperparameters, device, and notebook hash. Do not add best-checkpoint selection or a separate checkpoint manager because the notebook has neither.

## Dependencies

03 and 04.

## Files or components affected

Training module, single-run result and metadata record, training-contract tests.

## Acceptance criteria

- [x] Each update zeros gradients, generates a fresh batch, computes loss, backpropagates, captures selected gradients, clips parameter gradients at norm 5, steps Adam, then records pre-step train loss and scheduled post-step validation loss. The trial hook reports validation loss and can prune. Final MSE uses final weights. Every run returns or saves minimal metadata with configuration, seed, dependency versions, and effective hyperparameters.

## Required tests

Seeded short-run fixture comparison, optimizer-update count, validation at updates divisible by ten and final update, fresh-batch behavior, clip order, final-model evaluation, no-grad evaluation, trial report/prune behavior, metadata completeness and reload.

## Behavior-preservation checks

Keep the global training RNG stream and exact seed/init/data ordering. Preserve `evaluate`'s current unconditional return to training mode during extraction; its correction remains a separate follow-up task.

## Definition of Done

- [x] One reduced seeded training run matches the baseline fixture within declared tolerances and emits its minimal provenance record; no checkpoint-selection behavior has been added.

## Answer

Implemented `src/gradient_vanishing/training.py` with notebook-equivalent seeded updates, validation and final test MSE, pruning reports, and JSON-serializable run metadata. `tests/test_training.py` compares all three recorded three-update training cases with the immutable fixture and checks batch and split shapes, update order, validation cadence, no-grad evaluation, mode transitions, RNG repeatability, and provenance. The full suite passed: 68 tests, 0 failures. The source notebook and baseline fixture were not changed. The baseline does not record training-case parameter snapshots or final RNG states; a direct run of the audited notebook definitions matched the extracted module's final parameters and Torch RNG state exactly for all three models.
