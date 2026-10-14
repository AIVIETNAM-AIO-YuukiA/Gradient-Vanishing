# 13: Validate the project end to end and document use

Status: resolved
Type: task
Blocked by: 12

**Approved specification:** [Refactoring specification](../../../docs/refactoring_spec.md)
**Architecture:** [Architecture proposal](../../../docs/architecture_proposal.md)
**Source notebook:** [GradientVanishingAnalysis.ipynb](../../../Notebooks/GradientVanishingAnalysis.ipynb)

**Shared preservation:** Keep the source notebook unchanged at SHA-256 `71ec17cdb7fffcc39fef12bfe4977f75117ceedaa90c03109e79f51d98059f44`; do not silently fix documented bugs; treat its saved outputs as unverified observations.

## What to build

A new researcher can install, run, inspect, and rerun a reduced experiment using only documented commands.

## Objective

A new researcher can install, run, inspect, and rerun a reduced experiment using only documented commands.

## Scope

Integration tests, pinned-CPU baseline comparisons, fresh-environment instructions, result interpretation, known-issue links, and a reduced end-to-end run through tune → train → evaluate → save → report.

## Dependencies

12.

## Files or components affected

Integration tests, research usage documentation, reproducibility checklist, reduced-run artifacts.

## Acceptance criteria

- [x] Reduced pipeline completes twice in the pinned CPU environment within declared tolerances; raw results and manifests agree; test data never reaches tuning; reports regenerate without retraining; every cell in the approved spec's coverage map remains accounted for.

## Required tests

Clean install, two seeded reduced runs, split-isolation check, source hash check, baseline fixture comparison, artifact round-trip, CLI/report smoke test. Document any environment-specific discrepancy.

## Behavior-preservation checks

Known bugs remain separately documented; no best-checkpoint selection or new algorithm enters integration. Treat saved notebook outputs only as unverified reference observations.

## Definition of Done

- [x] Lightweight acceptance gates pass and documentation lets another researcher repeat them. The result is called structurally validated, not yet empirically reproduced.

## Answer

An isolated Python 3.13 CPU environment installed all 38 pinned dependencies and a non-editable wheel; `pip check`, package import outside the checkout, both CLI entry points, and bundled defaults passed. Two real reduced pipelines covered all three models, three Optuna studies each, twelve final training runs, seven gate-ablation conditions, three numeric sweep biases for LSTM and GRU, initial gradients, saved manifests, and independent reports. Scientific result records and RNG digests repeated exactly after normalizing Optuna-generated study names; the only report byte difference was that generated name in the Optuna table. Real result hashes matched manifests and report indexes. A focused integration test passed under the wheel, and the full wheel-installed regression suite passed with 136 tests and 0 failures. [The validation report](../../../docs/end_to_end_validation.md) records commands, paths, environment, selected values, evidence levels, and NOT VERIFIED conditions. The notebook and immutable fixture hashes remain unchanged. The project is structurally validated; the full empirical matrix and GPU remain NOT VERIFIED.
