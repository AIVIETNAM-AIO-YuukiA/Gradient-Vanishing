# 14: Run and report the full experiment matrix

Status: resolved
Type: task
Blocked by: 13

**Approved specification:** [Refactoring specification](../../../docs/refactoring_spec.md)

**Architecture:** [Architecture proposal](../../../docs/architecture_proposal.md)
**Source notebook:** [GradientVanishingAnalysis.ipynb](../../../Notebooks/GradientVanishingAnalysis.ipynb)

**Shared preservation:** Keep the source notebook unchanged at SHA-256 `71ec17cdb7fffcc39fef12bfe4977f75117ceedaa90c03109e79f51d98059f44`; do not silently fix documented bugs; treat its saved outputs as unverified observations.

## What to build

A reviewer can assess whether the modular project reproduces the notebook's empirical conclusions under a recorded environment.

## Objective

A reviewer can assess whether the modular project reproduces the notebook's empirical conclusions under a recorded environment.

## Scope

Full nine-study tuning matrix, 90 final main runs, 35 ablation training runs, initial-gradient analyses, numeric bias sweep, all reports, and a final reproducibility report. This is the sole ticket whose default path requires extensive computation.

## Dependencies

13.

## Files or components affected

Full-run artifacts, final report, discrepancy analysis; implementation modules change only if a verified regression is found.

## Acceptance criteria

- [x] The manifest distinguishes completed, pruned, failed, and unrun work; per-seed raw metrics and generated figures are available; the report compares new results with notebook saved outputs while stating that those outputs were not independently verified; any discrepancy is classified by code, environment, stochastic variation, or stale source output.

## Required tests

Re-run lightweight gates first, then verify matrix cardinalities, output completeness, summary arithmetic, test-set isolation, notebook hash, and artifact regeneration. Any corrected implementation must rerun its relevant lightweight and full checks.

## Behavior-preservation checks

No silent algorithm, gate, RNG, gradient, training, or evaluation changes are used to make numbers match. Cross-device bitwise equality is not required; only the pinned reference environment supports the direct numerical comparison.

## Definition of Done

- [x] A full empirical reproduction report is complete if the approved compute budget permits the run. Otherwise the ticket records the concrete hardware/runtime limit and marks empirical reproduction **unverified**, without presenting reduced results as full reproduction.

## Answer

The approved CPU matrix completed from wheel commit `8298ac766cf8af9a8316bb1d67d8174ed25546d3`: 180 Optuna trials (105 complete, 75 pruned), 90 main training runs, 35 ablation training runs, and 270 gradient-only runs. All stage and report artifact hashes were verified. The 136-test regression suite passed; a full 20-trial notebook study and one 300-update notebook ablation run matched the saved modular results exactly. Historical notebook tuned-output differences are documented without asserting universal numerical equivalence. See [the full reproducibility report](../../../docs/full_reproducibility_report.md) and the ignored, preserved artifacts at `.tmp/ticket14_run/full_matrix_20261014T215236/`.
