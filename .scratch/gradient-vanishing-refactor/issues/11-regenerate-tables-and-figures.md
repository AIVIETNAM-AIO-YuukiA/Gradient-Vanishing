# 11: Regenerate tables and figures from saved results

Status: resolved
Type: task
Blocked by: 08, 09, 10

**Approved specification:** [Refactoring specification](../../../docs/refactoring_spec.md)
**Architecture:** [Architecture proposal](../../../docs/architecture_proposal.md)
**Source notebook:** [GradientVanishingAnalysis.ipynb](../../../Notebooks/GradientVanishingAnalysis.ipynb)

**Shared preservation:** Keep the source notebook unchanged at SHA-256 `71ec17cdb7fffcc39fef12bfe4977f75117ceedaa90c03109e79f51d98059f44`; do not silently fix documented bugs; treat its saved outputs as unverified observations.

## What to build

A researcher can view the main comparison, gradient diagnostics, ablation, and bias sweep without rerunning models.

## Objective

A researcher can view the main comparison, gradient diagnostics, ablation, and bias sweep without rerunning models.

## Scope

Report the nine-row comparison table, log-scale mean validation curves, initial-gradient semilog curves, tuned seed-0 `T=100` heatmap, ablation table, and measured-versus-theoretical sweep plot from raw results.

## Dependencies

08, 09, and 10.

## Files or components affected

Reporting module, artifact schema validation, reporting tests.

## Acceptance criteria

- [x] All six notebook report types can be regenerated from persisted numeric inputs; mean/std uses `ddof=1`; the theoretical `1/3` line is distinguished from finite-sample test MSE; heatmap normalization and display clipping match the notebook.

## Required tests

Known synthetic result fixtures for table values, curve point alignment, log scaling, heatmap matrix and clip limits, sweep reference line, report regeneration without RNG consumption or training.

## Behavior-preservation checks

Keep the main learning curve mean-only during extraction, despite its misleading source comment; the uncertainty-band fix is a separate follow-up. Do not alter metrics merely to simplify plotting.

## Definition of Done

- [x] Reports regenerate from stored fixture results and match the specified visual/numeric semantics; raw result files remain unchanged.

## Answer

`src/gradient_vanishing/reporting.py` reads saved JSON result records and regenerates the main summary, mean-only log validation curves, initial-gradient semilog curves, tuned seed-zero training heatmap, ablation table, and numeric retention-bias plot. It also exports Optuna trial observations, numeric sweep plot inputs, and a source-hash report index. `docs/reporting.md` maps notebook cells to outputs and documents the API. No saved reduced experiment result files existed at implementation time, so renderer tests and representative artifacts use clearly marked synthetic saved records; the heatmap calculation additionally checks immutable training-gradient observations. Raw source records and RNG states remain unchanged. Full empirical reports await saved experiment results.

Code review found and resolved two partial-report issues: regeneration now removes artifacts from the previous report index, and a one-seed ablation decay spread is displayed as `NA`. The focused reporting suite passed (7 tests), and the full suite passed (127 tests, 0 failures).
