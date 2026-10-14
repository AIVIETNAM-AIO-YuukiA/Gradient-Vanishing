# 12: Standardize configured commands and run manifests

Status: resolved
Type: task
Blocked by: 11

**Approved specification:** [Refactoring specification](../../../docs/refactoring_spec.md)
**Architecture:** [Architecture proposal](../../../docs/architecture_proposal.md)
**Source notebook:** [GradientVanishingAnalysis.ipynb](../../../Notebooks/GradientVanishingAnalysis.ipynb)

**Shared preservation:** Keep the source notebook unchanged at SHA-256 `71ec17cdb7fffcc39fef12bfe4977f75117ceedaa90c03109e79f51d98059f44`; do not silently fix documented bugs; treat its saved outputs as unverified observations.

## What to build

A researcher can invoke `main`, `gradients`, `ablation`, `sweep`, and `report` from a clean install and identify exactly what each run produced through one consistent manifest.

## Objective

A researcher can invoke `main`, `gradients`, `ablation`, `sweep`, and `report` from a clean install and identify exactly what each run produced through one consistent manifest.

## Scope

Thin command entry points, configuration and output selection, and a standardized schema that assembles the provenance already emitted by tickets 05–10. Finalize run status, result persistence, and report regeneration. The training heatmap uses main-run gradient histories; initial-gradient analysis, ablation, and sweep can run independently.

## Dependencies

11.

## Files or components affected

CLI adapter, run manifest and raw-result serialization, command tests, usage examples.

## Acceptance criteria

- [x] Each command accepts validated config and an output location; manifest consistently aggregates existing per-run provenance and records source revision, notebook hash, effective config, environment/device/determinism, seeds, selected parameters, command, and completion status; `report` reads existing results without training.

## Required tests

Reduced command invocations, invalid-config exit, incomplete/pruned run status, artifact reload and schema check, report-only command with training disabled, no import-time side effects.

## Behavior-preservation checks

Serialization and reporting do not consume RNG or change training order. CUDA determinism setting is applied before Torch import; unsupported deterministic operations surface errors.

## Definition of Done

- [x] All five commands run on reduced inputs and produce inspectable, reloadable artifacts; no full empirical reproduction claim is made.

## Answer

The `gradient-vanishing` entry point now runs `main`, `gradients`, `ablation`, `sweep`, and `report` from validated JSON settings. The main stage includes Optuna and saves each study separately. Every invocation creates a unique directory with atomic strict-JSON results and a run manifest recording effective configuration, overrides, provenance, source and result hashes, execution status, and available environment details. The report stage reads only saved results. See [command documentation](../../../docs/commands.md) for verified reduced CPU commands and the manifest contract. All five stages completed on a reduced local config; the full notebook matrix remains unexecuted.

Focused CLI tests: 8 passed. Full regression suite: 135 passed, 0 failed. Notebook and immutable fixture hashes remained unchanged.
Code review found and resolved a manifest mismatch for diagnostic seeds and stage-specific model and length selections.
A non-editable wheel import and CLI help were verified outside the checkout; Git revision is `null` there because it is unavailable from that installed artifact.
