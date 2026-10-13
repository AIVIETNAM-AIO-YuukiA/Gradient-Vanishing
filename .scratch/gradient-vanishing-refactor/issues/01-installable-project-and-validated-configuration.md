# 01: Installable project and validated experiment configuration

Status: claimed
Blocked by: None (can start immediately)

**Approved specification:** [Refactoring specification](../../../docs/refactoring_spec.md)  
**Architecture:** [Architecture proposal](../../../docs/architecture_proposal.md)  
**Source notebook:** [GradientVanishingAnalysis.ipynb](../../../Notebooks/GradientVanishingAnalysis.ipynb)

**Shared preservation:** Keep the source notebook unchanged at SHA-256 `71ec17cdb7fffcc39fef12bfe4977f75117ceedaa90c03109e79f51d98059f44`; do not silently fix documented bugs; treat its saved outputs as unverified observations.

## What to build

A researcher can install the project in a pinned reference environment and validate a versioned JSON configuration without executing the notebook.

## Objective

A researcher can install the project in a pinned reference environment and validate a versioned JSON configuration without executing the notebook.

## Scope

Package metadata, a committed resolved dependency manifest for the reference CPU environment, a default JSON configuration matching notebook constants, validation and override precedence, and a small configuration check. Document CUDA as a separate environment if supported.

## Dependencies

None; can start immediately.

## Files or components affected

Package metadata, dependency manifest, configuration loader, default configuration, configuration tests, installation instructions.

## Acceptance criteria

- [ ] Clean install/import succeeds; unknown keys and invalid combinations fail clearly; command override takes precedence over configuration value and checked-in default; baseline values include the three sequence lengths, model names, seeds, update budgets, data sizes, search space, and gate mode `one`.

## Required tests

Load/default equality, accepted override, unknown-key rejection, invalid-type/range rejection, clean import without training or plotting.

## Behavior-preservation checks

No source notebook edit or runtime `pip install`; configuration defaults reproduce its constants. Deterministic CUDA setup must occur before Torch import when requested.

## Definition of Done

- [ ] Environment and configuration checks pass in a clean install; notebook hash is unchanged; dependency versions are recorded. No model implementation is extracted yet.
