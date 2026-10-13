# 03: Reproduce the first-element memorization data

Status: claimed
Type: task
Blocked by: 02

**Approved specification:** [Refactoring specification](../../../docs/refactoring_spec.md)  
**Architecture:** [Architecture proposal](../../../docs/architecture_proposal.md)  
**Source notebook:** [GradientVanishingAnalysis.ipynb](../../../Notebooks/GradientVanishingAnalysis.ipynb)

**Shared preservation:** Keep the source notebook unchanged at SHA-256 `71ec17cdb7fffcc39fef12bfe4977f75117ceedaa90c03109e79f51d98059f44`; do not silently fix documented bugs; treat its saved outputs as unverified observations.

## What to build

A researcher can request fresh training batches and fixed validation/test sets with the notebook's exact task semantics.

## Objective

A researcher can request fresh training batches and fixed validation/test sets with the notebook's exact task semantics.

## Scope

Extract synthetic sequence generation, fixed-set generators, per-sequence-length caching, and theoretical memoryless baseline. Keep CPU sampling before device transfer.

## Dependencies

02.

## Files or components affected

Data module, data-contract tests, small data demonstration.

## Acceptance criteria

- [ ] Inputs have shape `(T,B,1)` with only step zero sampled from `Uniform[-1,1]`; targets equal that first value and have shape `(B,1)`; validation/test sizes and seeds match the default config; same fixed-set seed pairs first values across lengths.

## Required tests

Shape, range, zero tail, target equality, split repeatability and separation, per-length cache, seeded small-case comparison to baseline fixture.

## Behavior-preservation checks

Training draws from the global Torch RNG after run seeding; fixed sets use separate CPU generators and do not consume that stream. No new data loader or dataset changes sampling order.

## Definition of Done

- [ ] Data module can be called alone, tests pass, and fixture comparison shows no data-semantic change.
