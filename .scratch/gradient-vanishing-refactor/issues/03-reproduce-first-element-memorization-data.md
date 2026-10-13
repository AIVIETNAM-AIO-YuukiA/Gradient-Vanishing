# 03: Reproduce the first-element memorization data

Status: resolved
Type: task
Blocked by: 02

**Approved specification:** [Refactoring specification](../../../docs/refactoring_spec.md)  
**Architecture:** [Architecture proposal](../../../docs/architecture_proposal.md)  
**Source notebook:** [GradientVanishingAnalysis.ipynb](../../../Notebooks/GradientVanishingAnalysis.ipynb)

**Shared preservation:** Keep the source notebook unchanged at SHA-256 `71ec17cdb7fffcc39fef12bfe4977f75117ceedaa90c03109e79f51d98059f44`; do not silently fix documented bugs; treat its saved outputs as unverified observations.

## Objective

A researcher can request fresh training batches and fixed validation/test sets with the notebook's exact task semantics.

## Scope

Extract synthetic sequence generation, fixed-set generators, per-sequence-length caching, and theoretical memoryless baseline. Keep CPU sampling before device transfer.

## Dependencies

02.

## Files or components affected

Data module, data-contract tests, small data demonstration.

## Acceptance criteria

- [x] Inputs have shape `(T,B,1)` with only step zero sampled from `Uniform[-1,1]`; targets equal that first value and have shape `(B,1)`; validation/test sizes and seeds match the default config; same fixed-set seed pairs first values across lengths.

## Required tests

Shape, range, zero tail, target equality, split repeatability and separation, per-length cache, seeded small-case comparison to baseline fixture.

## Behavior-preservation checks

Training draws from the global Torch RNG after run seeding; fixed sets use separate CPU generators and do not consume that stream. No new data loader or dataset changes sampling order.

## Definition of Done

- [x] Data module can be called alone, tests pass, and fixture comparison shows no data-semantic change.

## Answer

The [data module](../../../src/gradient_vanishing/data.py) extracts only notebook cell 7's first-element memorization generator, fixed-set generator, evaluation-set cache, and theoretical memoryless baseline. The [tests](../../../tests/test_data.py) compare the default `T=10` samples, fixed validation/test tensors, and RNG state exactly with the immutable Ticket 02 fixture, and check shapes, dtype, range, zero tail, cloned target, fresh batches, split separation, cross-length pairing, cache behavior, and configuration overrides. A [small usage example](../../../docs/data.md) documents the public interface and the cache-key distinction required for configured runs.

Verification: the installed package imported and generated data outside the checkout; `python -m pytest -q` passed 26 tests in the pinned environment, Python compilation passed, and `pip check` found no broken requirements. The original notebook SHA-256 remains `71ec17cdb7fffcc39fef12bfe4977f75117ceedaa90c03109e79f51d98059f44`, and the immutable fixture SHA-256 remains `edc601e85522254471424940869598836cfea940a8d8711580bf8f0eca0818b9`. No model, training, or gradient logic was extracted. Standards and spec code review found no blocking issues.
