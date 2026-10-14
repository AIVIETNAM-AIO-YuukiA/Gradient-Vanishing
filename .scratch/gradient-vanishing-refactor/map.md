# Gradient Vanishing refactor ticket map

The [approved specification](spec.md) and [ticket plan](../../docs/ticket_plan.md) define the full implementation sequence. All 14 issues are under [issues/](issues/).

| Ticket | Status | Result |
| --- | --- | --- |
| [01: Installable project and validated configuration](issues/01-installable-project-and-validated-configuration.md) | resolved | Installed package, pinned reference environment, validated notebook-equivalent configuration, and clean-environment checks. |
| [02: Capture immutable baseline fixtures](issues/02-capture-immutable-baseline-fixtures.md) | resolved | [22 directly executed notebook cases](../../tests/fixtures/notebook_baseline.json), source snapshot, provenance checks, and a reproducible capture command. |
| [03: Reproduce first-element memorization data](issues/03-reproduce-first-element-memorization-data.md) | resolved | [Data module](../../src/gradient_vanishing/data.py) matches the immutable notebook fixture and preserves the global training RNG and independent fixed-set generators. |
| [04: Extract recurrent models and gate initialization](issues/04-extract-recurrent-models-and-gate-initialization.md) | resolved | [Handwritten models and initialization](../../src/gradient_vanishing/models.py) match the immutable notebook fixtures and independent recurrent equation tests. |
| [05: Reproduce seeded training and evaluation](issues/05-reproduce-seeded-training-and-evaluation.md) | resolved | [Single-run training](../../src/gradient_vanishing/training.py) matches all three reduced notebook training probes and records minimal provenance. |
| [06: Reproduce hidden-gradient diagnostics and full BPTT checks](issues/06-reproduce-hidden-gradient-diagnostics.md) | resolved | [Gradient diagnostics](../../src/gradient_vanishing/gradients.py) match stored and direct notebook references and preserve the hidden-state metric. |
| [07: Reproduce one Optuna study](issues/07-reproduce-optuna-study.md) | resolved | [Optuna orchestration](../../src/gradient_vanishing/experiments.py) matches a reduced direct-notebook study and records trial and selection provenance. |
| [08: Reproduce the default-versus-tuned main comparison](issues/08-reproduce-main-comparison.md) | resolved | [Main comparison orchestration](../../src/gradient_vanishing/experiments.py) matches a reduced direct-notebook run and records raw per-seed results. |
