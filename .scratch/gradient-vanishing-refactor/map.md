# Gradient Vanishing refactor ticket map

The [approved specification](spec.md) and [ticket plan](../../docs/ticket_plan.md) define the full implementation sequence. All 14 issues are under [issues/](issues/).

| Ticket | Status | Result |
| --- | --- | --- |
| [01: Installable project and validated configuration](issues/01-installable-project-and-validated-configuration.md) | resolved | Installed package, pinned reference environment, validated notebook-equivalent configuration, and clean-environment checks. |
| [02: Capture immutable baseline fixtures](issues/02-capture-immutable-baseline-fixtures.md) | resolved | [22 directly executed notebook cases](../../tests/fixtures/notebook_baseline.json), source snapshot, provenance checks, and a reproducible capture command. |
| [03: Reproduce first-element memorization data](issues/03-reproduce-first-element-memorization-data.md) | ready-for-agent | Unblocked by Ticket 02. |
| [04: Extract recurrent models and gate initialization](issues/04-extract-recurrent-models-and-gate-initialization.md) | ready-for-agent | Unblocked by Ticket 02. |
