# Issue tracker: Local Markdown

Issues and specs for this repo live as Markdown files under `.scratch/`.

## Conventions

- One feature per directory: `.scratch/<feature-slug>/`.
- The spec is `.scratch/<feature-slug>/spec.md`.
- Implementation tickets are separate files at `.scratch/<feature-slug>/issues/<NN>-<slug>.md`, numbered from `01`.
- Record triage state as `Status: <label>` near the top of each issue.
- Append discussion under a `## Comments` heading.

## Skill operations

- Publish a spec or ticket by creating its file under the feature directory.
- Fetch a ticket by reading its referenced path or numbered file.
- Use `docs/agents/triage-labels.md` for status strings.

## Wayfinding

A map lives at `.scratch/<effort>/map.md`; child tickets live in its `issues/` directory. Record a child type as `Type: research`, `prototype`, `grilling`, or `task`. Record blockers as `Blocked by: NN, NN`. A child is available when its blockers are resolved and it is unclaimed. Claim it with `Status: claimed`; resolve it with `Status: resolved`, an `## Answer` section, and a pointer added to the map.
