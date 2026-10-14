# AI Coding Agent Instructions

This document defines core conventions, domain terminology, and engineering guardrails for AI coding assistants working in the **Gradient-Vanishing** repository.

---

## 1. Domain Terminology and Conventions

* **First-element memorization task:** Synthetic sequence regression where only timestep $t=0$ has a sampled value $X[0, b, 0] \sim \mathcal{U}[-1, 1]$, all later steps are zero, and the target is $Y[b, 0] = X[0, b, 0]$.
* **Memoryless baseline:** The theoretical MSE of predicting zero without memory: $\mathbb{E}[Y^2] = 1/3 \approx 0.3333$.
* **Retention bias ($b$):** Parameter initialization favoring temporal memory.
  * **LSTM:** Forget Gate bias $W_f.\text{bias} = b$. (Chrono also sets Input Gate bias $W_i.\text{bias} = -b$).
  * **GRU:** Update Gate bias $W_z.\text{bias} = -b$ under the update equation $h_t = (1-z_t)h_{t-1} + z_t \tilde{h}_t$. $W_r$ is the Reset Gate and remains untouched.
* **Configurations:**
  * `default`: Fixed $\eta = 0.001$, $\text{hidden\_size} = 32$, default initialization, gate mode `one`.
  * `tuned`: Architecture- and sequence-length-specific hyperparameters optimized via Optuna TPE search, gate mode `one`.

---

## 2. Scientific and Implementation Guardrails

* **Preserve Scientific Equations:** Never modify recurrent model equations in [`src/gradient_vanishing/models.py`](src/gradient_vanishing/models.py), synthetic task generation in [`data.py`](src/gradient_vanishing/data.py), or gradient norms in [`gradients.py`](src/gradient_vanishing/gradients.py).
* **Deterministic Seeding:** Reseeding order (Python, NumPy, PyTorch) before model construction and batch generation is critical to numerical determinism. Do not alter RNG call sequences.
* **Immutable Oracles:**
  * Reference notebook: [`Notebooks/GradientVanishingAnalysis.ipynb`](Notebooks/GradientVanishingAnalysis.ipynb) (SHA-256: `71ec17cdb7fffcc39fef12bfe4977f75117ceedaa90c03109e79f51d98059f44`).
  * Baseline fixture: [`tests/fixtures/notebook_baseline.json`](tests/fixtures/notebook_baseline.json) (SHA-256: `76e69564a5530b0a92e5cbac32b0b5ade492d516bd0e091c734c9e918b3dd336`).
* **Packaging and Wheel Provenance:** Do not remove [`setup.py`](setup.py). Its `BuildWithRevision` hook embeds `SOURCE_REVISION` into `gradient_vanishing/_build_info.py` during wheel builds.
* **Canonical Environment:** Canonical dependencies are pinned in [`requirements.lock`](requirements.lock) with **Optuna 5.0.0** and CPU PyTorch 2.11.0.

---

## 3. Active Documentation Structure

All active project documentation lives directly under `docs/`:

* [`docs/usage.md`](docs/usage.md): Installation instructions, CLI commands, configuration overrides, and reporting workflows.
* [`docs/methodology.md`](docs/methodology.md): Mathematical formulations, BPTT diagnostics, Optuna search spaces, ablation setups, and sweep definitions.
* [`docs/full_reproducibility_report.md`](docs/full_reproducibility_report.md): Definitive Ticket 14 empirical reproduction report, verification records, and artifact inventories.

---

## 4. Development History and Archival Reference

The initial 14-ticket refactoring, local issue tracker (`.scratch/`), and intermediate planning specifications are completed and durably preserved in Git history:
* **Archival Branch:** `archive/refactoring-docs-and-tickets`
* **Archival Tag:** `archive-refactor-history`
