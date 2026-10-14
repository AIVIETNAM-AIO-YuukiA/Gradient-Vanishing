# Scientific Research Methodology

This document details the mathematical models, algorithmic definitions, diagnostic quantities, and experimental design for the **Gradient-Vanishing** research project, faithfully preserving the scientific framework of [`Notebooks/GradientVanishingAnalysis.ipynb`](../Notebooks/GradientVanishingAnalysis.ipynb).

---

## 1. Research Problem and Dataset

### 1.1. The First-Element Memorization Task
The core problem investigates the capacity of recurrent networks to bridge long temporal delays. Synthetic sequences of length $T$ and batch size $B$ are generated as tensors $X \in \mathbb{R}^{T \times B \times 1}$:
* **Step $t = 0$:** $X[0, b, 0] \sim \mathcal{U}[-1, 1]$.
* **Steps $t = 1, \dots, T-1$:** $X[t, b, 0] = 0.0$.
* **Target:** $Y[b, 0] = X[0, b, 0]$.

To succeed, a recurrent model must propagate the scalar information introduced at $t=0$ through $T-1$ steps of zero inputs and reconstruct it from the final hidden state $h_{T-1}$.

### 1.2. Memoryless Baseline
For a predictor lacking temporal memory that uniformly outputs a constant prediction $\hat{y} = 0$, the expected Mean Squared Error (MSE) under the target distribution $Y \sim \mathcal{U}[-1, 1]$ is:
$$\mathbb{E}[(Y - 0)^2] = \int_{-1}^{1} y^2 \cdot \frac{1}{2} \, dy = \left[ \frac{y^3}{6} \right]_{-1}^{1} = \frac{1}{3} \approx 0.3333$$
Any model exhibiting a test MSE near $\approx 0.3333$ has failed to retain the initial scalar and behaves as a memoryless predictor.

### 1.3. Evaluation Set Caching and Seeding
Training draws fresh batches on-the-fly from PyTorch's global CPU random number generator after seeding. To ensure fair and deterministic evaluation across sequence lengths, fixed validation and test sets ($N = 2048$ samples each) are pre-generated using dedicated CPU generators with fixed seeds:
* Validation Seed: `10,000`
* Test Seed: `20,000`

---

## 2. Model Architectures

The project evaluates three recurrent architectures constructed from scratch, each emitting a prediction from its final hidden state: $\hat{y} = W_{out} h_{T-1} + b_{out}$.

### 2.1. Vanilla RNN
The hidden state update follows standard Elman recurrence:
$$h_t = \tanh(W_{ih} x_t + b_{ih} + W_{hh} h_{t-1} + b_{hh})$$

### 2.2. Long Short-Term Memory (LSTM)
The LSTM maintains both a hidden state $h_t$ and an internal cell state $c_t$:
$$\begin{aligned}
f_t &= \sigma(W_f x_t + U_f h_{t-1} + b_f) && \text{(Forget Gate)} \\
i_t &= \sigma(W_i x_t + U_i h_{t-1} + b_i) && \text{(Input Gate)} \\
\tilde{c}_t &= \tanh(W_c x_t + U_c h_{t-1} + b_c) && \text{(Candidate Cell State)} \\
c_t &= f_t \odot c_{t-1} + i_t \odot \tilde{c}_t && \text{(Cell State Update)} \\
o_t &= \sigma(W_o x_t + U_o h_{t-1} + b_o) && \text{(Output Gate)} \\
h_t &= o_t \odot \tanh(c_t) && \text{(Hidden State)}
\end{aligned}$$

### 2.3. Gated Recurrent Unit (GRU) & Gate Terminology
The GRU implementation uses the convention:
$$\begin{aligned}
r_t &= \sigma(W_r x_t + U_r h_{t-1} + b_r) && \text{(Reset Gate)} \\
z_t &= \sigma(W_z x_t + U_z h_{t-1} + b_z) && \text{(Update Gate)} \\
\tilde{h}_t &= \tanh(W_h x_t + U_h (r_t \odot h_{t-1}) + b_h) && \text{(Candidate Hidden State)} \\
h_t &= (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t && \text{(Hidden State Update)}
\end{aligned}$$

> [!IMPORTANT]
> **GRU Gate Terminology and Retention Bias:**
> In this project's formulation, $z_t$ blends the previous state with the candidate state. Setting $z_t \to 0$ forces $h_t \approx h_{t-1}$ (retention of previous state). Because $z_t = \sigma(W_z x_t + U_z h_{t-1} + b_z)$, a retention bias $b$ enters as:
> $$W_z.\text{bias} = -b$$
> Larger positive $b$ values decrease $z_t$, favoring memory retention. $W_r$ remains the Reset Gate and is untouched by retention initialization.

---

## 3. Backpropagation Through Time (BPTT) and Vanishing Gradients

During training, the loss function is the final-step mean squared error:
$$\mathcal{L} = \frac{1}{B} \sum_{b=1}^{B} (\hat{y}_b - y_b)^2$$
The gradient with respect to the initial hidden state $h_0$ expands via the chain rule across all $T$ timesteps:
$$\frac{\partial \mathcal{L}}{\partial h_0} = \frac{\partial \mathcal{L}}{\partial h_{T-1}} \prod_{t=1}^{T-1} \frac{\partial h_t}{\partial h_{t-1}}$$
In Vanilla RNNs, $\frac{\partial h_t}{\partial h_{t-1}} = \text{diag}(1 - h_t^2) W_{hh}^T$. If the eigenvalues of $W_{hh}$ or the saturation of $\tanh$ cause the spectral norm to satisfy $\left\| \frac{\partial h_t}{\partial h_{t-1}} \right\| < 1$, the gradient norm vanishes exponentially with sequence length $T$:
$$\left\| \frac{\partial \mathcal{L}}{\partial h_0} \right\| \propto \gamma^{T-1}, \quad \gamma < 1$$
LSTM and GRU mitigate this via additive state updates ($c_t = f_t \odot c_{t-1} + \dots$ and $h_t = (1-z_t) \odot h_{t-1} + \dots$), enabling gradient flow without exponential decay when gates favor retention.

---

## 4. Hidden-Gradient Measurement Definitions

To monitor vanishing gradients directly, the intermediate hidden state tensors $h_t$ register gradient retention hooks (`h_t.retain_grad()`) during forward propagation.

### 4.1. Hidden-State Gradient Norm ($g_t$)
After backpropagating from the final scalar loss $\mathcal{L}$, the diagnostic norm $g_t$ at timestep $t \in \{0, \dots, T-1\}$ is defined as the **batch-mean Euclidean norm**:
$$g_t = \frac{1}{B} \sum_{b=1}^{B} \left\| \frac{\partial \mathcal{L}}{\partial h_t[b, :]} \right\|_2$$
* Note: $g_t$ measures intermediate representations $h_t$, **not** parameter gradients $\frac{\partial \mathcal{L}}{\partial W}$.
* Full BPTT is unconstrained; no truncated window is applied.

### 4.2. Relative Normalized Gradient and Geometric Mean
For each seed, the gradient curve is normalized relative to the final timestep:
$$\rho_t = \frac{g_t}{g_{T-1}}$$
Across multiple seeds $s \in \{1, \dots, S\}$, the aggregated initial curve is the **geometric mean**:
$$\bar{\rho}_t = \exp\left( \frac{1}{S} \sum_{s=1}^{S} \log(\max(\rho_t^{(s)}, 10^{-300})) \right)$$

### 4.3. Exponential Decay Rate Estimation
To quantify vanishing gradients as a single scalar metric, an exponential decay curve $\rho_t \approx C \cdot \gamma^{T - 1 - t}$ is fitted via ordinary least squares in log-space over the latter half of the sequence:
$$t \in \left[ \lfloor T / 2 \rfloor, \, T - 2 \right]$$
Fitting $\log \rho_t = \alpha - \beta (T - 1 - t)$ yields the per-seed decay factor:
$$\gamma = \exp(-\beta)$$
The reported decay factor is the arithmetic mean and sample standard deviation ($\text{ddof}=1$) of $\gamma$ across seeds.

---

## 5. Training and Evaluation

### 5.1. Training Loop Specifications
* **Optimizer:** Adam with configured learning rate $\eta$.
* **Gradient Clipping:** Parameter gradients are clipped to maximum Euclidean norm $5.0$ (`torch.nn.utils.clip_grad_norm_`).
* **Batch Size:** 128 (default).
* **Budget:** 300 optimizer updates for final training; 150 updates for hyperparameter tuning trials.
* **Evaluation Interval:** Validation MSE is computed every 10 updates and at update 299 (the final step).

### 5.2. Evaluation Mode Restoration
The historical notebook contained a side effect where calling `evaluate()` temporarily set the model to `eval()` mode and unconditionally restored it to `train()`. In this package, the original training/eval state of the model is faithfully tracked and restored upon exit.

---

## 6. Optuna Hyperparameter Optimization

For each architecture and sequence length, hyperparameter optimization uses **Tree-structured Parzen Estimators (TPE)**:

| Hyperparameter | Search Domain | Scale |
|---|---|---|
| Learning Rate (`lr`) | $[10^{-4}, \, 5 \times 10^{-2}]$ | Logarithmic |
| Hidden Size (`hidden_size`) | $\{16, 32, 64\}$ | Categorical |
| Weight Initialization (`init_method`) | `{"orthogonal", "xavier", "default"}` | Categorical |

* **Sampler:** `TPESampler(seed=42)`.
* **Pruner:** `MedianPruner(n_startup_trials=5, n_warmup_steps=50)`.
* **Trials:** 20 trials per study.
* **Trial Seed:** $1000 + \text{trial.number}$.
* **Objective:** Final validation MSE after 150 updates (pruned trials terminate early).

---

## 7. Main Comparison Design

The main comparison executes a 2-factor grid across 3 sequence lengths:
* **Sequence Lengths:** $T \in \{10, 30, 100\}$.
* **Architectures:** Vanilla RNN, LSTM, GRU.
* **Configurations:**
  * **Default:** Fixed $\eta = 0.001$, $\text{hidden\_size} = 32$, default initialization, gate mode `one`.
  * **Tuned:** Architecture- and length-specific optimal parameters selected by the Optuna study, gate mode `one`.
* **Seeds:** 5 final training runs per condition ($\text{seeds} \in \{0, 1, 2, 3, 4\}$).
* **Metric:** Arithmetic mean and sample standard deviation ($\text{ddof}=1$) of final test MSE.

---

## 8. Gate Initialization Ablation ($T = 100$)

At sequence length $T=100$, the ablation isolates the impact of gate bias initialization on learning speed and gradient decay using fixed default hyperparameters ($\eta = 0.001$, hidden size 32):

| Condition | Architecture | Gate Mode | Bias Parameter Assignment |
|---|---|---|---|
| Reference | Vanilla RNN | `default` | Standard PyTorch initialization |
| 1 | LSTM | `zero` | $W_f.\text{bias} = 0.0$ |
| 2 | LSTM | `one` | $W_f.\text{bias} = 1.0$ |
| 3 | LSTM | `chrono` | $W_f.\text{bias} = \log U(1, T-1)$, $W_i.\text{bias} = -W_f.\text{bias}$ |
| 4 | GRU | `zero` | $W_z.\text{bias} = 0.0$ |
| 5 | GRU | `one` | $W_z.\text{bias} = -1.0$ |
| 6 | GRU | `chrono` | $W_z.\text{bias} = -\log U(1, T-1)$ |

* **Convergence Speed Metric:** Number of optimizer updates required to reach validation MSE $< 0.01$.
* **Initial Gradient Flow:** 10 float64 initial gradient measurements ($\text{seeds} \in \{0, \dots, 9\}$) to compute the initial decay factor $\gamma$.

---

## 9. Retention-Bias Sweep

To examine the continuous relationship between gate bias and initial gradient flow without training:
* **Sweep Range:** 17 uniform bias values $b \in [-4.0, +4.0]$ in increments of $0.5$.
* **Models:** LSTM ($W_f.\text{bias} = b$) and GRU ($W_z.\text{bias} = -b$).
* **Evaluation:** At sequence length $T=100$, 5 initial float64 gradient evaluations per bias level.
* **Theoretical Comparison:** The measured empirical decay $\gamma(b)$ is plotted against the theoretical state retention probability:
$$\text{Retention}(b) = \sigma(b) = \frac{1}{1 + e^{-b}}$$
* **Interpretation Note:** The line $y = x$ serves as a visual reference for the gate activation function, **not** an empirical claim that gradient decay must match the scalar sigmoid exactly across multidimensional hidden representations.
