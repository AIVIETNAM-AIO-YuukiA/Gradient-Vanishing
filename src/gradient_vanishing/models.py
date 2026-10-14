"""Handwritten recurrent models and initialization from notebook cells 9 and 11.

All inputs are sequence-first ``(T, B, 1)``. The custom GRU uses the notebook's
blend ``h_t = (1-z_t) * h_(t-1) + z_t * h_tilde``: ``W_z`` computes the Update
Gate preactivation and ``W_r`` computes the Reset Gate preactivation. A positive
retention bias therefore enters ``W_z.bias`` with a negative sign. These are
the original equations, not PyTorch's built-in recurrent layer conventions.
"""

from collections.abc import Mapping
from typing import Any

import torch
from torch import nn


class TrackerBase(nn.Module):
    """Retain intermediate hidden gradients for the notebook diagnostic."""

    def _track(self, h: torch.Tensor) -> None:
        if h.requires_grad:
            h.retain_grad()
        self.h_states.append(h)

    def get_gradient_norms(self) -> list[float]:
        """Mean batch L2 norm of each tracked hidden state's gradient."""
        return [h.grad.norm(p=2, dim=1).mean().item() if h.grad is not None else 0.0
                for h in self.h_states]


class VanillaRNN_Tracker(TrackerBase):
    def __init__(self, input_size: int, hidden_size: int, output_size: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.W_ih = nn.Linear(input_size, hidden_size)
        self.W_hh = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_ho = nn.Linear(hidden_size, output_size)
        self.h_states: list[torch.Tensor] = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len, batch_size, _ = x.size()
        h = torch.zeros(batch_size, self.hidden_size, dtype=x.dtype, device=x.device)
        self.h_states = []
        for t in range(seq_len):
            h = torch.tanh(self.W_ih(x[t]) + self.W_hh(h))
            self._track(h)
        return self.W_ho(h)


class LSTM_Tracker(TrackerBase):
    def __init__(self, input_size: int, hidden_size: int, output_size: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        combined_size = input_size + hidden_size
        self.W_i = nn.Linear(combined_size, hidden_size)
        self.W_f = nn.Linear(combined_size, hidden_size)
        self.W_c = nn.Linear(combined_size, hidden_size)
        self.W_o = nn.Linear(combined_size, hidden_size)
        self.W_out = nn.Linear(hidden_size, output_size)
        self.h_states: list[torch.Tensor] = []
        self.c_states: list[torch.Tensor] = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len, batch_size, _ = x.size()
        h = torch.zeros(batch_size, self.hidden_size, dtype=x.dtype, device=x.device)
        c = torch.zeros(batch_size, self.hidden_size, dtype=x.dtype, device=x.device)
        self.h_states, self.c_states = [], []
        for t in range(seq_len):
            combined = torch.cat((x[t], h), dim=1)
            i = torch.sigmoid(self.W_i(combined))
            f = torch.sigmoid(self.W_f(combined))
            c_tilde = torch.tanh(self.W_c(combined))
            o = torch.sigmoid(self.W_o(combined))
            c = f * c + i * c_tilde
            h = o * torch.tanh(c)
            self._track(h)
            self.c_states.append(c.detach())
        return self.W_out(h)


class GRU_Tracker(TrackerBase):
    def __init__(self, input_size: int, hidden_size: int, output_size: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        combined_size = input_size + hidden_size
        self.W_z = nn.Linear(combined_size, hidden_size)
        self.W_r = nn.Linear(combined_size, hidden_size)
        self.W_h = nn.Linear(combined_size, hidden_size)
        self.W_out = nn.Linear(hidden_size, output_size)
        self.h_states: list[torch.Tensor] = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len, batch_size, _ = x.size()
        h = torch.zeros(batch_size, self.hidden_size, dtype=x.dtype, device=x.device)
        self.h_states = []
        for t in range(seq_len):
            combined = torch.cat((x[t], h), dim=1)
            z = torch.sigmoid(self.W_z(combined))
            r = torch.sigmoid(self.W_r(combined))
            h_tilde = torch.tanh(self.W_h(torch.cat((x[t], r * h), dim=1)))
            h = (1 - z) * h + z * h_tilde
            self._track(h)
        return self.W_out(h)


MODEL_CLASSES = {"Vanilla RNN": VanillaRNN_Tracker, "LSTM": LSTM_Tracker, "GRU": GRU_Tracker}


def apply_weight_init(model: TrackerBase, init_method: str) -> None:
    """Replace each matrix weight for orthogonal/Xavier modes; leave biases."""
    for name, parameter in model.named_parameters():
        if "weight" in name and parameter.dim() >= 2:
            if init_method == "orthogonal":
                nn.init.orthogonal_(parameter)
            elif init_method == "xavier":
                nn.init.xavier_uniform_(parameter)


def apply_gate_init(model: TrackerBase, mode: str | int | float, T: int) -> None:
    """Apply the notebook's retention-bias convention to LSTM or custom GRU."""
    if mode == "default" or isinstance(model, VanillaRNN_Tracker):
        return
    H = model.hidden_size
    if isinstance(mode, (int, float)):
        b = torch.full((H,), float(mode))
    elif mode == "zero":
        b = torch.zeros(H)
    elif mode == "one":
        b = torch.ones(H)
    elif mode == "chrono":
        b = torch.log(torch.empty(H).uniform_(1, max(T - 1, 2)))
    else:
        raise ValueError(f"gate init không hợp lệ: {mode}")

    with torch.no_grad():
        if isinstance(model, LSTM_Tracker):
            model.W_f.bias.copy_(b)
            if mode == "chrono":
                model.W_i.bias.copy_(-b)
        elif isinstance(model, GRU_Tracker):
            model.W_z.bias.copy_(-b)


def build_model(
    name: str,
    config: Mapping[str, Any],
    gate_init: str | int | float,
    T: int,
    dtype: torch.dtype = torch.float32,
    *,
    device: torch.device | str,
) -> TrackerBase:
    """Construct and initialize on CPU before transfer, matching notebook RNG order.

    ``config`` accepts the notebook hyperparameter keys ``hidden_size`` and
    ``init_method``. The experiment layer will map validated project config to
    these effective hyperparameters in a later ticket.
    """
    model = MODEL_CLASSES[name](1, config["hidden_size"], 1)
    apply_weight_init(model, config["init_method"])
    apply_gate_init(model, gate_init, T)
    return model.to(device=device, dtype=dtype)
