"""Independent equations and regression oracles for notebook cell 9/11 models."""

import hashlib
import json
import math
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from gradient_vanishing.data import generate_memorization_task
from gradient_vanishing.models import (
    GRU_Tracker, LSTM_Tracker, VanillaRNN_Tracker, apply_gate_init,
    apply_weight_init, build_model,
)


CASES = json.loads((Path(__file__).parent / "fixtures" / "notebook_baseline.json").read_text(encoding="utf-8"))["cases"]


def _tensor(record):
    return torch.tensor(record["values"], dtype=getattr(torch, record["dtype"].split(".")[-1])).reshape(record["shape"])


def _matches(actual, record, tolerance, *, exact=False):
    expected = _tensor(record)
    assert list(actual.shape) == record["shape"]
    assert actual.dtype == expected.dtype
    if exact:
        assert torch.equal(actual.detach().cpu(), expected)
    else:
        torch.testing.assert_close(actual.detach().cpu(), expected, **tolerance)


def _rng_digest():
    return hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest()


@pytest.mark.parametrize("case", [c for c in CASES if c["kind"] == "model"], ids=lambda c: c["id"])
def test_model_matches_executed_notebook(case):
    config = case["configuration"]
    tolerance = case["comparison"]["computed_values"]
    torch.manual_seed(case["seed"])
    model = build_model(config["model"], config["hyperparameters"], config["gate_mode"],
                        config["sequence_length"], device="cpu", dtype=getattr(torch, case["precision"]))
    assert _rng_digest() == case["expected"]["rng_after_model_sha256"]
    assert model.state_dict().keys() == case["expected"]["initial_parameters"].keys()
    for name, parameter in model.state_dict().items():
        _matches(parameter, case["expected"]["initial_parameters"][name], tolerance, exact=True)

    x, y = generate_memorization_task(config["batch_size"], config["sequence_length"], device="cpu")
    x = x.to(dtype=getattr(torch, case["precision"])).requires_grad_()
    y = y.to(dtype=x.dtype)
    _matches(x, case["input"]["x"], tolerance, exact=True)
    _matches(y, case["input"]["y"], tolerance, exact=True)
    gate_outputs = {name: [] for name in ("W_z", "W_r", "W_h")}
    handles = []
    if config["model"] == "GRU":
        for name in gate_outputs:
            handles.append(getattr(model, name).register_forward_hook(
                lambda _module, _inputs, output, key=name: gate_outputs[key].append(output.detach())))
    prediction = model(x)
    for handle in handles:
        handle.remove()
    loss = F.mse_loss(prediction, y)
    loss.backward()
    expected = case["expected"]
    _matches(prediction, expected["prediction"], tolerance)
    assert math.isclose(loss.item(), expected["mse"], **dict(rel_tol=tolerance["rtol"], abs_tol=tolerance["atol"]))
    torch.testing.assert_close(torch.tensor(model.get_gradient_norms()),
                               torch.tensor(expected["hidden_gradient_norms"]), **tolerance)
    _matches(x.grad[0], expected["input_gradient_first_step"], tolerance)
    _matches(x.grad[-1], expected["input_gradient_last_step"], tolerance)
    _matches(next(model.parameters()).grad, expected["first_parameter_gradient"], tolerance)
    assert len(model.h_states) == config["sequence_length"]
    assert all(state.requires_grad and state.grad is not None for state in model.h_states)
    assert model.training is expected["training_mode_after_backward"]
    if config["model"] == "LSTM":
        observed = expected["cell_state_observations"]
        assert len(model.c_states) == config["sequence_length"]
        assert all(not cell.requires_grad for cell in model.c_states) is observed["stored_states_detached"]
        _matches(model.c_states[0], observed["first"], tolerance)
        _matches(model.c_states[-1], observed["last"], tolerance)
    if config["model"] == "GRU":
        for name, observed in expected["gate_observations"].items():
            _matches(gate_outputs[name][0], observed["first_preactivation"], tolerance)
            _matches(gate_outputs[name][-1], observed["last_preactivation"], tolerance)
            activation = torch.tanh(gate_outputs[name][0]) if name == "W_h" else torch.sigmoid(gate_outputs[name][0])
            _matches(activation, observed["first_activation"], tolerance)


@pytest.mark.parametrize("case", [c for c in CASES if c["kind"] == "gate_initialization"], ids=lambda c: c["id"])
def test_gate_initialization_matches_executed_notebook(case):
    config = case["configuration"]
    torch.manual_seed(case["seed"])
    model = build_model(config["model"], config["hyperparameters"], config["retention_bias_mode"],
                        config["sequence_length"], device="cpu")
    assert _rng_digest() == case["expected"]["rng_after_model_sha256"]
    target = model.W_z if config["model"] == "GRU" else model.W_f
    tol = case["comparison"]["computed_values"]
    _matches(target.weight, case["expected"]["target_gate_weight"], tol, exact=True)
    _matches(target.bias, case["expected"]["target_gate_bias"], tol, exact=True)
    if config["model"] == "GRU":
        _matches(model.W_r.bias, case["expected"]["reset_gate_bias"], tol, exact=True)
    else:
        _matches(model.W_i.bias, case["expected"]["input_gate_bias"], tol, exact=True)


def _linear(layer, weights, bias=None):
    with torch.no_grad():
        layer.weight.copy_(torch.tensor([weights], dtype=layer.weight.dtype))
        if bias is not None:
            layer.bias.fill_(bias)


def test_vanilla_rnn_two_step_equation_from_zero_state():
    model = VanillaRNN_Tracker(1, 1, 1).double()
    _linear(model.W_ih, [2.0], 0.3)
    _linear(model.W_hh, [0.8])
    _linear(model.W_ho, [1.5], -0.2)
    actual = model(torch.tensor([[[0.4]], [[-0.25]]], dtype=torch.float64))
    first_hidden = math.tanh(2.0 * 0.4 + 0.3)
    expected = 1.5 * math.tanh(2.0 * -0.25 + 0.8 * first_hidden + 0.3) - 0.2
    assert actual.shape == (1, 1)
    assert actual.item() == pytest.approx(expected, abs=1e-12)


def test_lstm_two_step_equation_from_zero_states():
    model = LSTM_Tracker(1, 1, 1).double()
    for layer, w, b in ((model.W_i, [0.7, 0.2], -0.1), (model.W_f, [-0.3, 0.8], 0.2),
                        (model.W_c, [1.1, -0.4], 0.05), (model.W_o, [0.9, 0.3], -0.15)):
        _linear(layer, w, b)
    _linear(model.W_out, [1.7], 0.25)
    sigmoid = lambda v: 1 / (1 + math.exp(-v))
    hidden, cell = 0.0, 0.0
    for x in (0.6, -0.2):
        input_gate = sigmoid(0.7*x + 0.2*hidden - 0.1)
        forget_gate = sigmoid(-0.3*x + 0.8*hidden + 0.2)
        candidate = math.tanh(1.1*x - 0.4*hidden + 0.05)
        output_gate = sigmoid(0.9*x + 0.3*hidden - 0.15)
        cell = forget_gate*cell + input_gate*candidate
        hidden = output_gate*math.tanh(cell)
    expected = 1.7*hidden + 0.25
    actual = model(torch.tensor([[[0.6]], [[-0.2]]], dtype=torch.float64))
    assert actual.item() == pytest.approx(expected, abs=1e-12)
    assert model.c_states[0].requires_grad is False


def test_custom_gru_two_step_equation_uses_update_and_reset_gates():
    model = GRU_Tracker(1, 1, 1).double()
    _linear(model.W_z, [1.2, -0.7], 0.3)
    _linear(model.W_r, [-0.4, 0.9], -0.2)
    _linear(model.W_h, [0.8, 1.1], 0.1)
    _linear(model.W_out, [1.4], -0.05)
    sigmoid = lambda v: 1 / (1 + math.exp(-v))
    h = 0.0
    for x in (0.5, -0.3):
        z = sigmoid(1.2*x-0.7*h+0.3)
        r = sigmoid(-0.4*x+0.9*h-0.2)
        candidate = math.tanh(0.8*x+1.1*r*h+0.1)
        h = (1-z)*h+z*candidate
    actual = model(torch.tensor([[[0.5]], [[-0.3]]], dtype=torch.float64))
    assert actual.item() == pytest.approx(1.4*h-0.05, abs=1e-12)
    assert len(model.h_states) == 2


@pytest.mark.parametrize("cls", [VanillaRNN_Tracker, LSTM_Tracker, GRU_Tracker])
@pytest.mark.parametrize("method", ["orthogonal", "xavier"])
def test_weight_initialization_scope_preserves_biases(cls, method):
    torch.manual_seed(73)
    model = cls(1, 4, 1)
    before = {name: value.detach().clone() for name, value in model.named_parameters()}
    apply_weight_init(model, method)
    for name, value in model.named_parameters():
        if "weight" in name and value.ndim >= 2:
            assert not torch.equal(value, before[name])
            if method == "orthogonal":
                gram = value @ value.T if value.shape[0] <= value.shape[1] else value.T @ value
                torch.testing.assert_close(gram, torch.eye(gram.shape[0]), atol=1e-6, rtol=1e-6)
        else:
            assert torch.equal(value, before[name])


def test_gate_modes_have_original_retention_sign_and_no_reset_gate_change():
    for mode, retention in (("zero", 0.0), ("one", 1.0), (2.0, 2.0)):
        torch.manual_seed(19)
        model = GRU_Tracker(1, 3, 1)
        reset_before = model.W_r.bias.detach().clone()
        update_weight_before = model.W_z.weight.detach().clone()
        apply_gate_init(model, mode, 30)
        torch.testing.assert_close(model.W_z.bias, torch.full_like(model.W_z.bias, -retention))
        assert torch.equal(model.W_r.bias, reset_before)
        assert torch.equal(model.W_z.weight, update_weight_before)


def test_lstm_non_chrono_modes_only_assign_forget_bias():
    for mode, retention in (("zero", 0.0), ("one", 1.0), (2.0, 2.0)):
        torch.manual_seed(19)
        model = LSTM_Tracker(1, 3, 1)
        input_before = model.W_i.bias.detach().clone()
        forget_weight_before = model.W_f.weight.detach().clone()
        apply_gate_init(model, mode, 30)
        torch.testing.assert_close(model.W_f.bias, torch.full_like(model.W_f.bias, retention))
        assert torch.equal(model.W_i.bias, input_before)
        assert torch.equal(model.W_f.weight, forget_weight_before)


def test_lstm_chrono_pairs_forget_and_input_bias_without_detaching_live_cell():
    torch.manual_seed(21)
    model = LSTM_Tracker(1, 3, 1)
    apply_gate_init(model, "chrono", 30)
    torch.testing.assert_close(model.W_i.bias, -model.W_f.bias)
    x = torch.tensor([[[0.5]], [[0.0]]], requires_grad=True)
    model(x).sum().backward()
    assert x.grad[0].abs().sum().item() > 0
    assert all(not cell.requires_grad for cell in model.c_states)


@pytest.mark.parametrize("cls", [VanillaRNN_Tracker, LSTM_Tracker, GRU_Tracker])
def test_forward_resets_observations_and_no_grad_does_not_retain(cls):
    model = cls(1, 2, 1)
    model(torch.zeros(3, 2, 1))
    assert len(model.h_states) == 3
    with torch.no_grad():
        result = model(torch.zeros(2, 2, 1))
    assert result.shape == (2, 1)
    assert len(model.h_states) == 2
    assert all(not state.requires_grad for state in model.h_states)
    assert model.get_gradient_norms() == [0.0, 0.0]
    if isinstance(model, LSTM_Tracker):
        assert len(model.c_states) == 2
