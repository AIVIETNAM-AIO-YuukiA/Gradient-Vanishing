"""Data behavior is checked against the immutable original-notebook fixture."""

import hashlib
import json
from pathlib import Path

import torch

from gradient_vanishing.config import load_config
from gradient_vanishing.data import (
    MEMORYLESS_BASELINE,
    generate_memorization_task,
    get_eval_sets,
    make_fixed_set,
)


FIXTURE = Path(__file__).parent / "fixtures" / "notebook_baseline.json"


def _baseline_data():
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    return next(case for case in fixture["cases"] if case["id"] == "data_t10_seed7")


def _tensor_sha256(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def test_seeded_training_batch_matches_original_notebook_exactly():
    reference = _baseline_data()
    torch.manual_seed(7)
    assert _tensor_sha256(torch.get_rng_state()) == reference["input"]["rng_before_sha256"]
    x, y = generate_memorization_task(4, 10, device="cpu")
    assert x.tolist() == reference["expected"]["x"]["values"]
    assert y.tolist() == reference["expected"]["y"]["values"]
    assert list(x.shape) == [10, 4, 1]
    assert list(y.shape) == [4, 1]
    assert x.dtype == y.dtype == torch.float32
    assert _tensor_sha256(torch.get_rng_state()) == reference["expected"]["rng_after_sha256"]


def test_fixed_splits_match_original_without_consuming_training_rng():
    reference = _baseline_data()["expected"]["fixed_sets"]
    torch.manual_seed(123)
    state_before = torch.get_rng_state().clone()
    for split, seed in (("val", 10000), ("test", 20000)):
        x, y = make_fixed_set(2048, 10, seed, device="cpu")
        expected = reference[split]
        assert list(x.shape) == expected["x_shape"]
        assert list(y.shape) == expected["y_shape"]
        assert _tensor_sha256(x) == expected["x_sha256"]
        assert _tensor_sha256(y) == expected["y_sha256"]
        assert x[0, :4, 0].tolist() == expected["x_first_values"]
        assert y[:4, 0].tolist() == expected["y_first_values"]
        assert torch.equal(torch.get_rng_state(), state_before)


def test_per_length_fixed_set_cache_and_configuration():
    config = load_config()
    torch.manual_seed(123)
    state_before = torch.get_rng_state().clone()
    short = get_eval_sets(10, config, device="cpu")
    assert get_eval_sets(10, config, device="cpu") is short
    longer = get_eval_sets(30, config, device="cpu")
    assert longer is not short
    for split in ("val", "test"):
        short_x, short_y = short[split]
        long_x, long_y = longer[split]
        assert list(short_x.shape) == [10, 2048, 1]
        assert list(long_x.shape) == [30, 2048, 1]
        assert torch.equal(short_x[0], long_x[0])
        assert torch.equal(short_y, long_y)
    assert not torch.equal(short["val"][0], short["test"][0])
    assert torch.equal(torch.get_rng_state(), state_before)

    changed_config = load_config(overrides={"training": {"evaluation_size": 7},
                                            "split_seeds": {"validation": 30000}})
    changed = get_eval_sets(10, changed_config, device="cpu")
    assert changed is not short
    assert list(changed["val"][0].shape) == [10, 7, 1]
    assert not torch.equal(changed["val"][0][0], short["val"][0][0, :7])
    assert torch.equal(torch.get_rng_state(), state_before)
    assert MEMORYLESS_BASELINE == _baseline_data()["expected"]["memoryless_baseline"]


def test_fresh_batches_preserve_task_shape_range_and_target_copy():
    torch.manual_seed(7)
    first_x, first_y = generate_memorization_task(4, 10, device="cpu")
    second_x, _ = generate_memorization_task(4, 10, device="cpu")
    assert torch.all(first_x[1:] == 0)
    assert torch.equal(first_x[0], first_y)
    assert first_x[0].data_ptr() != first_y.data_ptr()
    assert torch.all((first_x[0] >= -1) & (first_x[0] <= 1))
    assert not torch.equal(first_x, second_x)
    torch.manual_seed(7)
    repeated_x, repeated_y = generate_memorization_task(4, 10, device="cpu")
    assert torch.equal(repeated_x, first_x)
    assert torch.equal(repeated_y, first_y)
    one_step_x, one_step_y = generate_memorization_task(2, 1, device="cpu")
    assert list(one_step_x.shape) == [1, 2, 1]
    assert torch.equal(one_step_x[0], one_step_y)


def test_explicit_generator_matches_fixture_without_advancing_global_rng():
    torch.manual_seed(99)
    state_before = torch.get_rng_state().clone()
    generator = torch.Generator().manual_seed(7)
    x, y = generate_memorization_task(4, 10, generator, device="cpu")
    reference = _baseline_data()["expected"]
    assert x.tolist() == reference["x"]["values"]
    assert y.tolist() == reference["y"]["values"]
    assert torch.equal(torch.get_rng_state(), state_before)
