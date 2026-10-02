"""CPU optimizer-update traces: strict LR options and unchanged legacy schedule."""

import json
import math
import os
import subprocess
import sys
import warnings
from copy import deepcopy
from pathlib import Path

import pytest
import torch

from qwen21_trainer.lr_schedule import (
    build_lr_scheduler,
    lr_scheduler_report,
    normalize_lr_options,
)


def settings(**overrides):
    result = {"max_steps": 8, "learning_rate": 0.12}
    result.update(overrides)
    return result


def run_updates(options, *, accumulation=1, reference_factory=None, initial_value=1.0):
    """Exercise real gradients/CPU SGD; record used LR before optimizer.step()."""
    parameter = torch.nn.Parameter(torch.tensor(initial_value, dtype=torch.float64))
    optimizer = torch.optim.SGD([parameter], lr=options["learning_rate"])
    scheduler = (
        build_lr_scheduler(optimizer, options)
        if reference_factory is None
        else reference_factory(optimizer)
    )
    rows = []
    forwards = 0
    optimizer.zero_grad(set_to_none=True)
    for _ in range(options["max_steps"]):
        before = parameter.item()
        for _ in range(accumulation):
            loss = (parameter - 3.0).square() / accumulation
            assert torch.isfinite(loss)
            loss.backward()
            forwards += 1
        assert torch.isfinite(parameter.grad)
        used_lr = optimizer.param_groups[0]["lr"]
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        assert torch.isfinite(parameter)
        rows.append((used_lr, before, parameter.item()))
    return rows, scheduler, optimizer, forwards


def test_defaults_copy_without_mutating_nested_settings():
    original = settings(optimizer_params={"betas": [0.9, 0.999]})
    snapshot = deepcopy(original)
    actual = normalize_lr_options(original)
    assert actual["lr_scheduler"] == "diffsynth"
    assert actual["num_warmup"] == 0
    actual["optimizer_params"]["betas"][0] = 0.5
    assert original == snapshot


@pytest.mark.parametrize("name", ["diffsynth", "constant", "constant_with_warmup", "cosine"])
def test_missing_warmup_defaults_to_zero(name):
    actual = normalize_lr_options(settings(lr_scheduler=name))
    assert actual["num_warmup"] == 0
    assert actual["lr_scheduler"] == name


@pytest.mark.parametrize("value", [None, True, 3, [], {}, "", "linear", "Cosine", "cosine "])
def test_invalid_scheduler_is_rejected(value):
    with pytest.raises(ValueError, match="lr_scheduler"):
        normalize_lr_options(settings(lr_scheduler=value))


@pytest.mark.parametrize("value", [True, False, -1, 1.0, "1", None, [], {}, float("nan")])
def test_invalid_warmup_is_rejected(value):
    with pytest.raises(ValueError, match="num_warmup"):
        normalize_lr_options(settings(lr_scheduler="cosine", num_warmup=value))


@pytest.mark.parametrize("value", [True, False, 0, -1, 8.0, "8", None, [], float("inf")])
def test_invalid_optimizer_update_budget_is_rejected(value):
    with pytest.raises(ValueError, match="max_steps"):
        normalize_lr_options(settings(max_steps=value))


def test_missing_optimizer_update_budget_is_rejected():
    with pytest.raises(ValueError, match="max_steps"):
        normalize_lr_options({})


@pytest.mark.parametrize("value", [None, True, [], "constant"])
def test_non_mapping_options_are_rejected(value):
    with pytest.raises(ValueError, match="training"):
        normalize_lr_options(value)


@pytest.mark.parametrize("name", ["constant", "diffsynth"])
def test_nonwarmup_schedules_reject_warmup_instead_of_ignoring_it(name):
    with pytest.raises(ValueError, match="num_warmup=0"):
        normalize_lr_options(settings(lr_scheduler=name, num_warmup=1))


@pytest.mark.parametrize("name", ["diffsynth", "constant", "constant_with_warmup", "cosine"])
def test_warmup_beyond_total_budget_is_rejected(name):
    with pytest.raises(ValueError, match="num_warmup"):
        normalize_lr_options(settings(lr_scheduler=name, num_warmup=9))


def test_cosine_requires_at_least_one_decay_step():
    with pytest.raises(ValueError, match="小於"):
        normalize_lr_options(settings(lr_scheduler="cosine", num_warmup=8))


def test_constant_warmup_can_span_the_entire_budget():
    options = settings(lr_scheduler="constant_with_warmup", num_warmup=8)
    rows, scheduler, optimizer, _ = run_updates(options)
    assert [row[0] for row in rows] == pytest.approx([0.12 * step / 8 for step in range(8)])
    assert rows[-1][0] < options["learning_rate"]
    assert scheduler.last_epoch == options["max_steps"]
    assert optimizer.param_groups[0]["lr"] == options["learning_rate"]


def test_validation_happens_before_mutating_optimizer():
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.SGD([parameter], lr=0.12)
    before = deepcopy(optimizer.state_dict())
    with pytest.raises(ValueError, match="num_warmup=0"):
        build_lr_scheduler(optimizer, settings(lr_scheduler="constant", num_warmup=1))
    assert optimizer.state_dict() == before
    assert "initial_lr" not in optimizer.param_groups[0]


@pytest.mark.parametrize("explicit", [False, True])
@pytest.mark.parametrize("budget", [1, 4, 5, 6, 12])
def test_legacy_is_exactly_the_old_constantlr_for_lr_and_parameters(explicit, budget):
    options = settings(max_steps=budget)
    if explicit:
        options["lr_scheduler"] = "diffsynth"
    actual, scheduler, optimizer, _ = run_updates(options)
    expected, reference, reference_optimizer, _ = run_updates(
        options, reference_factory=torch.optim.lr_scheduler.ConstantLR
    )
    # Equality is intentionally exact, including floating-point multiplication
    # at the five-update transition, not just approximately the same formula.
    assert actual == expected
    assert scheduler.state_dict() == reference.state_dict()
    assert optimizer.param_groups[0]["lr"] == reference_optimizer.param_groups[0]["lr"]
    assert [row[0] for row in actual] == pytest.approx(
        [
            options["learning_rate"] / 3 if step < 5 else options["learning_rate"]
            for step in range(budget)
        ]
    )


@pytest.mark.parametrize("name", ["constant", "constant_with_warmup"])
def test_true_constant_and_zero_warmup_use_base_lr_from_first_to_last(name):
    options = settings(lr_scheduler=name, num_warmup=0)
    rows, scheduler, optimizer, _ = run_updates(options)
    assert [row[0] for row in rows] == [options["learning_rate"]] * options["max_steps"]
    assert all(before != after for _, before, after in rows)
    assert optimizer.param_groups[0]["lr"] == options["learning_rate"]
    assert scheduler.last_epoch == options["max_steps"]


def test_linear_warmup_exact_used_lr_sequence():
    options = settings(lr_scheduler="constant_with_warmup", num_warmup=3)
    rows, scheduler, _, _ = run_updates(options)
    expected = [0.0, 0.12 / 3, 0.12 * 2 / 3] + [0.12] * 5
    assert [row[0] for row in rows] == pytest.approx(expected, abs=1e-15)
    assert rows[0][1] == rows[0][2]  # Real first optimizer step uses zero LR.
    assert all(before != after for _, before, after in rows[1:])
    assert scheduler.last_epoch == options["max_steps"]


@pytest.mark.parametrize("warmup,budget", [(0, 1), (0, 8), (1, 8), (2, 8), (7, 8)])
def test_half_cosine_full_curve_and_used_final_lr(warmup, budget):
    options = settings(lr_scheduler="cosine", num_warmup=warmup, max_steps=budget)
    rows, scheduler, optimizer, _ = run_updates(options)
    factors = [
        step / warmup
        if step < warmup
        else 0.5 * (1.0 + math.cos(math.pi * (step - warmup) / (budget - warmup)))
        for step in range(budget)
    ]
    assert [row[0] for row in rows] == pytest.approx([0.12 * factor for factor in factors])
    assert rows[-1][0] > 0  # Final update uses curve(N-1), not curve(N).
    assert optimizer.param_groups[0]["lr"] == 0.0
    assert scheduler.get_last_lr() == [0.0]
    assert scheduler.last_epoch == budget
    if warmup:
        assert rows[0][0] == 0.0
        assert rows[0][1] == rows[0][2]
    decay_lrs = [row[0] for row in rows[warmup:]]
    assert decay_lrs == sorted(decay_lrs, reverse=True)


@pytest.mark.parametrize(
    "name,warmup",
    [("constant", 0), ("constant_with_warmup", 0), ("constant_with_warmup", 3), ("cosine", 3)],
)
def test_standard_schedules_match_installed_hf_reference(name, warmup):
    from transformers.optimization import (
        get_constant_schedule,
        get_constant_schedule_with_warmup,
        get_cosine_schedule_with_warmup,
    )

    options = settings(lr_scheduler=name, num_warmup=warmup)

    def reference(optimizer):
        if name == "constant":
            return get_constant_schedule(optimizer)
        if name == "constant_with_warmup":
            return get_constant_schedule_with_warmup(optimizer, num_warmup_steps=warmup)
        return get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup, num_training_steps=8, num_cycles=0.5
        )

    actual, _, optimizer, _ = run_updates(options)
    expected, _, reference_optimizer, _ = run_updates(options, reference_factory=reference)
    assert actual == expected
    assert optimizer.param_groups[0]["lr"] == reference_optimizer.param_groups[0]["lr"]


@pytest.mark.parametrize(
    "name,warmup",
    [("diffsynth", 0), ("constant", 0), ("constant_with_warmup", 2), ("cosine", 2)],
)
@pytest.mark.parametrize("accumulation", [1, 3, 5])
def test_accumulation_does_not_advance_schedule_on_microsteps(name, warmup, accumulation):
    options = settings(lr_scheduler=name, num_warmup=warmup)
    with warnings.catch_warnings(record=True) as captured:
        rows, scheduler, _, forwards = run_updates(options, accumulation=accumulation)
    assert not [w for w in captured if "lr_scheduler.step()" in str(w.message)]
    baseline, _, _, _ = run_updates(options)
    assert [row[0] for row in rows] == [row[0] for row in baseline]
    assert len(rows) == options["max_steps"]
    assert forwards == options["max_steps"] * accumulation
    assert scheduler.last_epoch == options["max_steps"]
    assert rows[-1][2] != rows[0][1]
    assert rows[-1][2] == pytest.approx(baseline[-1][2])


@pytest.mark.parametrize("name,warmup", [("diffsynth", 0), ("constant", 0), ("cosine", 2)])
def test_multiple_parameter_groups_keep_their_base_lr_ratios(name, warmup):
    first = torch.nn.Parameter(torch.tensor(1.0))
    second = torch.nn.Parameter(torch.tensor(2.0))
    optimizer = torch.optim.SGD([{"params": [first], "lr": 0.12}, {"params": [second], "lr": 0.06}])
    scheduler = build_lr_scheduler(optimizer, settings(lr_scheduler=name, num_warmup=warmup))
    for _ in range(8):
        assert optimizer.param_groups[0]["lr"] == 2 * optimizer.param_groups[1]["lr"]
        (first.square() + second.square()).backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
    assert torch.isfinite(first) and torch.isfinite(second)


def test_weight_only_warm_start_builds_a_new_schedule():
    options = settings(lr_scheduler="cosine", num_warmup=2)
    previous, _, _, _ = run_updates(options)
    restarted, scheduler, _, _ = run_updates(options, initial_value=previous[-1][2])
    assert restarted[0][1] == previous[-1][2]
    assert restarted[0][0] == 0.0
    assert [row[0] for row in restarted] == [row[0] for row in previous]
    assert scheduler.last_epoch == options["max_steps"]


@pytest.mark.parametrize(
    "name,warmup",
    [("diffsynth", 0), ("constant", 0), ("constant_with_warmup", 2), ("cosine", 2)],
)
def test_report_is_json_safe_and_describes_effective_schedule(name, warmup):
    report = lr_scheduler_report(settings(lr_scheduler=name, num_warmup=warmup))
    assert json.loads(json.dumps(report, allow_nan=False)) == report
    assert report["name"] == name
    assert report["num_warmup"] == warmup
    assert report["max_steps"] == 8
    assert report["last_update_schedule_step"] == 7
    assert report["after_last_update_schedule_step"] == 8
    assert report["step_unit"] == "optimizer_update"
    assert report["warm_start"] == "reset_optimizer_and_scheduler"
    assert report["num_cycles"] == (0.5 if name == "cosine" else None)
    assert report["zero_lr_at_schedule_step"] == (8 if name == "cosine" else None)
    assert report["legacy_factor"] == (1.0 / 3.0 if name == "diffsynth" else None)
    assert report["legacy_total_iters"] == (5 if name == "diffsynth" else None)
    expected_first = 1.0 / 3.0 if name == "diffsynth" else (0.0 if warmup else 1.0)
    assert report["first_update_lr_factor"] == expected_first


def test_import_validation_and_report_do_not_load_torch_or_transformers():
    # Fresh process: this test module itself uses torch for real CPU updates.
    # Import blocking proves the lightweight API does not merely reuse a module
    # cached by another test, and also covers Config's eventual integration.
    source = Path(__file__).resolve().parents[1] / "src"
    script = """
import importlib.abc
import sys
class NoModelLibraries(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'transformers'}:
            raise AssertionError(f'heavy import: {fullname}')
sys.meta_path.insert(0, NoModelLibraries())
from qwen21_trainer.config import Config
from qwen21_trainer.lr_schedule import normalize_lr_options, lr_scheduler_report
assert normalize_lr_options({'max_steps': 8})['lr_scheduler'] == 'diffsynth'
assert lr_scheduler_report({'max_steps': 8})['legacy_total_iters'] == 5
assert not {'torch', 'transformers'} & sys.modules.keys()
print('torch-free validation and report verified')
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        env={**os.environ, "PYTHONPATH": str(source), "CUDA_VISIBLE_DEVICES": ""},
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    assert "torch-free validation and report verified" in completed.stdout
