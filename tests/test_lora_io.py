"""CPU-only real-PEFT / tiny real-Qwen round trips; no pretrained files or downloads."""

from __future__ import annotations

import copy
import hashlib
import random
from pathlib import Path

import numpy as np
import pytest
import torch
from peft import LoraConfig, inject_adapter_in_model
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from qwen21_trainer.lora_io import (
    FORMAT,
    FORMAT_VERSION,
    load_training_checkpoint,
    read_checkpoint,
    save_checkpoint,
)


@pytest.fixture(autouse=True)
def cpu_only():
    assert not torch.cuda.is_available(), "Run with CUDA_VISIBLE_DEVICES=''"
    assert torch.get_num_threads() == 1, "Run with OMP/MKL/OPENBLAS_NUM_THREADS=1"


def make_model(kind="linear", dtype=torch.float32, rank=32, alpha=16):
    from diffsynth.models.qwen_image_21_dit import QwenImage21DiT
    from diffsynth.pipelines.qwen_image_21 import QwenImage21Pipeline

    from qwen21_trainer.runtime import make_training_module

    torch.manual_seed(71)
    pipe = QwenImage21Pipeline(device="cpu", torch_dtype=dtype)
    if kind == "linear":
        pipe.dit = torch.nn.Sequential(
            torch.nn.Linear(64, 64), torch.nn.Tanh(), torch.nn.Linear(64, 64)
        ).to(dtype=dtype)
        targets = "0,2"
    else:
        pipe.dit = QwenImage21DiT(
            in_channels=64,
            out_channels=64,
            num_layers=1,
            num_attention_heads=2,
            attention_head_dim=32,
            context_in_dim=64,
            axes_dims_rope=(8, 12, 12),
        ).to(dtype=dtype)
        targets = "to_q,to_k,to_v,to_out.0"
    base = copy.deepcopy(pipe.dit)
    model = make_training_module(
        pipe,
        dict(rank=rank, alpha=alpha, gradient_checkpointing=True, checkpointing_offload=False),
        targets=targets,
        attention="segmented",
    )
    if kind == "linear":
        inputs = torch.randn(3, 64, dtype=dtype)

        def forward(dit):
            return dit(inputs)

        def loss():
            return forward(model.pipe.dit).float().square().mean()
    else:
        tensors = {
            "input_latents": torch.randn(1, 64, 4, 4, dtype=dtype),
            "prompt_embeds": torch.randn(1, 5, 64, dtype=dtype),
            "edit_image_pad_mask": torch.zeros(1, 5, dtype=torch.bool),
        }
        deterministic = dict(
            hidden_states=torch.randn(1, 16, 64, dtype=dtype),
            encoder_hidden_states=tensors["prompt_embeds"],
            timestep=torch.tensor([0.4], dtype=dtype),
            img_shapes=[[(1, 4, 4)]],
            img_mask=torch.tensor([[False] * 5 + [True] * 4]),
            use_flex_attention=False,
        )

        def forward(dit):
            return dit(**deterministic)

        def loss():
            return model(tensors)

    return model, base, forward, loss


def train_twice(model, loss):
    before = {k: v.detach().clone() for k, v in model.named_parameters()}
    optimizer = torch.optim.AdamW(model.trainable_modules(), lr=0.01)
    model.train()
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        value = loss()
        assert torch.isfinite(value)
        value.backward()
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        optimizer.step()
    changed = []
    for key, value in model.named_parameters():
        if "lora_" in key:
            changed.append((key, not torch.equal(value, before[key])))
        else:
            assert torch.equal(value, before[key]), key
    assert any(changed for key, changed in changed if "lora_A" in key)
    assert any(changed for key, changed in changed if "lora_B" in key)
    model.eval()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def snapshot(model):
    return dict(
        state={k: v.detach().clone() for k, v in model.state_dict().items()},
        parameters={
            k: (id(p), p.requires_grad, None if p.grad is None else p.grad.clone())
            for k, p in model.named_parameters()
        },
        module_state={
            k: (
                m.training,
                copy.deepcopy(getattr(m, "lora_alpha", None)),
                copy.deepcopy(getattr(m, "scaling", None)),
            )
            for k, m in model.named_modules()
        },
        torch_rng=torch.get_rng_state().clone(),
        python_rng=random.getstate(),
        numpy_rng=copy.deepcopy(np.random.get_state()),
    )


def assert_preserved(model, before, *, allow_lora_change=False):
    for key, value in model.state_dict().items():
        if not allow_lora_change or "lora_" not in key:
            assert torch.equal(value, before["state"][key]), key
    for key, p in model.named_parameters():
        identity, requires_grad, grad = before["parameters"][key]
        assert id(p) == identity and p.requires_grad == requires_grad
        assert (p.grad is None) == (grad is None)
        if grad is not None:
            assert torch.equal(p.grad, grad)
    for key, m in model.named_modules():
        assert (m.training, getattr(m, "lora_alpha", None), getattr(m, "scaling", None)) == before[
            "module_state"
        ][key]
    assert torch.equal(torch.get_rng_state(), before["torch_rng"])
    assert random.getstate() == before["python_rng"]
    current = np.random.get_state()
    assert current[0] == before["numpy_rng"][0]
    assert np.array_equal(current[1], before["numpy_rng"][1])
    assert current[2:] == before["numpy_rng"][2:]


def raw_state(rank=32, alpha=16, dtype=torch.float32):
    state = {
        f"{target}.lora_{part}.default.weight": torch.full(shape, 0.125, dtype=dtype)
        for target in ("0", "2")
        for part, shape in (("A", (rank, 64)), ("B", (64, rank)))
    }
    if alpha is not None:
        state.update(
            {f"{target}.alpha": torch.tensor(alpha, dtype=torch.float64) for target in ("0", "2")}
        )
    return state


def write_checkpoint(tmp_path, state=None, metadata=None, name="input.safetensors"):
    path = tmp_path / name
    save_file(raw_state() if state is None else state, str(path), metadata=metadata or {})
    return path


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("kind", ["linear", "qwen"])
def test_actual_training_save_raw_warm_start_and_upstream_fusion(tmp_path, dtype, kind):
    from diffsynth.diffusion.base_pipeline import BasePipeline
    from diffsynth.utils.lora.general import GeneralLoRALoader

    model, base, forward, loss = make_model(kind, dtype)
    target = copy.deepcopy(model)
    train_twice(model, loss)
    expected = forward(model.pipe.dit).detach()
    path = tmp_path / f"{kind}-{dtype}.safetensors"
    before_save = snapshot(model)
    save_checkpoint(model, path, {"rank": 32, "alpha": 16, "fixture": "CPU-only"})
    assert_preserved(model, before_save)
    source_digest = digest(path)
    raw = load_file(str(path))
    checkpoint = read_checkpoint(path)
    assert checkpoint.rank == 32 and checkpoint.alpha == 16 and not checkpoint.legacy
    assert checkpoint.metadata["rank"] == "32"
    assert checkpoint.metadata["alpha"] == "16.0"
    assert checkpoint.metadata["scale"] == "0.5"
    assert checkpoint.metadata["lora_format"] == FORMAT
    assert checkpoint.metadata["lora_format_version"] == FORMAT_VERSION
    assert checkpoint.metadata["lora_storage"] == "raw"
    exported = model.export_trainable_state_dict(model.state_dict(), remove_prefix="pipe.dit.")
    assert set(raw) == set(checkpoint.inference_state)
    for key, value in exported.items():
        assert value.dtype == dtype
        assert torch.equal(raw[key], value), "A/B must NOT be pre-multiplied by alpha/rank"
        assert torch.equal(checkpoint.training_state[key], value)
    alpha_keys = [key for key in raw if key.endswith(".alpha")]
    assert len(alpha_keys) == len(exported) // 2
    assert all(
        raw[key].dtype == torch.float64 and raw[key].ndim == 0 and raw[key].item() == 16
        for key in alpha_keys
    )
    before_load = snapshot(target)
    report = load_training_checkpoint(target, path)
    assert_preserved(target, before_load, allow_lora_change=True)
    assert report["rank"] == 32 and report["alpha"] == 16 and report["scale"] == 0.5
    assert report["tensor_count"] == len(exported)
    assert report["resume_kind"] == "weights-only" and not report["legacy"]
    target.eval()
    assert torch.equal(forward(target.pipe.dit), expected), (
        "same-config raw warm-start must be bitwise"
    )
    assert digest(path) == source_digest
    loader = GeneralLoRALoader(device="cpu", torch_dtype=dtype)
    with pytest.warns(UserWarning, match="Alpha detected"):
        converted = loader.convert_state_dict(checkpoint.inference_state)
    for key, value in exported.items():
        expected_weight = value * 0.5 if ".lora_A." in key else value
        assert torch.equal(converted[key.replace(".default", "")], expected_weight)
    # Real BasePipeline.load_lora converts once, then its fuse calls convert again:
    # the intermediate state has no alpha, so the second conversion cannot rescale.
    fused = copy.deepcopy(base)
    pipe = BasePipeline(device="cpu", torch_dtype=dtype)
    with pytest.warns(UserWarning, match="Alpha detected"):
        pipe.load_lora(fused, state_dict=checkpoint.inference_state, hotload=False, verbose=0)
    distinguish_twice = []
    for name, module in fused.named_modules():
        a_key = f"{name}.lora_A.default.weight"
        if a_key not in exported:
            continue
        a, b = exported[a_key], exported[a_key.replace(".lora_A.", ".lora_B.")]
        original = base.get_submodule(name).weight
        once = original + b @ (a * 0.5)
        twice = original + b @ (a * 0.25)
        assert torch.equal(module.weight, once)
        distinguish_twice.append(not torch.equal(module.weight, twice))
    assert any(distinguish_twice), "fixture must detect accidental double scaling"
    tolerance = 2e-2 if dtype == torch.bfloat16 else 5e-6
    torch.testing.assert_close(forward(fused), expected, rtol=tolerance, atol=tolerance)
    # Fused base-weight rounding is not expected to be bitwise PEFT-equivalent.
    assert digest(path) == source_digest


@pytest.mark.parametrize("alpha", [1.0, 16.123456789012345, 31.99999999999999, 32.0])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_float_alpha_exact_fp64_and_explicit_one_not_sentinel(tmp_path, alpha, dtype):
    model, _, forward, loss = make_model(dtype=dtype, alpha=alpha)
    target = copy.deepcopy(model)
    train_twice(model, loss)
    path = tmp_path / "float.safetensors"
    save_checkpoint(model, path, {"ss_network_alpha": repr(alpha)})
    checkpoint = read_checkpoint(path)
    assert checkpoint.alpha == alpha
    assert float(checkpoint.metadata["alpha"]) == alpha
    assert all(
        value.item() == alpha and value.dtype == torch.float64
        for key, value in checkpoint.inference_state.items()
        if key.endswith(".alpha")
    )
    load_training_checkpoint(target, path)
    target.eval()
    assert torch.equal(forward(target.pipe.dit), forward(model.pipe.dit))


def test_cpu_state_preserves_float_alpha_before_bf16_pipeline_conversion(tmp_path):
    from diffsynth.diffusion.base_pipeline import BasePipeline
    from diffsynth.utils.lora.general import GeneralLoRALoader

    alpha = 16.123456789012345
    state = raw_state(alpha=alpha)
    path = write_checkpoint(tmp_path, state)
    checkpoint = read_checkpoint(path)
    assert checkpoint.alpha == alpha
    loader = GeneralLoRALoader(device="cpu", torch_dtype=torch.bfloat16)
    with pytest.warns(UserWarning, match="Alpha detected"):
        converted = loader.convert_state_dict(checkpoint.inference_state)
    expected = state["0.lora_A.default.weight"] * (torch.tensor(alpha, dtype=torch.float64) / 32)
    assert torch.equal(converted["0.lora_A.weight"], expected)
    rounded_alpha = torch.tensor(alpha, dtype=torch.bfloat16)
    incorrect = state["0.lora_A.default.weight"] * (rounded_alpha / 32)
    assert not torch.equal(expected, incorrect)
    # Exercise the full BasePipeline route, not just its converter.
    base = torch.nn.Sequential(
        torch.nn.Linear(64, 64), torch.nn.Tanh(), torch.nn.Linear(64, 64)
    ).to(torch.bfloat16)
    before = base[0].weight.detach().clone()
    with pytest.warns(UserWarning, match="Alpha detected"):
        BasePipeline(device="cpu", torch_dtype=torch.bfloat16).load_lora(
            base, state_dict=checkpoint.inference_state, hotload=False
        )
    assert torch.equal(
        base[0].weight, before + state["0.lora_B.default.weight"].bfloat16() @ expected.bfloat16()
    )


@pytest.mark.parametrize("canonical", [True, False])
def test_legacy_ab_only_is_rank_alpha_and_source_unchanged(tmp_path, canonical):
    state = raw_state(alpha=None)
    if not canonical:
        state = {key.replace(".default", ""): value for key, value in state.items()}
    path = write_checkpoint(
        tmp_path, state, {"rank": "32", "alpha": "32", "ss_network_alpha": "32"}
    )
    original = digest(path)
    checkpoint = read_checkpoint(path)
    assert checkpoint.legacy and checkpoint.alpha == checkpoint.rank == 32
    assert all(".default.weight" in key for key in checkpoint.training_state)
    model, _, _, _ = make_model(alpha=32)
    before = snapshot(model)
    report = load_training_checkpoint(model, path)
    assert report["legacy"] and report["scale"] == 1.0
    assert_preserved(model, before, allow_lora_change=True)
    assert digest(path) == original


@pytest.mark.parametrize("canonical", [True, False])
def test_generic_raw_ab_alpha_with_consistent_metadata_accepted(tmp_path, canonical):
    state = raw_state(alpha=16.3)
    state["0.alpha"] = torch.tensor(16.3, dtype=torch.float32)
    state["2.alpha"] = torch.tensor(16.3, dtype=torch.float32)
    if not canonical:
        state = {key.replace(".default", ""): value for key, value in state.items()}
    checkpoint = read_checkpoint(
        write_checkpoint(
            tmp_path, state, {"alpha": "16.3", "ss_network_alpha": "16.3", "ss_network_dim": "32"}
        )
    )
    assert not checkpoint.legacy and checkpoint.rank == 32
    assert checkpoint.alpha == state["0.alpha"].item()


@pytest.mark.parametrize(
    "metadata",
    [
        {"rank": "16"},
        {"alpha": "32"},
        {"ss_network_alpha": "32"},
        {"alpha": "16", "ss_network_alpha": "32"},
        {"ss_network_dim": "16"},
        {"scale": "1"},
        {"lora_alpha": "32"},
        {"lora_scale": "1"},
        {"rank": "32.5"},
        {"alpha": "nan"},
        {"alpha": "inf"},
        {"alpha": "-1"},
        {"lora_format": "unknown"},
        {"lora_storage": "baked"},
        {"storage": "baked"},
        {"format": "baked"},
        {"alpha_baked": "true"},
        {"scale_baked": "unknown"},
        {"format_version": "99"},
    ],
)
def test_metadata_disagreements_and_unknown_storage_rejected(tmp_path, metadata):
    path = write_checkpoint(tmp_path, metadata=metadata)
    before = digest(path)
    with pytest.raises(ValueError):
        read_checkpoint(path)
    assert digest(path) == before


@pytest.mark.parametrize(
    "metadata",
    [
        {"alpha": "16"},
        {"ss_network_alpha": "16"},
        {"alpha": "32", "ss_network_alpha": "16"},
    ],
)
def test_unmarked_ab_only_nonrank_alpha_is_ambiguous_rejected(tmp_path, metadata):
    path = write_checkpoint(tmp_path, raw_state(alpha=None), metadata)
    with pytest.raises(ValueError, match="alpha"):
        read_checkpoint(path)


@pytest.mark.parametrize(
    "damage",
    [
        "empty",
        "unpaired",
        "extra_base",
        "extra_key",
        "other_adapter",
        "duplicate_alias",
        "shape",
        "vector",
        "zero_shape",
        "weight_nan",
        "weight_inf",
        "integer_weight",
        "mixed_dtype",
        "partial_alpha",
        "extra_alpha",
        "alpha_vector",
        "alpha_zero",
        "alpha_negative",
        "alpha_nan",
        "alpha_inf",
        "nonuniform_rank",
        "nonuniform_alpha",
    ],
)
def test_strict_reader_rejects_malformed_tensors(tmp_path, damage):
    state = raw_state()
    key = "2.lora_B.default.weight"
    if damage == "empty":
        state = {}
    elif damage == "unpaired":
        del state[key]
    elif damage == "extra_base":
        state["2.base_layer.weight"] = torch.zeros(64, 64)
    elif damage == "extra_key":
        state["unexplained"] = torch.ones(1)
    elif damage == "other_adapter":
        state[key.replace("default", "other")] = state.pop(key)
    elif damage == "duplicate_alias":
        state[key.replace(".default", "")] = state[key].clone()
    elif damage == "shape":
        state[key] = torch.ones(64, 16)
    elif damage == "vector":
        state[key] = torch.ones(32)
    elif damage == "zero_shape":
        state[key] = torch.ones(0, 32)
    elif damage in {"weight_nan", "weight_inf"}:
        state[key][0, 0] = float("nan" if damage == "weight_nan" else "inf")
    elif damage == "integer_weight":
        state[key] = state[key].long()
    elif damage == "mixed_dtype":
        state[key] = state[key].bfloat16()
    elif damage == "partial_alpha":
        del state["2.alpha"]
    elif damage == "extra_alpha":
        state["other.alpha"] = torch.tensor(16.0)
    elif damage == "alpha_vector":
        state["2.alpha"] = torch.tensor([16.0])
    elif damage.startswith("alpha_"):
        value = {
            "alpha_zero": 0,
            "alpha_negative": -1,
            "alpha_nan": float("nan"),
            "alpha_inf": float("inf"),
        }[damage]
        state["2.alpha"] = torch.tensor(value, dtype=torch.float64)
    elif damage == "nonuniform_rank":
        state["2.lora_A.default.weight"] = torch.ones(16, 64)
        state[key] = torch.ones(64, 16)
    elif damage == "nonuniform_alpha":
        state["2.alpha"] = torch.tensor(32.0)
    path = write_checkpoint(tmp_path, state)
    with pytest.raises(ValueError):
        read_checkpoint(path)


@pytest.mark.parametrize(
    "damage",
    [
        "strip_all_alpha",
        "strip_one_alpha",
        "cast_alpha",
        "omit_scale",
        "wrong_version",
        "omit_format",
    ],
)
def test_new_format_cannot_be_disguised_as_legacy(tmp_path, damage):
    model, _, _, _ = make_model()
    path = tmp_path / "new.safetensors"
    save_checkpoint(model, path, {})
    with safe_open(str(path), framework="pt") as handle:
        metadata = handle.metadata()
    state = load_file(str(path))
    if damage == "strip_all_alpha":
        state = {key: value for key, value in state.items() if not key.endswith(".alpha")}
    elif damage == "strip_one_alpha":
        del state["2.alpha"]
    elif damage == "cast_alpha":
        state = {
            key: value.bfloat16() if key.endswith(".alpha") else value
            for key, value in state.items()
        }
    elif damage == "omit_scale":
        del metadata["scale"]
    elif damage == "wrong_version":
        metadata["lora_format_version"] = "99"
    elif damage == "omit_format":
        del metadata["lora_format"]
    damaged = write_checkpoint(tmp_path, state, metadata)
    with pytest.raises(ValueError):
        read_checkpoint(damaged)


@pytest.mark.parametrize(
    "damage", ["rank", "alpha", "target", "last_shape", "extra", "late_nonfinite", "dtype_overflow"]
)
def test_warm_start_validates_whole_file_before_mutating_model(tmp_path, damage):
    model, _, _, _ = make_model(
        dtype=torch.float16 if damage == "dtype_overflow" else torch.float32
    )
    state = raw_state(rank=16 if damage == "rank" else 32, alpha=32 if damage == "alpha" else 16)
    if damage == "target":
        state = {key.replace("2.", "other."): value for key, value in state.items()}
    elif damage == "last_shape":
        state["2.lora_B.default.weight"] = torch.ones(63, 32)
    elif damage == "extra":
        state["base.weight"] = torch.ones(1)
    elif damage == "late_nonfinite":
        state["2.lora_B.default.weight"][0, 0] = float("nan")
    elif damage == "dtype_overflow":
        state["2.lora_B.default.weight"].fill_(1e10)
    path = write_checkpoint(tmp_path, state)
    source = digest(path)
    before = snapshot(model)
    with pytest.raises(ValueError):
        load_training_checkpoint(model, path)
    assert_preserved(model, before)
    assert digest(path) == source


@pytest.mark.parametrize(
    "metadata",
    [
        {"rank": 16},
        {"alpha": 32},
        {"alpha": True},
        {"ss_network_alpha": 32},
        {"scale": 1},
        {"lora_storage": "baked"},
        {"rank": True},
        {"lora_format": "unknown"},
    ],
)
def test_save_disagreement_does_not_replace_destination(tmp_path, metadata):
    model, _, _, _ = make_model()
    path = tmp_path / "valid.safetensors"
    save_checkpoint(model, path, {})
    source = digest(path)
    before = snapshot(model)
    with pytest.raises(ValueError):
        save_checkpoint(model, path, metadata)
    assert_preserved(model, before)
    assert digest(path) == source


@pytest.mark.parametrize(
    "damage",
    [
        "rslora",
        "dora",
        "scaling",
        "nonuniform_alpha",
        "nonuniform_rank",
        "nonfinite",
        "base_trainable",
        "frozen_adapter",
        "multiple_adapters",
        "merged",
        "disabled",
        "biased_adapter",
    ],
)
def test_save_rejects_unsupported_model_contract(tmp_path, damage):
    model, _, _, _ = make_model()
    layer = model.pipe.dit[2]
    if damage == "rslora":
        layer.use_rslora["default"] = True
    elif damage == "dora":
        layer.use_dora["default"] = True
    elif damage == "scaling":
        layer.scaling["default"] = 1.0
    elif damage == "nonuniform_alpha":
        layer.lora_alpha["default"] = 32
        layer.scaling["default"] = 1.0
    elif damage == "nonuniform_rank":
        layer.r["default"] = 16
        layer.scaling["default"] = 1.0
    elif damage == "nonfinite":
        with torch.no_grad():
            layer.lora_B["default"].weight[0, 0] = float("nan")
    elif damage == "base_trainable":
        layer.base_layer.weight.requires_grad_(True)
    elif damage == "frozen_adapter":
        layer.lora_B["default"].weight.requires_grad_(False)
    elif damage == "multiple_adapters":
        inject_adapter_in_model(
            LoraConfig(r=32, lora_alpha=16, target_modules=["0", "2"]),
            model.pipe.dit,
            adapter_name="other",
        )
    elif damage == "merged":
        layer.merge()
    elif damage == "disabled":
        layer.enable_adapters(False)
    elif damage == "biased_adapter":
        layer.lora_A["default"].bias = torch.nn.Parameter(torch.zeros(32))
    path = tmp_path / "rejected.safetensors"
    with pytest.raises(ValueError):
        save_checkpoint(model, path, {})
    assert not path.exists()


def test_training_and_inference_dicts_have_independent_weight_storage(tmp_path):
    checkpoint = read_checkpoint(write_checkpoint(tmp_path))
    key = "0.lora_A.default.weight"
    raw = checkpoint.training_state[key].clone()
    checkpoint.inference_state[key].mul_(0.5)
    assert torch.equal(checkpoint.training_state[key], raw)


@pytest.mark.parametrize("alpha", [1, 16, 32])
def test_generic_integer_scalar_alpha_is_not_a_sentinel(tmp_path, alpha):
    state = raw_state(alpha=alpha)
    for key in ("0.alpha", "2.alpha"):
        state[key] = torch.tensor(alpha, dtype=torch.int64)
    checkpoint = read_checkpoint(write_checkpoint(tmp_path, state))
    assert checkpoint.alpha == alpha
    assert not checkpoint.legacy


@pytest.mark.parametrize("alpha", [True, 1e-300, 1e300])
def test_alpha_bool_or_nonrepresentable_fp32_scale_rejected(tmp_path, alpha):
    state = raw_state(alpha=None)
    for target in ("0", "2"):
        state[f"{target}.alpha"] = torch.tensor(
            alpha, dtype=torch.bool if isinstance(alpha, bool) else torch.float64
        )
    with pytest.raises(ValueError):
        read_checkpoint(write_checkpoint(tmp_path, state))


def test_known_float_alpha_mismatch_is_not_hidden_by_fp32_tolerance(tmp_path):
    model, _, _, _ = make_model(alpha=16.0000001)
    path = tmp_path / "precise.safetensors"
    save_checkpoint(model, path, {})
    target, _, _, _ = make_model(alpha=16.0)
    before = snapshot(target)
    with pytest.raises(ValueError, match="alpha"):
        load_training_checkpoint(target, path)
    assert_preserved(target, before)


@pytest.mark.parametrize("variant", ["rslora", "dora", "conv2d"])
def test_real_peft_unsupported_adapter_constructors_rejected(tmp_path, variant):
    from diffsynth.diffusion.training_module import DiffusionTrainingModule

    base = torch.nn.Sequential(
        torch.nn.Conv2d(4, 4, 1) if variant == "conv2d" else torch.nn.Linear(64, 64)
    )
    # r=1 makes rsLoRA's numerical scaling equal alpha/r; the flag must still reject it.
    config = LoraConfig(
        r=1 if variant == "rslora" else 32,
        lora_alpha=16,
        target_modules=["0"],
        use_rslora=variant == "rslora",
        use_dora=variant == "dora",
    )
    module = DiffusionTrainingModule()
    module.pipe = torch.nn.Module()
    module.pipe.dit = inject_adapter_in_model(config, base)
    with pytest.raises(ValueError):
        save_checkpoint(module, tmp_path / "unsupported.safetensors", {})


def test_exporter_cannot_bake_scaling_before_save(tmp_path, monkeypatch):
    model, _, _, _ = make_model()
    original = model.export_trainable_state_dict

    def baked(*args, **kwargs):
        return {key: value * 0.5 for key, value in original(*args, **kwargs).items()}

    monkeypatch.setattr(model, "export_trainable_state_dict", baked)
    with pytest.raises(ValueError, match="未縮放"):
        save_checkpoint(model, tmp_path / "baked.safetensors", {})


def test_corrupt_safetensors_is_clear_error_and_does_not_touch_model(tmp_path):
    model, _, _, _ = make_model()
    path = tmp_path / "corrupt.safetensors"
    path.write_bytes(b"not a safetensors file")
    original = digest(path)
    before = snapshot(model)
    with pytest.raises(ValueError, match="safetensors"):
        load_training_checkpoint(model, path)
    assert_preserved(model, before)
    assert digest(path) == original


def test_successful_reload_preserves_existing_grad_flags_modes_and_rng(tmp_path):
    model, _, forward, loss = make_model()
    target = copy.deepcopy(model)
    train_twice(model, loss)
    path = tmp_path / "trained.safetensors"
    save_checkpoint(model, path, {})
    forward(target.pipe.dit).float().square().mean().backward()
    assert any(p.grad is not None for p in target.parameters())
    target.eval()
    target.pipe.dit[0].train()
    target.pipe.dit[0].lora_A["default"].weight.requires_grad_(False)
    before = snapshot(target)
    load_training_checkpoint(target, path)
    assert_preserved(target, before, allow_lora_change=True)
    assert torch.equal(forward(target.pipe.dit), forward(model.pipe.dit))
