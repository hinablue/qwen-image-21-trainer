"""CPU-only sampling orchestration and real tiny-DiT tests.

All images here use explicit fixture TE/VAE components, NOT pretrained Qwen
text/image encoders. No production weights, network download, or CUDA is used.
"""

import copy
import gc
import hashlib
import json
import random
import weakref
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from qwen21_trainer import runtime
from qwen21_trainer.sampling import TrainingSampler


def cfg(**sample):
    return SimpleNamespace(
        sample={
            "enabled": True,
            "every": 2,
            "prompts": [],
            "prompt": "fixture cup",
            "negative_prompt": "fixture negative",
            "seed": 17,
            "width": 64,
            "height": 64,
            "steps": 2,
            "cfg_scale": 3.0,
            **sample,
        },
        training={"max_steps": 5},
        model={"device": "cpu", "attention": "segmented", "cpu_offload": False},
    )


def model_fixture():
    model = torch.nn.Module()
    model.pipe = torch.nn.Module()
    model.pipe.dit = torch.nn.Sequential(torch.nn.Linear(3, 3), torch.nn.Dropout(0.25))
    model.pipe.scheduler = SimpleNamespace(
        training=True,
        timesteps=torch.tensor([999.0, 500.0]),
        linear_timesteps_weights=torch.tensor([0.4, 0.9]),
    )
    model.train()
    model.pipe.dit[1].eval()  # Deliberately mixed modes.
    model.pipe.dit[0].bias.requires_grad_(False)
    model.pipe.dit[0].weight.grad = torch.ones_like(model.pipe.dit[0].weight)
    return model


def consume_rng():
    random.random()
    np.random.random(7)
    torch.rand(7)


def snapshot(model):
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state().clone(),
        "modes": [(module, module.training) for module in model.modules()],
        "parameters": [
            (
                p,
                p.detach().clone(),
                p.requires_grad,
                p.grad,
                None if p.grad is None else p.grad.clone(),
                p.device,
            )
            for p in model.parameters()
        ],
        "scheduler": model.pipe.scheduler,
        "scheduler_state": copy.deepcopy(vars(model.pipe.scheduler)),
    }


def assert_equal_state(actual, expected):
    if isinstance(actual, torch.Tensor):
        assert torch.equal(actual, expected)
    elif isinstance(actual, dict):
        assert actual.keys() == expected.keys()
        for key in actual:
            assert_equal_state(actual[key], expected[key])
    elif isinstance(actual, (list, tuple)):
        assert len(actual) == len(expected)
        for a, b in zip(actual, expected):
            assert_equal_state(a, b)
    else:
        assert actual == expected


def assert_preserved(model, before):
    assert random.getstate() == before["python"]
    numpy = np.random.get_state()
    assert numpy[0] == before["numpy"][0]
    assert np.array_equal(numpy[1], before["numpy"][1])
    assert numpy[2:] == before["numpy"][2:]
    assert torch.equal(torch.get_rng_state(), before["torch"])
    for module, training in before["modes"]:
        assert module.training == training
    for p, value, requires_grad, grad, grad_value, device in before["parameters"]:
        assert torch.equal(p, value)
        assert p.requires_grad == requires_grad
        assert p.grad is grad
        assert p.device == device
        if grad is not None:
            assert torch.equal(p.grad, grad_value)
    assert model.pipe.scheduler is before["scheduler"]
    assert_equal_state(vars(model.pipe.scheduler), before["scheduler_state"])


def encoder_fixture():
    root = torch.nn.Module()
    node = root
    for name in ("model", "model", "language_model"):
        child = torch.nn.Module()
        setattr(node, name, child)
        node = child
    node.norm = torch.nn.LayerNorm(3)
    return root, node.norm


class FixturePipeline(torch.nn.Module):
    """Noise-colour fixture PNG, with no claim of real generative inference."""

    def __init__(self, dit, calls, *, fail=False):
        super().__init__()
        self.dit = None
        self.text_encoder, self.norm = encoder_fixture()
        self.vae = torch.nn.Linear(1, 1)
        self.processor = object()
        self.scheduler = SimpleNamespace(training=False, timesteps=torch.tensor([1.0]))
        self.expected_dit = weakref.ref(dit)
        self.calls = calls
        self.fail = fail
        self.norm.register_forward_hook(lambda *args: None)
        self.existing_hooks = dict(self.norm._forward_hooks)

    def __call__(self, **kwargs):
        assert self.dit is self.expected_dit()
        assert not torch.is_grad_enabled()
        assert not any(module.training for module in self.dit.modules())
        assert any(p.requires_grad for p in self.dit.parameters())
        assert kwargs["use_kv_cache"] is False
        assert kwargs["rand_device"] == "cpu"
        assert kwargs["use_flex_attention"] is False
        assert self.norm._forward_hooks == self.existing_hooks
        self.calls.append(kwargs)
        consume_rng()
        self.scheduler.timesteps = torch.tensor([3.0, 2.0, 1.0])
        self.norm.register_forward_hook(lambda *args: None)
        bar = kwargs["progress_bar_cmd"](range(kwargs["num_inference_steps"]))
        for _ in bar:
            if self.fail:
                raise RuntimeError("fixture inference failure")
        generator = torch.Generator().manual_seed(kwargs["seed"])
        colour = tuple(torch.randint(0, 256, (3,), generator=generator).tolist())
        return Image.new("RGB", (kwargs["width"], kwargs["height"]), colour)


def factory_fixture(monkeypatch, model, *, fail=False, retain=True):
    calls, factories, pipes = [], [], []

    def build(config, assets, roles, **kwargs):
        assert roles == ["text_encoder", "vae"]
        assert kwargs == {}
        factories.append((config, assets, roles))
        consume_rng()
        print("FIXTURE loader: no pretrained weights read")
        pipe = FixturePipeline(model.pipe.dit, calls, fail=fail)
        pipes.append(
            pipe
            if retain
            else (weakref.ref(pipe), weakref.ref(pipe.text_encoder), weakref.ref(pipe.vae))
        )
        return pipe

    monkeypatch.setattr(runtime, "build_pipeline", build)
    return calls, factories, pipes


def test_disabled_has_no_loading_files_or_model_access(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime, "build_pipeline", lambda *a, **k: pytest.fail("must not load"))
    config = cfg(enabled=False)
    config.model["cpu_offload"] = True
    sampler = TrainingSampler(config, object(), object(), tmp_path / "unused")
    assert not sampler.should_sample(2)
    assert sampler.generate(2) == []
    assert sampler.generate(5) == []
    assert not (tmp_path / "unused").exists()


def test_legacy_config_missing_enabled_is_disabled(tmp_path):
    config = cfg()
    del config.sample["enabled"]
    assert TrainingSampler(config, {}, object(), tmp_path).generate(5) == []


def test_cadence_final_step_dedup_and_fixed_seed(tmp_path, monkeypatch, capsys):
    model = model_fixture()
    config = cfg(prompts=["first fixture", "第二個 fixture"])
    assets = {"fixture_only": True}
    calls, factories, pipes = factory_fixture(monkeypatch, model)
    sampler = TrainingSampler(config, assets, model, tmp_path)
    assert [s for s in range(8) if sampler.should_sample(s)] == [2, 4, 5]
    assert not sampler.should_sample(True)
    assert not sampler.should_sample(2.0)
    all_records = []
    for step in (2, 4, 5):
        before = snapshot(model)
        with torch.enable_grad():
            records = sampler.generate(step)
            assert torch.is_grad_enabled()
        assert_preserved(model, before)
        assert sampler.generate(step) == []
        assert not sampler.should_sample(step)
        assert len(records) == 2
        all_records.append(records)
        for index, record in enumerate(records):
            assert record["prompt"] == config.sample["prompts"][index]
            assert record["seed"] == 17
            assert record["step"] == step and record["index"] == index
            path = Path(record["path"])
            assert path == tmp_path / "samples" / f"step-{step:06d}-{index:02d}.png"
            with Image.open(path) as image:
                assert image.format == "PNG" and image.size == (64, 64)
            assert record["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
        manifest = json.loads((tmp_path / "samples" / f"step-{step:06d}.json").read_text())
        assert manifest["samples"] == records
        assert manifest["negative_prompt"] == "fixture negative"
        assert manifest["num_inference_steps"] == 2
    assert len(calls) == 6 and len(factories) == 3
    assert all(entry[1] is assets for entry in factories)
    assert [call["prompt"] for call in calls] == config.sample["prompts"] * 3
    assert {call["seed"] for call in calls} == {17}
    for index in range(2):
        assert len({group[index]["sha256"] for group in all_records}) == 1
    assert all(
        pipe.dit is None and pipe.vae is None and pipe.text_encoder is None for pipe in pipes
    )
    assert all(pipe.norm._forward_hooks == pipe.existing_hooks for pipe in pipes)
    assert "FIXTURE loader" not in capsys.readouterr().out
    assert (tmp_path / "samples/loading.log").read_text().count("FIXTURE loader") == 3


def test_last_step_multiple_and_single_prompt(tmp_path, monkeypatch):
    model = model_fixture()
    config = cfg()
    config.training["max_steps"] = 4
    calls, factories, _ = factory_fixture(monkeypatch, model)
    sampler = TrainingSampler(config, {}, model, tmp_path)
    assert sampler.generate(1) == []
    assert sampler.generate(4)[0]["prompt"] == "fixture cup"
    assert sampler.generate(4) == []
    assert len(calls) == len(factories) == 1


@pytest.mark.parametrize("where", ["construction", "inference", "png_save"])
def test_error_restores_rng_modes_grads_scheduler_and_cleans_group(tmp_path, monkeypatch, where):
    model = model_fixture()
    _, _, pipes = factory_fixture(monkeypatch, model, fail=where == "inference")
    if where == "construction":

        def broken(*a, **k):
            consume_rng()
            raise RuntimeError("fixture construction failure")

        monkeypatch.setattr(runtime, "build_pipeline", broken)
    elif where == "png_save":

        def broken_save(*a, **k):
            consume_rng()
            raise RuntimeError("fixture png_save failure")

        monkeypatch.setattr(Image.Image, "save", broken_save)
    before = snapshot(model)
    sampler = TrainingSampler(cfg(), {}, model, tmp_path)
    with torch.no_grad(), pytest.raises(RuntimeError, match="fixture"):
        sampler.generate(2)
    assert_preserved(model, before)
    assert sampler.should_sample(2)
    assert not list((tmp_path / "samples").glob("step-*"))
    for pipe in pipes:
        assert pipe.dit is pipe.vae is pipe.text_encoder is None
        assert pipe.norm._forward_hooks == pipe.existing_hooks


@pytest.mark.parametrize("fail", [False, True])
def test_cuda_rng_guard_with_cpu_simulation_only(monkeypatch, fail):
    from qwen21_trainer.sampling import _training_state

    # Exercise fork_rng's CUDA save/restore API without creating a CUDA context.
    # This proves device selection/control flow, NOT real GPU inference safety.
    state = {3: torch.tensor([1, 2, 3], dtype=torch.uint8)}
    calls = []

    def get_rng(device):
        calls.append(("get", device))
        return state[device].clone()

    def set_rng(value, device):
        calls.append(("set", device))
        state[device] = value.clone()

    monkeypatch.setattr(torch.cuda, "get_rng_state", get_rng)
    monkeypatch.setattr(torch.cuda, "set_rng_state", set_rng)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: pytest.fail("no CUDA init"))
    model = model_fixture()
    before = snapshot(model)
    try:
        with _training_state(model, "cuda:3"):
            state[3].zero_()
            consume_rng()
            if fail:
                raise RuntimeError("simulated failure")
    except RuntimeError:
        assert fail
    assert calls == [("get", 3), ("set", 3)]
    assert torch.equal(state[3], torch.tensor([1, 2, 3], dtype=torch.uint8))
    assert_preserved(model, before)


def test_partial_group_removed_and_retry_is_safe(tmp_path, monkeypatch):
    model = model_fixture()
    calls, _, _ = factory_fixture(monkeypatch, model)
    original_call = FixturePipeline.__call__

    def fail_second(self, **kwargs):
        self.fail = len(calls) == 1
        return original_call(self, **kwargs)

    monkeypatch.setattr(FixturePipeline, "__call__", fail_second)
    sampler = TrainingSampler(cfg(prompts=["one", "two"]), {}, model, tmp_path)
    before = snapshot(model)
    with pytest.raises(RuntimeError, match="inference"):
        sampler.generate(2)
    assert_preserved(model, before)
    assert not list((tmp_path / "samples").glob("step-*"))
    assert sampler.should_sample(2)
    monkeypatch.setattr(FixturePipeline, "__call__", original_call)
    assert len(sampler.generate(2)) == 2


def test_temporary_models_are_collectable(tmp_path, monkeypatch):
    model = model_fixture()
    _, _, refs = factory_fixture(monkeypatch, model, retain=False)
    TrainingSampler(cfg(), {}, model, tmp_path).generate(2)
    gc.collect()
    assert all(ref() is None for ref in refs[0])
    assert model.pipe.dit is not None


@pytest.mark.parametrize("entry", ["step-000002-00.png", "step-000002-01.png", "step-000002.json"])
@pytest.mark.parametrize("symlink", [False, True])
def test_no_overwrite_or_following_file_symlinks(tmp_path, monkeypatch, entry, symlink):
    folder = tmp_path / "samples"
    folder.mkdir()
    sentinel = tmp_path / "protected"
    sentinel.write_bytes(b"do not touch")
    path = folder / entry
    if symlink:
        path.symlink_to(sentinel)
    else:
        path.write_bytes(b"existing")
    monkeypatch.setattr(runtime, "build_pipeline", lambda *a, **k: pytest.fail("must not load"))
    sampler = TrainingSampler(cfg(prompts=["one", "two"]), {}, model_fixture(), tmp_path)
    with pytest.raises(FileExistsError):
        sampler.generate(2)
    assert sentinel.read_bytes() == b"do not touch"
    assert path.read_bytes() == (b"do not touch" if symlink else b"existing")
    assert sorted(p.name for p in folder.iterdir()) == [entry]


@pytest.mark.parametrize("location", ["samples", "root", "ancestor", "loading.log"])
def test_directory_and_log_symlinks_rejected(tmp_path, monkeypatch, location):
    outside = tmp_path / "outside"
    outside.mkdir()
    out = tmp_path / "out"
    if location == "root":
        out.symlink_to(outside, target_is_directory=True)
    elif location == "ancestor":
        out.symlink_to(outside, target_is_directory=True)
        out = out / "nested"
    else:
        out.mkdir()
        if location == "samples":
            (out / "samples").symlink_to(outside, target_is_directory=True)
        else:
            (out / "samples").mkdir()
            (out / "samples/loading.log").symlink_to(outside / "missing")
    monkeypatch.setattr(runtime, "build_pipeline", lambda *a, **k: pytest.fail("must not load"))
    with pytest.raises(OSError):
        TrainingSampler(cfg(), {}, model_fixture(), out).generate(2)
    assert list(outside.iterdir()) == []


def test_path_traversal_rejected(tmp_path):
    with pytest.raises(ValueError, match="path"):
        TrainingSampler(cfg(), {}, model_fixture(), tmp_path / "inside/../outside").generate(2)
    assert not (tmp_path / "outside").exists()


def test_existing_manifest_rejected_by_new_sampler(tmp_path, monkeypatch):
    model = model_fixture()
    factory_fixture(monkeypatch, model)
    TrainingSampler(cfg(), {}, model, tmp_path).generate(2)
    before = {p.name: p.read_bytes() for p in (tmp_path / "samples").iterdir()}
    with pytest.raises(FileExistsError):
        TrainingSampler(cfg(), {}, model, tmp_path).generate(2)
    assert {p.name: p.read_bytes() for p in (tmp_path / "samples").iterdir()} == before


@pytest.mark.parametrize("kind", ["pipeline", "scheduler", "second_dit", "managed"])
def test_unsafe_factory_result_rejected_without_mutating_training(tmp_path, monkeypatch, kind):
    model = model_fixture()
    original_dit = model.pipe.dit
    pipe = FixturePipeline(original_dit, [])
    if kind == "pipeline":
        pipe = model.pipe
    elif kind == "scheduler":
        pipe.scheduler = model.pipe.scheduler
    elif kind == "second_dit":
        pipe.dit = torch.nn.Linear(1, 1)
    else:
        pipe.vram_management_enabled = True
    monkeypatch.setattr(runtime, "build_pipeline", lambda *a, **k: pipe)
    before = snapshot(model)
    with pytest.raises((RuntimeError, ValueError)):
        TrainingSampler(cfg(), {}, model, tmp_path).generate(2)
    assert_preserved(model, before)
    assert model.pipe.dit is original_dit


def test_offload_rejected_before_factory(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime, "build_pipeline", lambda *a, **k: pytest.fail("must not load"))
    config = cfg()
    config.model["cpu_offload"] = True
    with pytest.raises(ValueError, match="cpu_offload"):
        TrainingSampler(config, {}, model_fixture(), tmp_path)
    model = model_fixture()
    model._offload_manager = object()
    with pytest.raises(ValueError, match="offload"):
        TrainingSampler(cfg(), {}, model, tmp_path).generate(2)


class TinyFixtureEncoder(torch.nn.Module):
    """Deterministic synthetic tokens/embeddings, not a pretrained text encoder."""

    def __init__(self):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.model = torch.nn.Module()
        self.model.model.language_model = torch.nn.Module()
        self.model.model.language_model.norm = torch.nn.LayerNorm(64)

    def forward(self, input_ids, attention_mask):
        norm = self.model.model.language_model.norm
        norm.register_forward_hook(lambda *args: None)  # Pinned upstream leak fixture.
        embedding = torch.sin(input_ids[..., None].float() + torch.arange(64).float())
        return norm(embedding.to(norm.weight.dtype))


class TinyFixtureProcessor:
    tokenizer = SimpleNamespace(encode=lambda _: [999])

    def apply_chat_template(self, *a, **k):
        return [1]

    def __call__(self, text, **kwargs):
        tokens = torch.tensor([[1, *[ord(c) % 97 + 1 for c in text[0][-5:]]]])
        return SimpleNamespace(
            input_ids=tokens,
            attention_mask=torch.ones_like(tokens),
            to=lambda device: SimpleNamespace(
                input_ids=tokens.to(device), attention_mask=torch.ones_like(tokens).to(device)
            ),
        )


class TinyFixtureVAE(torch.nn.Module):
    """Fixture decoder only: latent channels resized into an RGB test image."""

    def decode(self, latents, **kwargs):
        return torch.nn.functional.interpolate(latents[:, :3].float().tanh(), scale_factor=16)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_real_tiny_dit_lora_sampling_keeps_optimizer_checkpoint_and_next_update(
    tmp_path,
    monkeypatch,
    dtype,
):
    from diffsynth.models.qwen_image_21_dit import QwenImage21DiT
    from diffsynth.pipelines.qwen_image_21 import QwenImage21Pipeline

    assert not torch.cuda.is_available(), "Run these tests with CUDA_VISIBLE_DEVICES=''"
    torch.set_num_threads(1)
    torch.manual_seed(71)
    pipe = QwenImage21Pipeline(device="cpu", torch_dtype=dtype)
    pipe.dit = QwenImage21DiT(
        in_channels=64,
        out_channels=64,
        num_layers=1,
        num_attention_heads=2,
        attention_head_dim=32,
        context_in_dim=64,
        axes_dims_rope=(8, 12, 12),
    ).to(dtype=dtype)
    settings = dict(rank=2, gradient_checkpointing=True, checkpointing_offload=False)
    model = runtime.make_training_module(pipe, settings, targets="to_q,to_k,to_v,to_out.0")
    tensors = {
        "input_latents": torch.randn(1, 64, 4, 4, dtype=dtype),
        "prompt_embeds": torch.randn(1, 5, 64, dtype=dtype),
        "edit_image_pad_mask": torch.zeros(1, 5, dtype=torch.bool),
    }
    optimizer = torch.optim.AdamW(model.trainable_modules(), lr=1e-3)
    lr_scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)
    model.train()
    model(tensors).backward()
    optimizer.step()
    lr_scheduler.step()
    # Keep accumulated grad buffers to verify sampling does not clear or replace them.
    reference = copy.deepcopy(model)
    reference_optimizer = torch.optim.AdamW(reference.trainable_modules(), lr=1e-3)
    reference_optimizer.load_state_dict(copy.deepcopy(optimizer.state_dict()))
    optimizer_state = copy.deepcopy(optimizer.state_dict())
    lr_state = copy.deepcopy(lr_scheduler.state_dict())
    before_checkpoint = tmp_path / "before.safetensors"
    runtime.save_lora(model, before_checkpoint, {"test_only": "fixture"})
    refs, forwards = [], []

    def build(config, assets, roles):
        assert roles == ["text_encoder", "vae"]
        # Real production __call__, scheduler and model_fn. Only TE/VAE/tokenizer
        # are explicit CPU fixtures; the trained tiny DiT is NOT mocked.
        consume_rng()
        sample_pipe = QwenImage21Pipeline(device="cpu", torch_dtype=dtype)
        sample_pipe.text_encoder = TinyFixtureEncoder().to(dtype=dtype)
        sample_pipe.processor = TinyFixtureProcessor()
        sample_pipe.vae = TinyFixtureVAE()
        refs.append(weakref.ref(sample_pipe))
        return sample_pipe

    monkeypatch.setattr(runtime, "build_pipeline", build)
    hook = pipe.dit.register_forward_hook(lambda *args: forwards.append(torch.is_grad_enabled()))
    before = snapshot(model)
    freqs = pipe.dit.pos_embed.freqs
    sampler = TrainingSampler(
        cfg(prompts=["tiny fixture A", "tiny fixture B"]), {}, model, tmp_path
    )
    records = sampler.generate(2)
    hook.remove()
    assert len(records) == 2 and len(forwards) == 8  # 2 prompts x 2 steps x pos/neg CFG.
    assert not any(forwards)
    assert_preserved(model, before)
    assert pipe.dit.pos_embed.freqs is freqs
    assert_equal_state(optimizer.state_dict(), optimizer_state)
    assert_equal_state(lr_scheduler.state_dict(), lr_state)
    after_checkpoint = tmp_path / "after.safetensors"
    runtime.save_lora(model, after_checkpoint, {"test_only": "fixture"})
    # Compare saved tensor values (safetensors metadata ordering is not semantic).
    from safetensors.torch import load_file

    assert_equal_state(load_file(str(before_checkpoint)), load_file(str(after_checkpoint)))
    gc.collect()
    assert all(ref() is None for ref in refs)
    # Sampling must not change the next stochastic training loss or optimizer update.
    state = torch.get_rng_state()
    optimizer.zero_grad(set_to_none=True)
    loss = model(tensors)
    loss.backward()
    optimizer.step()
    torch.set_rng_state(state)
    reference_optimizer.zero_grad(set_to_none=True)
    reference_loss = reference(tensors)
    reference_loss.backward()
    reference_optimizer.step()
    assert torch.equal(loss, reference_loss)
    assert_equal_state(model.state_dict(), reference.state_dict())
    assert_equal_state(optimizer.state_dict(), reference_optimizer.state_dict())
    for record in records:
        with Image.open(record["path"]) as image:
            assert image.size == (64, 64) and image.format == "PNG"
