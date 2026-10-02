"""Config -> CLI -> real PEFT targets -> gradients/updates/checkpoint, CPU only."""

from __future__ import annotations

import copy
import json

import pytest
import torch
from peft.tuners.lora.layer import LoraLayer
from safetensors.torch import save_file

from qwen21_trainer import cli, data, runtime
from qwen21_trainer.block_filter import filter_lora_targets, parse_cli_block_patterns
from qwen21_trainer.config import Config, load_config
from qwen21_trainer.lora_io import load_training_checkpoint, read_checkpoint, save_checkpoint

TARGETS = "to_q,to_k,to_v,to_out.0,gate_layer,proj,out"
CANDIDATES = [
    f"transformer_blocks.{block}.{suffix}"
    for block in range(2)
    for suffix in (
        "attn.to_q",
        "attn.to_k",
        "attn.to_v",
        "attn.to_out.0",
        "img_mlp.proj",
        "img_mlp.out",
        "img_mlp.gate_layer",
    )
]


@pytest.mark.parametrize(
    "options,expected",
    [
        ({}, CANDIDATES),
        ({"include_blocks": [], "exclude_blocks": []}, CANDIDATES),
        ({"include_blocks": ["*mlp*"]}, [n for n in CANDIDATES if "mlp" in n]),
        ({"exclude_blocks": ["*mlp*"]}, [n for n in CANDIDATES if "mlp" not in n]),
        (
            {"include_blocks": ["*.attn.to_q", "*.attn.to_v"]},
            [n for n in CANDIDATES if n.endswith(("to_q", "to_v"))],
        ),
        (
            {"include_blocks": ["*"], "exclude_blocks": ["*mlp*", "*.attn.to_q"]},
            [n for n in CANDIDATES if "mlp" not in n and not n.endswith("to_q")],
        ),
        ({"include_blocks": ["transformer_blocks.0.*"]}, CANDIDATES[:7]),
        (
            {"include_blocks": ["transformer_blocks.[01].attn.to_?"]},
            [n for n in CANDIDATES if n.endswith(("to_q", "to_k", "to_v"))],
        ),
        (
            {"include_blocks": ["*mlp*", "*mlp*", "*.img_mlp.proj"]},
            [n for n in CANDIDATES if "mlp" in n],
        ),
    ],
)
def test_glob_semantics(options, expected):
    report = filter_lora_targets(CANDIDATES, options)
    assert report["selected_targets"] == sorted(expected)
    assert report["selected_count"] == len(expected)
    assert report["candidate_count"] == len(CANDIDATES)


@pytest.mark.parametrize("key", ["include_blocks", "exclude_blocks"])
@pytest.mark.parametrize(
    "value", ["*mlp*", None, True, 3, {}, ("*mlp*",), [1], [""], [" "], ["a\n"], ["a\x00"]]
)
def test_config_rejects_invalid_pattern_lists(tmp_path, key, value):
    with pytest.raises(ValueError, match=key):
        Config(path=tmp_path / "c.toml", training={key: value})


@pytest.mark.parametrize("extension", ["toml", "yaml"])
def test_native_configs_preserve_filters_and_defaults_are_independent(tmp_path, extension):
    path = tmp_path / f"c.{extension}"
    path.write_text(
        "[training]\ninclude_blocks=['*mlp*', '*.attn.q*']\nexclude_blocks=['*.0.*']\n"
        if extension == "toml"
        else "training:\n  include_blocks: ['*mlp*', '*.attn.q*']\n  exclude_blocks: ['*.0.*']\n"
    )
    cfg = load_config(path)
    assert cfg.training["include_blocks"] == ["*mlp*", "*.attn.q*"]
    assert cfg.training["exclude_blocks"] == ["*.0.*"]
    assert cfg.source_document["training"]["include_blocks"] == cfg.training["include_blocks"]
    json.dumps(cfg.to_dict())
    a, b = Config(path=path), Config(path=path)
    a.training["include_blocks"].append("*")
    assert b.training["include_blocks"] == []


@pytest.mark.parametrize(
    "options",
    [
        {"include_blocks": ["*.attn.q*"]},
        {"include_blocks": ["*MLP*"]},
        {"include_blocks": [".*mlp.*"]},  # regex is not a glob
        {"include_blocks": ["*.attn.to_q.weight"]},
        {"include_blocks": ["transformer_blocks.0"]},
        {"exclude_blocks": ["*"]},
        {"include_blocks": ["*mlp*"], "exclude_blocks": ["*mlp*"]},
    ],
)
def test_empty_selection_fails_closed(options):
    with pytest.raises(ValueError, match="沒有可訓練"):
        filter_lora_targets(CANDIDATES, options)


def test_empty_candidates_does_not_fall_back_to_everything():
    with pytest.raises(ValueError, match="找不到"):
        filter_lora_targets([], {})


def test_partial_misses_warn_and_are_recorded():
    with pytest.warns(UserWarning) as warnings:
        report = filter_lora_targets(
            CANDIDATES,
            {"include_blocks": ["*mlp*", "*.attn.q*"], "exclude_blocks": ["missing*"]},
        )
    assert len(warnings) == 2
    assert report["unmatched_patterns"] == {
        "include_blocks": ["*.attn.q*"],
        "exclude_blocks": ["missing*"],
    }
    assert report["selected_count"] == 6


@pytest.mark.parametrize("flag", ["--include_blocks", "--include-blocks"])
@pytest.mark.parametrize("values", [["['*mlp*', '*.attn.to_q']"], ["*mlp*", "*.attn.to_q"]])
def test_cli_overrides_propagate_through_main_without_rewriting_config(
    tmp_path, monkeypatch, capsys, flag, values
):
    path = tmp_path / "c.toml"
    source = "[training]\ninclude_blocks=['old*']\nexclude_blocks=['old*']\n"
    path.write_text(source)
    monkeypatch.setattr(cli, "verify_upstream", lambda: {})
    captured = []

    def train(cfg, **kwargs):
        captured.append((cfg, kwargs))
        return cfg.to_dict()

    monkeypatch.setattr(runtime, "train", train)
    assert (
        cli.main(
            ["train", "--config", str(path), "--dry-run", flag, *values, "--exclude-blocks", "[]"]
        )
        == 0
    )
    cfg, kwargs = captured[0]
    assert cfg.training["include_blocks"] == ["*mlp*", "*.attn.to_q"]
    assert cfg.training["exclude_blocks"] == []
    assert cfg.source_document["training"]["include_blocks"] == ["old*"]
    assert cfg.source_format == "toml"
    assert kwargs["dry_run"] is True
    assert path.read_text() == source
    assert json.loads(capsys.readouterr().out)["training"]["exclude_blocks"] == []


@pytest.mark.parametrize("value", ["[bad", "[1]", "['']", "[__import__('os').system('false')]"])
def test_cli_literal_list_is_strict_and_not_evaluated(value):
    with pytest.raises(ValueError):
        parse_cli_block_patterns([value], "--include_blocks")


def test_cli_omission_preserves_file_filter(tmp_path):
    cfg = Config(path=tmp_path / "c.toml", training={"exclude_blocks": ["*mlp*"]})
    assert cli._apply_block_filter_overrides(cfg, cli.parser().parse_args(["train"])) is cfg


@pytest.mark.parametrize("pattern", ["[tT]*", "[!x]*", "[t]ransformer_blocks.*"])
def test_cli_accepts_globs_starting_with_character_classes(pattern):
    assert parse_cli_block_patterns([pattern], "--include_blocks") == [pattern]


def test_changing_filters_preserves_dataset_and_cache_identity(tmp_path):
    from PIL import Image

    images = tmp_path / "images"
    images.mkdir()
    Image.new("RGB", (64, 64), "white").save(images / "one.png")
    (images / "one.txt").write_text("a fixture")
    cfg = Config(
        path=tmp_path / "c.toml", dataset={"path": str(images)},
        training={"cache_dir": str(tmp_path / "cache"), "output_dir": str(tmp_path / "out")},
    )
    original = data.prepare_dataset(cfg)
    filtered = cli._apply_block_filter_overrides(
        cfg, cli.parser().parse_args(["train", "--exclude_blocks", "*mlp*"]),
    )
    recovered = data.load_manifest(filtered)
    assert original["dataset_fingerprint"] == recovered["dataset_fingerprint"]
    assert runtime._cache_identity(original, "fixture-model") == runtime._cache_identity(
        recovered, "fixture-model"
    )


def tiny_pipe(dtype=torch.float32):
    from diffsynth.models.qwen_image_21_dit import QwenImage21DiT
    from diffsynth.pipelines.qwen_image_21 import QwenImage21Pipeline

    pipe = QwenImage21Pipeline(device="cpu", torch_dtype=dtype)
    pipe.dit = QwenImage21DiT(
        in_channels=4,
        out_channels=4,
        num_layers=2,
        num_attention_heads=2,
        attention_head_dim=32,
        context_in_dim=64,
        axes_dims_rope=(8, 12, 12),
    ).to(dtype=dtype)
    return pipe


def sample_tensors(dtype=torch.float32):
    return {
        "input_latents": torch.randn(1, 4, 4, 4, dtype=dtype),
        "prompt_embeds": torch.randn(1, 5, 64, dtype=dtype),
        "edit_image_pad_mask": torch.zeros(1, 5, dtype=torch.bool),
    }


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize(
    "filters,expected",
    [
        (
            {"include_blocks": ["*mlp*"], "exclude_blocks": ["*.1.*"]},
            {n for n in CANDIDATES[:7] if "mlp" in n},
        ),
        ({"exclude_blocks": ["*mlp*"]}, {n for n in CANDIDATES if "mlp" not in n}),
        (
            {"include_blocks": ["transformer_blocks.0.attn.to_q"]},
            {"transformer_blocks.0.attn.to_q"},
        ),
    ],
)
def test_real_peft_updates_only_selected_adapters_and_round_trips(
    tmp_path, dtype, filters, expected
):
    assert not torch.cuda.is_available()
    assert torch.get_num_threads() == 1
    torch.manual_seed(71)
    cfg = Config(path=tmp_path / "c.toml", training={"rank": 2, "alpha": 1, **filters})
    model = runtime.make_training_module(tiny_pipe(dtype), cfg.training, targets=TARGETS)
    actual = {n for n, m in model.pipe.dit.named_modules() if isinstance(m, LoraLayer)}
    assert actual == expected == set(model.lora_target_filter["selected_targets"])
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    assert trainable == {
        f"pipe.dit.{name}.lora_{part}.default.weight" for name in expected for part in ("A", "B")
    }
    optimizer = torch.optim.AdamW(model.trainable_modules(), lr=0.01)
    assert {id(p) for group in optimizer.param_groups for p in group["params"]} == {
        id(p) for p in model.parameters() if p.requires_grad
    }
    restored = copy.deepcopy(model)
    tensors = sample_tensors(dtype)
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        loss = model(tensors)
        assert torch.isfinite(loss)
        loss.backward()
        for n, p in model.named_parameters():
            if n in trainable:
                assert p.grad is not None and torch.isfinite(p.grad).all(), n
            else:
                assert p.grad is None, n
        optimizer.step()
    for n, p in model.named_parameters():
        assert (not torch.equal(before[n], p)) == (n in trainable), n
    checkpoint = tmp_path / "filtered.safetensors"
    save_checkpoint(model, checkpoint, {})
    loaded = read_checkpoint(checkpoint)
    assert {key.split(".lora_")[0] for key in loaded.training_state} == expected
    load_training_checkpoint(restored, checkpoint)
    assert all(
        torch.equal(value, restored.state_dict()[name])
        for name, value in model.state_dict().items()
    )
    incompatible = runtime.make_training_module(
        tiny_pipe(dtype),
        {**cfg.training, "include_blocks": [], "exclude_blocks": []},
        targets=TARGETS,
    )
    with pytest.raises(ValueError, match="targets 不一致"):
        load_training_checkpoint(incompatible, checkpoint)


@pytest.mark.parametrize(
    "filters,count",
    [
        ({}, 224),
        ({"include_blocks": ["*mlp*"]}, 96),
        ({"exclude_blocks": ["*mlp*"]}, 128),
        ({"include_blocks": ["*.attn.to_q"]}, 32),
        ({"include_blocks": ["transformer_blocks.0.*"], "exclude_blocks": ["*mlp*"]}, 4),
    ],
)
def test_production_auto_detection_and_injection_on_meta(filters, count):
    from diffsynth.models.qwen_image_21_dit import QwenImage21DiT
    from diffsynth.pipelines.qwen_image_21 import QwenImage21Pipeline

    pipe = QwenImage21Pipeline(device="cpu", torch_dtype=torch.float32)
    with torch.device("meta"):
        pipe.dit = QwenImage21DiT()
        model = runtime.make_training_module(pipe, {"rank": 2, **filters})
    assert model.lora_target_filter["candidate_count"] == 224
    assert model.lora_target_filter["selected_count"] == count
    assert sum(isinstance(m, LoraLayer) for m in pipe.dit.modules()) == count
    assert all(p.is_meta for p in model.parameters())


def test_cli_real_tiny_training_records_resolved_targets(tmp_path, monkeypatch, capsys):
    """Only pretrained asset/cache discovery is replaced; all training/IO is real."""
    path = tmp_path / "c.toml"
    path.write_text(
        '[model]\ndevice="cpu"\n[training]\nrank=2\nalpha=1\n'
        'cache_dir="cache"\noutput_dir="out"\nmax_steps=2\nsave_every=1\n'
    )
    cache = tmp_path / "cache"
    cache.mkdir()
    save_file(sample_tensors(), str(cache / "fixture.safetensors"))
    monkeypatch.setattr(cli, "verify_upstream", lambda: {})
    monkeypatch.setattr(data, "load_manifest", lambda cfg: {"dataset_fingerprint": "tiny-fixture"})
    monkeypatch.setattr(runtime, "model_assets", lambda cfg: {})
    monkeypatch.setattr(
        runtime,
        "load_cache_index",
        lambda *args: {
            "identity": {"model_signature": "tiny-fixture"},
            "items": [{"file": "fixture.safetensors"}],
        },
    )
    monkeypatch.setattr(runtime, "build_pipeline", lambda *args, **kwargs: tiny_pipe())
    make = runtime.make_training_module
    monkeypatch.setattr(
        runtime,
        "make_training_module",
        lambda pipe, training, **kw: make(pipe, training, targets=TARGETS, **kw),
    )
    assert (
        cli.main(
            [
                "train",
                "--config",
                str(path),
                "--include_blocks",
                "['*mlp*', '*.attn.to_q']",
                "--exclude_blocks",
                "['*.1.*']",
            ]
        )
        == 0
    )
    plan = json.loads((tmp_path / "out/run.json").read_text())
    expected = sorted(n for n in CANDIDATES[:7] if "mlp" in n or n.endswith("to_q"))
    assert plan["lora_target_filter"]["selected_targets"] == expected
    assert plan["lora_target_filter"]["selected_count"] == plan["adapter_count"] == 4
    checkpoint = read_checkpoint(tmp_path / "out/final.safetensors")
    assert {key.split(".lora_")[0] for key in checkpoint.training_state} == set(expected)
    assert (tmp_path / "out/completed.json").is_file()
    assert "LoRA targets: 4/14 selected" in capsys.readouterr().out


def test_filtered_smoke_uses_config_and_reload(tmp_path):
    from qwen21_trainer.smoke import run_smoke

    result = run_smoke(tmp_path / "smoke", training_options={"include_blocks": ["*.attn.to_q"]})
    for run in result["runs"]:
        assert run["lora_target_filter"]["selected_count"] == 2
        assert run["saved_lora_tensors"] == 4
        assert run["reload_max_abs_error"] == 0
