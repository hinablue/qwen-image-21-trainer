"""Tiny fixtures only: no pretrained files, downloads, CUDA, or model saves."""

from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save

from qwen21_trainer import model_io as io


def write_fixture(path, values, metadata=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(save(values, metadata=metadata))
    return path


def descriptor(shape, dtype="BF16"):
    return {"shape": shape, "dtype": dtype}


def config(root, fmt="comfy_bf16", **kwargs):
    return SimpleNamespace(
        model={"root": root, "format": fmt, "device": "cpu", "cpu_offload": False, **kwargs}
    )


def tiny_tree(root, fmt="comfy_bf16"):
    processor = root / "processor"
    processor.mkdir(parents=True)
    (processor / "tokenizer_config.json").write_text("{}")
    (processor / "vocab.txt").write_text("test")
    role_values = {
        "transformer": {
            "transformer_blocks.0.img_mlp.gate_up.weight": torch.arange(24).reshape(6, 4).bfloat16()
        },
        "text_encoder": {"model.norm.weight": torch.arange(4).bfloat16()},
        "vae": {"encoder.conv1.weight": torch.arange(24).reshape(2, 3, 1, 2, 2).bfloat16()},
    }
    assets = {"format": fmt, "processor": processor}
    for role, values in role_values.items():
        if fmt == "official":
            name = (
                "model.safetensors"
                if role == "text_encoder"
                else "diffusion_pytorch_model.safetensors"
            )
            path = root / role / name
            values = {"transformer_blocks.0.img_mlp.gate_layer.weight": torch.ones(2, 2).bfloat16()}
        else:
            path = root / io.COMFY_FILES[role]
        assets[role] = [write_fixture(path, values)]
    return assets


def use_tiny_shapes(monkeypatch):
    expected = {
        "transformer": {
            "transformer_blocks.0.img_mlp.gate_layer.weight": (3, 4),
            "transformer_blocks.0.img_mlp.proj.weight": (3, 4),
        },
        "text_encoder": {"model.model.language_model.norm.weight": (4,)},
        "vae": {"encoder.conv_in.weight": (2, 3, 2, 2)},
    }
    monkeypatch.setattr(io, "_expected_shapes", lambda role: expected[role])
    return expected


def test_dit_split_views_and_unmodified_identity():
    source = torch.arange(24).reshape(6, 4).bfloat16()
    bias = torch.ones(4).bfloat16()
    raw = {"transformer_blocks.0.img_mlp.gate_up.weight": source, "bias": bias}
    plan = io.comfy_mapping_plan("transformer", {k: descriptor(v.shape) for k, v in raw.items()})
    mapped = {k: v.apply(raw[v.source]) for k, v in plan.items()}
    gate = mapped["transformer_blocks.0.img_mlp.gate_layer.weight"]
    proj = mapped["transformer_blocks.0.img_mlp.proj.weight"]
    assert torch.equal(gate, source[:3]) and torch.equal(proj, source[3:])
    assert (
        gate.untyped_storage().data_ptr()
        == proj.untyped_storage().data_ptr()
        == source.untyped_storage().data_ptr()
    )
    assert proj.storage_offset() == 12
    assert mapped["bias"] is bias
    assert all(v.dtype == torch.bfloat16 for v in mapped.values())


@pytest.mark.parametrize("shape", [(5, 4), (6, 4, 1), (0, 4)])
def test_invalid_fused_shapes(shape):
    with pytest.raises(ValueError, match="gate_up"):
        io.comfy_mapping_plan("transformer", {"x.img_mlp.gate_up.weight": descriptor(shape)})


def test_mapping_collision():
    with pytest.raises(ValueError, match="重複"):
        io.comfy_mapping_plan(
            "transformer",
            {
                "x.img_mlp.gate_up.weight": descriptor((6, 4)),
                "x.img_mlp.proj.weight": descriptor((3, 4)),
            },
        )


def test_te_native_then_official_converter_exactly_once():
    raw = {
        key: descriptor((2,))
        for key in [
            "model.layers.0.mlp.gate_proj.weight",
            "model.embed_tokens.weight",
            "model.norm.weight",
            "model.visual.patch_embed.proj.weight",
            "lm_head.weight",
        ]
    }
    plan = io.comfy_mapping_plan("text_encoder", raw)
    assert "model.language_model.norm.weight" in plan
    assert not any(k.startswith("model.model.") for k in plan)
    final = io._wrapper_state("text_encoder", plan)
    assert set(final) == {
        "model.model.language_model.layers.0.mlp.gate_proj.weight",
        "model.model.language_model.embed_tokens.weight",
        "model.model.language_model.norm.weight",
        "model.model.visual.patch_embed.proj.weight",
        "model.lm_head.weight",
    }
    assert len(final) == len(raw)


@pytest.mark.parametrize(
    "key", ["other.weight", "model.language_model.norm.weight", "model.model.norm.weight"]
)
def test_unknown_te_source_rejected(key):
    with pytest.raises(ValueError, match="不明"):
        io.comfy_mapping_plan("text_encoder", {key: descriptor((2,))})


@pytest.mark.parametrize(
    "source,target",
    [
        ("conv1.weight", "quant_conv.weight"),
        ("conv2.bias", "post_quant_conv.bias"),
        ("encoder.conv1.weight", "encoder.conv_in.weight"),
        ("decoder.head.0.gamma", "decoder.norm_out.gamma"),
        ("encoder.head.2.weight", "encoder.conv_out.weight"),
        ("encoder.middle.0.residual.0.gamma", "encoder.mid_block.resnets.0.norm1.gamma"),
        ("decoder.middle.2.residual.3.gamma", "decoder.mid_block.resnets.1.norm2.gamma"),
        ("encoder.middle.1.to_qkv.weight", "encoder.mid_block.attentions.0.to_qkv.weight"),
        (
            "encoder.downsamples.0.downsamples.0.residual.2.weight",
            "encoder.down_blocks.0.resnets.0.conv1.weight",
        ),
        (
            "decoder.upsamples.1.upsamples.2.residual.6.bias",
            "decoder.up_blocks.1.resnets.2.conv2.bias",
        ),
        (
            "encoder.downsamples.1.downsamples.0.shortcut.weight",
            "encoder.down_blocks.1.resnets.0.conv_shortcut.weight",
        ),
        (
            "encoder.downsamples.0.downsamples.2.resample.1.weight",
            "encoder.down_blocks.0.downsampler.resample.1.weight",
        ),
        (
            "decoder.upsamples.0.upsamples.3.time_conv.weight",
            "decoder.up_blocks.0.upsampler.time_conv.weight",
        ),
    ],
)
def test_vae_key_families(source, target):
    assert io._vae_key(source) == target


@pytest.mark.parametrize(
    "key",
    [
        "garbage",
        "encoder.head.1.weight",
        "decoder.middle.4.weight",
        "encoder.downsamples.0.downsamples.9.resample.1.weight",
        "encoder.unknown.weight",
    ],
)
def test_vae_unknown_keys(key):
    with pytest.raises(ValueError, match="不明"):
        io._vae_key(key)


def test_vae_only_singleton_squeeze_view():
    source = torch.arange(24).reshape(2, 3, 1, 2, 2).bfloat16()
    view = io.comfy_mapping_plan("vae", {"conv1.weight": descriptor(source.shape)})[
        "quant_conv.weight"
    ]
    result = view.apply(source)
    assert result.shape == (2, 3, 2, 2)
    assert result.untyped_storage().data_ptr() == source.untyped_storage().data_ptr()
    assert torch.equal(result.reshape(-1), source.reshape(-1))
    with pytest.raises(ValueError, match="size-1"):
        io.comfy_mapping_plan("vae", {"conv1.weight": descriptor((2, 3, 2, 2, 2))})


@pytest.mark.parametrize("role", io.ROLES)
def test_cpu_tiny_load_preserves_header_stat_dtype_values(tmp_path, monkeypatch, role):
    use_tiny_shapes(monkeypatch)
    assets = tiny_tree(tmp_path)
    path = assets[role][0]
    before = (path.read_bytes(), path.stat().st_mtime_ns)
    state = io.load_comfy_state_dict(role, assets[role])
    assert state
    assert all(v.dtype == torch.bfloat16 and v.device.type == "cpu" for v in state.values())
    assert before == (path.read_bytes(), path.stat().st_mtime_ns)
    if role == "transformer":
        assert torch.equal(
            torch.cat(list(state.values())), torch.arange(24).reshape(6, 4).bfloat16()
        )
    elif role == "text_encoder":
        assert torch.equal(state["model.language_model.norm.weight"], torch.arange(4).bfloat16())
    else:
        assert torch.equal(
            state["encoder.conv_in.weight"], torch.arange(24).reshape(2, 3, 2, 2).bfloat16()
        )


@pytest.mark.parametrize("dtype", ["F32", "F16", "F8_E4M3", "I8", "U8"])
def test_all_non_bf16_rejected(dtype):
    with pytest.raises(ValueError, match="BF16"):
        io._require_bf16({"weight": descriptor((2,), dtype)}, {}, "fixture")


@pytest.mark.parametrize(
    "marker",
    [
        "layer.comfy_quant",
        "layer.weight_scale",
        "layer.weight_scale_2",
        "layer.qweight",
        "layer.scale_weight",
        "quantization_config",
    ],
)
def test_quant_markers_rejected_even_if_bf16(marker):
    with pytest.raises(ValueError, match="BF16"):
        io._require_bf16({marker: descriptor((2,))}, {}, "fixture")


@pytest.mark.parametrize(
    "metadata",
    [
        {"quantization": "none"},
        {"format": "fp8"},
        {"format": "gguf"},
        {"format": '{"format":"fp8"}'},
        {"comfy_quant": '{"format":"int8_tensorwise"}'},
    ],
)
def test_quant_metadata_rejected(metadata):
    with pytest.raises(ValueError, match="metadata"):
        io._require_bf16({"weight": descriptor((2,))}, metadata, "fixture")


@pytest.mark.parametrize("mutation", ["missing", "extra", "shape"])
def test_complete_shapes_fail_closed(monkeypatch, mutation):
    use_tiny_shapes(monkeypatch)
    actual = {"encoder.conv_in.weight": (2, 3, 2, 2)}
    if mutation == "missing":
        actual = {}
    elif mutation == "extra":
        actual["extra"] = (1,)
    else:
        actual["encoder.conv_in.weight"] = (3, 3, 2, 2)
    with pytest.raises(ValueError, match="完整 key/shape"):
        io._assert_shapes("vae", actual)


def test_mixed_fused_split_source_rejected(tmp_path, monkeypatch):
    entries = {
        "transformer_blocks.0.img_mlp.gate_up.weight": descriptor((6, 4)),
        "transformer_blocks.1.img_mlp.gate_layer.weight": descriptor((3, 4)),
        "transformer_blocks.1.img_mlp.proj.weight": descriptor((3, 4)),
    }
    monkeypatch.setattr(io, "_header", lambda path: (entries, {}))
    with pytest.raises(ValueError, match="split/fused"):
        io.validate_comfy_headers("transformer", [tmp_path / "fixture"])


def test_processor_symlink_rename_invalidates_signature(tmp_path):
    assets = tiny_tree(tmp_path / "models")
    target = tmp_path / "target.json"
    target.write_text("{}")
    link = assets["processor"] / "preprocessor_config.json"
    link.symlink_to(target)
    cfg = config(tmp_path / "models")
    first = io.source_signature(cfg, assets)
    link.rename(link.with_name("processor_config.json"))
    assert io.source_signature(cfg, assets) != first


def test_assets_and_unknown_format(tmp_path, monkeypatch):
    use_tiny_shapes(monkeypatch)
    assets = tiny_tree(tmp_path)
    assert io.model_assets(config(tmp_path)) == assets
    with pytest.raises(ValueError, match="model.format"):
        io.model_assets(config(tmp_path, "unknown"))


def test_missing_assets_do_not_download(tmp_path):
    with pytest.raises(FileNotFoundError):
        io.model_assets(config(tmp_path))


def test_official_regression_and_processor_override(tmp_path):
    assets = tiny_tree(tmp_path, "official")
    cfg = SimpleNamespace(model={"root": tmp_path})
    assert io.model_assets(cfg) == assets
    external = tmp_path / "external"
    external.mkdir()
    (external / "tokenizer_config.json").write_text("{}")
    cfg.model["processor_path"] = str(external)
    assert io.model_assets(cfg)["processor"] == external
    write_fixture(
        assets["transformer"][0], {"x.img_mlp.gate_up.weight": torch.ones(2, 2).bfloat16()}
    )
    with pytest.raises(ValueError, match="split-MLP"):
        io.model_assets(cfg)


def test_signature_binds_external_processor_all_content_paths_and_format(tmp_path):
    root = tmp_path / "models"
    assets = tiny_tree(root)
    external = tmp_path / ".cache" / "huggingface" / "processor"
    external.parent.mkdir(parents=True)
    assets["processor"].rename(external)
    assets["processor"] = external
    cfg = config(root, processor_path=str(external))
    first = io.source_signature(cfg, assets)
    (external / "tokenizer_config.json").write_text('{"new":true}')
    second = io.source_signature(cfg, assets)
    assert second != first
    (external / "vocab.txt").write_text("changed")
    third = io.source_signature(cfg, assets)
    assert third != second
    # Unrelated shared-root JSON files should not invalidate this model's caches.
    (root / "unrelated.json").write_text("{}")
    assert io.source_signature(cfg, assets) == third
    p = assets["vae"][0]
    import os

    os.utime(p, ns=(p.stat().st_atime_ns, p.stat().st_mtime_ns + 1000))
    assert io.source_signature(cfg, assets) != third
    same_content_new_path = tmp_path / "processor-copy"
    external.rename(same_content_new_path)
    moved = dict(assets, processor=same_content_new_path)
    assert io.source_signature(cfg, moved) != io.source_signature(cfg, assets)
    assert io.source_signature(config(root, "official"), moved) != io.source_signature(cfg, moved)


def test_official_signature_tracks_model_json(tmp_path):
    assets = tiny_tree(tmp_path, "official")
    cfg = config(tmp_path, "official")
    first = io.source_signature(cfg, assets)
    (tmp_path / "transformer/config.json").write_text("{}")
    assert io.source_signature(cfg, assets) != first


@pytest.mark.parametrize(
    "roles,training,offload",
    [
        (("transformer",), True, True),
        (("text_encoder", "vae"), False, False),
        (("vae",), False, False),
        ((), False, False),
    ],
)
def test_comfy_role_specific_pipeline_load_only_requested(
    tmp_path, monkeypatch, roles, training, offload
):
    from diffsynth.models.model_loader import ModelPool
    from diffsynth.pipelines.qwen_image_21 import QwenImage21Pipeline

    from qwen21_trainer import runtime

    assets = tiny_tree(tmp_path)
    use_tiny_shapes(monkeypatch)
    calls, skeleton_calls = [], []
    pipe = SimpleNamespace(
        dit=None, text_encoder=None, vae=None, check_vram_management_state=lambda: False
    )
    monkeypatch.setattr(runtime, "check_device", lambda cfg: torch.device("cuda"))

    def skeleton(**kwargs):
        skeleton_calls.append(kwargs)
        return pipe

    monkeypatch.setattr(QwenImage21Pipeline, "from_pretrained", skeleton)

    def loader(self, registry, paths, vram, **kwargs):
        name = registry["model_name"]
        calls.append((name, paths, vram, kwargs["state_dict"]))
        return object()

    monkeypatch.setattr(ModelPool, "load_model_file", loader)
    result = io.build_pipeline(
        config(tmp_path, cpu_offload=offload), assets, roles, training=training
    )
    assert result is pipe
    assert [x[0] for x in calls] == ["qwen_image_21_" + io.ATTRS[r] for r in roles]
    assert skeleton_calls[0]["model_configs"] == []
    assert bool(skeleton_calls[0]["processor_config"]) == ("text_encoder" in roles)
    target_device = "cpu" if training and offload else "cuda"
    assert skeleton_calls[0]["device"] == target_device
    for role, (_, paths, vram, state) in zip(roles, calls):
        assert paths == [str(p) for p in assets[role]]
        assert vram["computation_device"] == target_device
        assert vram["computation_dtype"] == torch.bfloat16
        assert state
    for role in set(io.ROLES) - set(roles):
        assert getattr(result, io.ATTRS[role]) is None


def test_official_pipeline_uses_original_from_pretrained(tmp_path, monkeypatch):
    from diffsynth.pipelines.qwen_image_21 import QwenImage21Pipeline

    from qwen21_trainer import runtime

    assets = tiny_tree(tmp_path, "official")
    captured = {}

    def original(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(dit=object(), text_encoder=None, vae=None)

    monkeypatch.setattr(QwenImage21Pipeline, "from_pretrained", original)
    monkeypatch.setattr(runtime, "check_device", lambda cfg: torch.device("cpu"))
    monkeypatch.setattr(
        io, "load_comfy_state_dict", lambda *a: pytest.fail("official must not remap")
    )
    io.build_pipeline(config(tmp_path, "official"), assets, ["transformer"])
    assert len(captured["model_configs"]) == 1
    assert captured["model_configs"][0].path == [str(p) for p in assets["transformer"]]
    assert captured["processor_config"] is None


@pytest.mark.parametrize("roles", [("unknown",), ("vae", "vae")])
def test_bad_roles_rejected(tmp_path, roles):
    with pytest.raises(ValueError, match="roles"):
        io.build_pipeline(config(tmp_path), {}, roles)
