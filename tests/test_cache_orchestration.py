"""Cache orchestration tests with explicit encoder doubles, not pretrained outputs."""

from types import SimpleNamespace

import torch
from PIL import Image

from qwen21_trainer import runtime
from qwen21_trainer.config import Config
from qwen21_trainer.data import prepare_dataset


class EncoderDouble(torch.nn.Module):
    def __init__(self):
        super().__init__()
        from diffsynth.diffusion.flow_match import FlowMatchScheduler

        self.scheduler = FlowMatchScheduler("Qwen-Image")
        self.norm = torch.nn.Identity()
        self.text_encoder = SimpleNamespace(
            model=SimpleNamespace(
                model=SimpleNamespace(language_model=SimpleNamespace(norm=self.norm))
            )
        )
        self.units = [object()]
        self.calls = 0

    def unit_runner(self, unit, pipe, shared, positive, negative):
        self.calls += 1
        self.norm.register_forward_hook(lambda m, a, o: None)
        self.norm(torch.ones(1))
        shared["input_latents"] = torch.ones(
            1, 64, shared["height"] // 16, shared["width"] // 16, dtype=torch.bfloat16
        )
        positive.update(
            prompt_embeds=torch.ones(1, 5, 4096, dtype=torch.bfloat16),
            edit_image_pad_mask=torch.zeros(1, 5, dtype=torch.bool),
            prompt_embeds_mask=None,
        )
        return shared, positive, negative


def test_cache_resume_and_stale_per_file_identity(tmp_path, monkeypatch):
    images = tmp_path / "images"
    images.mkdir()
    Image.new("RGB", (64, 64), "red").save(images / "one.png")
    (images / "one.txt").write_text("one red image")
    cfg = Config(
        path=tmp_path / "config.toml",
        dataset={"path": str(images)},
        training={"cache_dir": str(tmp_path / "cache"), "output_dir": str(tmp_path / "out")},
    )
    prepare_dataset(cfg)
    monkeypatch.setattr(runtime, "model_assets", lambda cfg: {})
    monkeypatch.setattr(runtime, "source_signature", lambda cfg, assets: "model-A")
    double = EncoderDouble()
    roles = []

    def build(cfg, assets, requested):
        roles.append(requested)
        return double

    monkeypatch.setattr(runtime, "build_pipeline", build)
    index = runtime.cache_dataset(cfg)
    assert roles == [["text_encoder", "vae"]]
    assert double.calls == 1 and len(double.norm._forward_hooks) == 0
    assert index["items"][0]["latent_shape"] == [1, 64, 4, 4]
    assert runtime.cache_dataset(cfg) == index
    assert double.calls == 1  # matching cache does not re-encode
    runtime.load_cache_index(cfg, prepare_dataset(cfg), {})

    # Simulate an interrupted rebuild: pending has new identity but old tensor file remains.
    (tmp_path / "cache/cache-index.json").unlink()
    monkeypatch.setattr(runtime, "source_signature", lambda cfg, assets: "model-B")
    identity = runtime._cache_identity(prepare_dataset(cfg), "model-B")
    runtime.atomic_json(tmp_path / "cache/cache-pending.json", identity)
    newer = runtime.cache_dataset(cfg)
    assert double.calls == 2
    assert newer["identity"]["model_signature"] == "model-B"
    assert len(double.norm._forward_hooks) == 0
