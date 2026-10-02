import pytest
import torch
from test_runtime import ScalarModel, settings

from qwen21_trainer.config import Config
from qwen21_trainer.runtime import optimization_loop


def test_public_num_warmup_config_and_optimizer_step_lr(tmp_path):
    cfg = Config(
        path=tmp_path / "c.toml",
        training={
            "rank": 32,
            "alpha": 16,
            "lr_scheduler": "constant_with_warmup",
            "num_warmup": 2,
            "max_steps": 6,
            "learning_rate": 0.01,
            "gradient_accumulation_steps": 3,
        },
    )
    model = ScalarModel()
    rows = optimization_loop(model, [torch.tensor(2.0)], cfg.training)
    assert model.calls == 18
    assert [r["learning_rate"] for r in rows] == pytest.approx([0, 0.005, 0.01, 0.01, 0.01, 0.01])
    assert [r["images_seen"] for r in rows] == [3, 6, 9, 12, 15, 18]
    for i, row in enumerate(rows):
        assert row["loss_average"] == pytest.approx(sum(r["loss"] for r in rows[: i + 1]) / (i + 1))


def test_short_smoke_can_use_long_schedule_prefix_without_extending_budget():
    model = ScalarModel()
    row = optimization_loop(
        model,
        [torch.tensor(2.0)],
        settings(max_steps=3),
        lr_settings={"max_steps": 100, "lr_scheduler": "cosine", "num_warmup": 20},
    )
    assert len(row) == 3 and model.calls == 9
    assert [r["learning_rate"] for r in row] == pytest.approx([0, 0.01 / 20, 0.01 * 2 / 20])


@pytest.mark.parametrize("scheduler,num", [("constant", 1), ("cosine", 10), ("diffsynth", 1)])
def test_config_rejects_misleading_warmup(tmp_path, scheduler, num):
    with pytest.raises(ValueError):
        Config(
            path=tmp_path / "c.toml",
            training={"max_steps": 10, "lr_scheduler": scheduler, "num_warmup": num},
        )
