import pytest

from qwen21_trainer.config import Config


def test_wandb_disabled_by_default(tmp_path):
    cfg = Config(path=tmp_path / "config.toml")
    assert cfg.wandb == {
        "enabled": False,
        "project": "qwen-image-21-trainer",
        "entity": "",
        "name": "",
        "mode": "online",
        "log_every": 1,
        "log_config": True,
        "log_samples": True,
        "log_training_log": True,
    }


@pytest.mark.parametrize(
    "settings",
    [
        {"enabled": 1},
        {"mode": "automatic"},
        {"project": ""},
        {"project": "a/b"},
        {"log_every": 0},
        {"log_every": True},
        {"api_key": "must-not-be-stored"},
    ],
)
def test_bad_tracking_config_rejected(tmp_path, settings):
    with pytest.raises(ValueError):
        Config(path=tmp_path / "config.toml", wandb=settings)
