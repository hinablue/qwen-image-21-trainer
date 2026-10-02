"""Instantiate the actual requested optimizer, with no silent AdamW fallback."""

from __future__ import annotations

import importlib.metadata

from .training_options import normalize_training_options


def build_optimizer(parameters, settings):
    settings = normalize_training_options(settings)
    kwargs = dict(settings["optimizer_params"])
    if "betas" in kwargs:
        kwargs["betas"] = tuple(kwargs["betas"])
    kwargs.update(lr=settings["learning_rate"], weight_decay=settings["weight_decay"])
    if settings["optimizer"] == "adopt_adv":
        from adv_optm import Adopt_adv

        optimizer = Adopt_adv(parameters, **kwargs)
    else:
        from torch.optim import AdamW

        optimizer = AdamW(parameters, **kwargs)
    return optimizer


def optimizer_report(optimizer, settings):
    settings = normalize_training_options(settings)
    return {
        "name": settings["optimizer"],
        "class": f"{type(optimizer).__module__}.{type(optimizer).__qualname__}",
        "package_version": importlib.metadata.version(
            "adv_optm" if settings["optimizer"] == "adopt_adv" else "torch"
        ),
        "learning_rate": optimizer.param_groups[0]["lr"],
        "weight_decay": optimizer.param_groups[0]["weight_decay"],
        "requested_params": settings["optimizer_params"],
        "effective_betas": list(optimizer.param_groups[0]["betas"]),
        "effective_eps": optimizer.param_groups[0]["eps"],
        "adopt_stochastic_rounding": getattr(optimizer, "stochastic_rounding", None),
        "adopt_use_atan2": getattr(optimizer, "use_atan2", None),
        "adopt_kourkoutas_beta": getattr(optimizer, "kourkoutas_beta", None),
        "effective_cautious_wd": optimizer.param_groups[0].get("cautious_wd"),
        "effective_state_precision": optimizer.param_groups[0].get("state_precision"),
    }
