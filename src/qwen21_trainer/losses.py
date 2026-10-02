"""Flow Matching objectives; default dispatches to the unchanged upstream loss."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class HaarWaveletLoss(nn.Module):
    """ai-toolkit-style single-level Haar, zero padding, equally weighted LL/LH/HL/HH.

    On even H/W this orthonormal, equal-band mean loss is Parseval-equivalent to
    velocity MSE (up to floating-point error); it does NOT upweight high frequencies.
    The conditioning target uses the sampled training noise, as ai-toolkit does.
    """

    def __init__(self):
        super().__init__()
        from pytorch_wavelets import DWTForward

        self.dwt = DWTForward(J=1, mode="zero", wave="haar").float()

    def _bands(self, value):
        low, high = self.dwt(value)
        return torch.cat([low, *high[0].unbind(dim=2)], dim=1)

    def forward(self, prediction, clean_latents, noise):
        if (
            prediction.ndim != 4
            or prediction.shape != clean_latents.shape
            or noise.shape != prediction.shape
        ):
            raise ValueError(
                "wavelet loss 要求 prediction／latents／noise 為同 shape 的 BCHW tensors。"
            )
        self.dwt.to(device=prediction.device, dtype=torch.float32)
        with torch.autocast(device_type=prediction.device.type, enabled=False):
            with torch.no_grad():
                target = self._bands(clean_latents.float())
            recovered = noise.float() - prediction.float()
            return F.mse_loss(self._bands(recovered), target)


def flow_matching_loss(pipe, inputs, *, loss_type="mse", weighting="diffsynth", wavelet=None):
    if loss_type not in ("mse", "wavelet") or weighting not in ("diffsynth", "none"):
        raise ValueError("不支援的 loss 或 weighting。")
    if loss_type == "mse" and weighting == "diffsynth":
        from diffsynth.diffusion.loss import FlowMatchSFTLoss

        return FlowMatchSFTLoss(pipe, **inputs)
    # Image-only counterpart of the pinned upstream sampling/forward sequence.
    # No video, image-to-LoRA hot loading, or scheduler-direction changes here.
    if "lora" in inputs or "first_frame_latents" in inputs:
        raise ValueError("這個 loss 分支只支援 T2I LoRA。")
    shared = dict(inputs)
    length = len(pipe.scheduler.timesteps)
    minimum = int(shared.get("min_timestep_boundary", 0) * length)
    maximum = int(shared.get("max_timestep_boundary", 1) * length)
    timestep_id = torch.randint(minimum, maximum, (1,))
    timestep = pipe.scheduler.timesteps[timestep_id].to(dtype=pipe.torch_dtype, device=pipe.device)
    clean = shared["input_latents"]
    noise = torch.randn_like(clean) * shared.get("noise_scale", 1.0)
    shared["latents"] = pipe.scheduler.add_noise(clean, noise, timestep)
    models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
    prediction = pipe.model_fn(**models, **shared, timestep=timestep)
    if loss_type == "wavelet":
        if wavelet is None:
            raise ValueError("wavelet loss module 未初始化。")
        loss = wavelet(prediction, clean, noise)
    else:
        target = pipe.scheduler.training_target(clean, noise, timestep)
        loss = F.mse_loss(prediction.float(), target.float())
    if weighting == "diffsynth":
        loss = loss * pipe.scheduler.training_weight(timestep)
    return loss
