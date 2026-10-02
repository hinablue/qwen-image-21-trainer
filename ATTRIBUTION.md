# Attribution and implementation boundary

This project vendors the Apache-2.0-licensed Python package from
https://github.com/modelscope/DiffSynth-Studio at revision
`974cfa37f27ac55eba3b6d10efa21f876900572d` (ModelScope Team).
The original license is retained at `vendor/DiffSynth-Studio/LICENSE` and `LICENSE`.
The upstream Qwen 2.1 training example is retained under the vendor tree for reference.

`upstream-manifest.json` records SHA-256 of the copied files. `qwen21 verify-upstream`
checks the installed upstream Python files before model operations. No vendored Python
source has been intentionally modified. Keep attribution and source manifests when redistributing.

The new wrapper in `src/qwen21_trainer` provides strict configuration, local dataset
manifests, safe tensor caches, exact optimizer-step budgets, logs and CLI commands.
It calls upstream QwenImage21Pipeline, QwenImage21DiT, DiffusionTrainingModule's
PEFT injection, FlowMatchScheduler and FlowMatchSFTLoss; it does not reimplement their
architecture. The default MSE path keeps the original upstream loss. An optional
wavelet path follows the ai-toolkit Haar/zero-padding/four-band objective, and the
optimizer factory can directly instantiate adv_optm.Adopt_adv. The cache wrapper
cleans up newly added TE capture hooks without editing upstream source.

The Comfy BF16 VAE key mapping and wavelet objective are based on ai-toolkit
(https://github.com/ostris/ai-toolkit), Copyright (c) 2024 Ostris, LLC, MIT-licensed.
Its complete notice is retained at `licenses/ai-toolkit-MIT.txt`. The local wavelet
reference was read at ai-toolkit revision `963b608cb2bf7bcde4cbd33f2f352046cf845aeb`;
the Qwen 2.1 Comfy mapping was inspected at `ecee894ed2b1f3716d9d7326693061ec1a3105bb`.
Dependencies adv_optm and pytorch_wavelets retain their own distributed notices.

Model weights are not included. Qwen model licenses and usage terms are independent
of this code license and remain those of their original publishers.
