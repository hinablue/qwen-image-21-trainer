# Qwen-Image 2.1 LoRA trainer for small datasets

[正體中文](README.md) | [English](README.en.md)

A standalone training wrapper around a pinned version of **DiffSynth-Studio**. It does not depend on ai-toolkit or modify an existing trainer. Use a small set of images to check how a LoRA learns before committing to a larger run.

- Pinned upstream: `modelscope/DiffSynth-Studio@974cfa37f27ac55eba3b6d10efa21f876900572d`.
- Includes the upstream Python package source, not an isolated `train.py` that cannot run on its own. Shared code and model registrations for other models remain in the vendor tree, but the public CLI **only supports Qwen-Image 2.1 text-to-image (T2I) LoRA**.
- Initial scope: **one GPU training process, with optional CPU prefetch workers, and one image per microstep**. Gradient accumulation is supported; editing datasets, multi-GPU training, LoKr, quantized training, and full optimizer-state resume are not.
- LoRAs are not directly interchangeable with ai-toolkit's fused `gate_up` LoRA format. Validate this trainer's output with the included DiffSynth `sample` command first.

## Installation

```bash
git clone https://github.com/hinablue/qwen-image-21-trainer.git
cd qwen-image-21-trainer
uv sync --locked --group dev
./run.sh --help
./run.sh verify-upstream
```

Requires Python 3.11 or 3.12 and uv. To move the project to another host without cloning, copy **the entire project, including vendor**. You do not need to copy `.venv`; recreate it with:

```bash
uv sync --locked --group dev
```

`uv.lock` pins the dependencies. Installation has been verified on Linux aarch64 with Python 3.11; other operating systems and GPU platforms have not been tested. Do not install the standalone wheel and substitute an arbitrary DiffSynth version from PyPI. The CLI checks the hashes of the installed upstream Python files.

## Public examples and local configuration

`configs/small.toml` contains an MSE + AdamW baseline. `configs/wavelet-adopt.example.toml` demonstrates wavelet + Adopt_adv with rank=32 and alpha=16. Weights & Biases (W&B) and sampling during training are disabled by default in the public examples.

Copy an example to `configs/my-run.toml`, set your own paths, and add `--config configs/my-run.toml` to each command. Only allowlisted public examples under `configs/` are tracked in Git; other configurations and import provenance are excluded by default. Models, datasets, training artifacts, credentials, and private experiment notes are also excluded from the repository. See [Repository hygiene and privacy](docs/repository-hygiene.md). For option definitions, see [Wavelet loss and Adopt_adv](docs/wavelet-adopt.md).

## Getting started

### 1. Add images and matching captions

```text
data/train/
  image01.png
  image01.txt
  image02.jpg
  image02.txt
```

- Each `.txt` file must contain a nonempty UTF-8 caption.
- Supported formats: PNG, JPG, JPEG, WEBP, and BMP. Subdirectories are scanned recursively by default.
- Original images and captions are not modified. Invalid images, missing captions, multiple images with the same filename stem in one directory, symlinks, and other ambiguous inputs are rejected.
- One image is enough to test the workflow. To evaluate whether the model learns a person or style, use a representative small dataset and compare outputs using fixed sample prompts held out from training.

### 2. Edit `configs/small.toml`

Start by setting the model and dataset paths:

```toml
[model]
format = "official"
root = "/absolute/path/to/Qwen-Image-2.1"

[dataset]
path = "/absolute/path/to/my_small_dataset"
```

Edit the existing sections rather than adding duplicate sections. **Relative paths in the configuration resolve from the TOML/YAML file's directory**, not the shell's current working directory.

Default settings for a small test run:

- Maximum image area: `262144` pixels (512×512). Aspect ratio is preserved, with dimensions aligned to 32 px using the upstream resize/crop rules; images are not all forced into squares.
- `rank = 32`; alpha defaults to rank when omitted, but you can set an independent value such as `alpha = 16`.
- Learning rate (LR): `1e-4`, AdamW, weight decay `0.01`.
- `max_steps = 300` means exactly **300 optimizer updates**, not epochs or microsteps.
- `gradient_accumulation_steps = 1`; setting it to 4 uses 1200 microsteps for 300 updates.
- Checkpoints are saved every 100 updates; gradient checkpointing is enabled.
- The diffusion transformer (DiT) and LoRA use BF16, without quantization.

### 3. Choose a model format

**Comfy BF16 files used by ai-toolkit:** Set `model.format="comfy_bf16"`, point `model.root` to the models directory containing `diffusion_models`, `text_encoders`, and `vae`, and set `model.processor_path` to the official processor directory. See [Model loading](docs/model-loading.md) for a complete example. Loading converts keys, splits gate_up, and removes the size-1 temporal axis without modifying the source or saving another full model copy. Only unquantized BF16 files are accepted; quantized files are rejected.

**Official split format:** Set `model.format="official"` (the default) and use this directory layout:

```text
Qwen-Image-2.1/
  transformer/diffusion_pytorch_model*.safetensors
  text_encoder/model*.safetensors
  vae/diffusion_pytorch_model*.safetensors
  processor/tokenizer_config.json
  processor/...
```

**The adapter does not support ComfyUI INT8, FP8, ConvRot, or other quantized files.** BF16 files with fused `gate_up` can be loaded through the Comfy mode described above. Splitting the base model weights does not make fused LoRAs directly interchangeable.

If you already have the complete official model directory, set `model.root`; no download is needed. Only download when you are using the official format and do not have the model. You must explicitly start the download:

```bash
./run.sh download --confirm-large-download
```

This downloads tens of GB of official model files and **does not start training**. The command first resolves the Hugging Face revision to a commit and writes download provenance into the model directory. You can also pass `--revision <commit>`. Other commands do not automatically download models.

### 4. Prepare, cache, and train

```bash
./run.sh prepare
./run.sh cache
./run.sh train --dry-run
./run.sh train
```

To use another configuration, add `--config /path/to/experiment.toml` to every command.

- `prepare`: Validates all images and captions, computes content hashes, and creates `cache/small/dataset.json`.
- `cache`: Loads only the **text encoder (TE) and variational autoencoder (VAE)**. Uses the upstream image preprocessing and conditioning pipeline, preserves RGBA, and uses the posterior mean. Saves the resulting cache as safetensors.
- `train --dry-run`: Checks dataset, model, and cache consistency and prints the effective configuration. Does not load the full model onto the GPU or run training.
- `train`: Loads only the **DiT** by default and trains from the cache. When you explicitly enable `sample.enabled`, the TE/VAE are also loaded during sampling.

The `cache` stage still needs memory for the TE and VAE; sampling needs the full pipeline. Caching reduces the models kept in memory during training, but does not mean the workflow can run with arbitrarily little VRAM.

After changing images, captions, resolution, or the model, rebuild:

```bash
./run.sh prepare --overwrite
./run.sh cache --overwrite
```

Model-weight provenance uses file paths, sizes, and modification times; processor/config files use content hashes. The trainer does not hash every byte of the tens of GB of model weights. Dataset images, captions, and the cache itself have content hashes.

### 5. Generate validation images

Generate a base image first, then a LoRA image using the same prompt and seed:

```bash
./run.sh sample --output output/base.png --prompt "your fixed validation prompt"
./run.sh sample --lora output/small/final.safetensors \
  --output output/lora.png --prompt "your fixed validation prompt"
```

Images are saved as PNG to preserve alpha; existing files are never overwritten. Set sampling options in `[sample]`. Defaults are seed 42, 30 steps, classifier-free guidance (CFG) 3.0, and 512×512. This CFG applies to sampling, not training.

## Rank, alpha, and LR schedules

Add these settings to the existing `[training]` section; do not create a duplicate TOML table:

```toml
rank = 32
alpha = 16
lr_scheduler = "constant_with_warmup"
num_warmup = 100
```

Alpha takes effect in the PEFT forward and backward passes. Checkpoints store raw A/B weights and per-layer alpha; warm-starting with the same rank/alpha does not apply scaling twice. `configs/wavelet-adopt.example.toml` demonstrates rank=32, alpha=16, and a separate output directory. Choose the LR schedule explicitly for each experiment; omitting it preserves the diffsynth baseline. See `docs/lora-alpha.md` and `docs/lr-schedule.md` for details.

W&B retains the original train/* metrics and adds Krea2-style loss/current, loss/average (the cumulative run average), lr/dit (the LR used for that update), sample_N, and a metrics.csv artifact containing numeric metrics. It does not fabricate epoch/DPO/TQD metrics or upload the entire console log.

## Select blocks and layers to train

Set arrays in the existing `[training]` section. Both TOML and YAML are supported:

```toml
# Train only MLP and attention Q targets; matching any include pattern is enough.
include_blocks = ["*mlp*", "*.attn.to_q"]
# Then exclude block 0; exclude wins when both include and exclude match.
exclude_blocks = ["transformer_blocks.0.*"]
```

- Omitted or empty `include_blocks` (`[]`): Keep the default eligible LoRA targets. A nonempty list **trains only matching targets**; no exclude-all pattern is needed.
- Omitted or empty `exclude_blocks` (`[]`): No additional exclusions. A nonempty list excludes targets matching any listed pattern.
- Patterns are case-sensitive **globs**, not Python regular expressions: `*` matches any number of characters, `?` matches one character, and `[01]` matches a character set. A dot (`.`) is a literal character; `*` can span multiple dot-separated levels.
- Patterns match the **full module path before LoRA attachment**, without `pipe.dit.`, `.weight`, or `.lora_A...`. Filtering only narrows the existing eligible Linear layers; it does not add LayerNorm, embeddings, or layers outside the blocks.
- Filtering happens **before** PEFT adapter creation. Unselected layers get no adapters, their base weights remain frozen, and they are excluded from the optimizer. Their forward passes still run.
- Any pattern that matches no candidate layer produces a warning. If no trainable targets remain, training fails rather than falling back to training everything.

Examples of actual module names in this Qwen-Image 2.1 version:

```text
transformer_blocks.0.attn.to_q
transformer_blocks.0.attn.to_k
transformer_blocks.0.attn.to_v
transformer_blocks.0.attn.to_out.0
transformer_blocks.0.img_mlp.proj
transformer_blocks.0.img_mlp.out
transformer_blocks.0.img_mlp.gate_layer
```

`*.attn.q*` therefore **does not match** this version's Q projection; use `*.attn.to_q`. To select a whole block, use `transformer_blocks.0.*`. The full model has 224 adapter pairs by default: MLP-only selects 96, excluding MLP selects 128, and Q-only selects 32.

You can override either list through the train CLI without modifying the configuration file. `--include-blocks` and `--exclude-blocks` are equivalent aliases:

```bash
./run.sh train --config configs/small.toml \
  --include_blocks "['*mlp*', '*.attn.to_q']" \
  --exclude_blocks "['transformer_blocks.0.*']"

# Multiple patterns are also accepted; quote them to prevent shell expansion.
./run.sh train --config configs/small.toml --exclude_blocks '*mlp*' '*.attn.to_q'

# [] clears the configured list; the other list stays unchanged if not specified.
./run.sh train --config configs/small.toml --include_blocks '[]'
```

Training initialization reports the selected and candidate target counts. The `lora_target_filter` field in `run.json` records patterns, precedence, unmatched patterns, and the complete `selected_targets` list. `--dry-run` preserves the effective configuration but does not load the DiT, so it does not claim to validate actual layer matches. Changing targets does not require rebuilding the TE/VAE cache. A warm-start checkpoint must still have **the same actual targets, rank, and alpha**; extra adapters are not silently discarded.

`smoke-test --config ...` also applies the filters, but its fixture has only two tiny blocks. This fixture cannot validate filters that select later blocks in the full model.

## CPU prefetching and W&B

Set `training.num_workers=2` and `prefetch_factor=2` to read existing safetensors ahead of time using spawn-based CPU workers, with at most 4 outstanding tasks. By default, only the main process's DiT runs on the GPU; sampling temporarily loads the TE/VAE when enabled. Workers do not run the TE/VAE, change data order, or affect noise/timestep random number generation (RNG). The default `num_workers=0` reads synchronously. Existing caches do not need rebuilding. Image preprocessing during caching does not use multiple workers.

Configure W&B in `[wandb]`; the public examples disable it by default. When you explicitly enable online logging, it uses `WANDB_API_KEY` from the environment. `log_config=true` records the full redacted configuration and a configuration artifact. `log_samples=true` can upload validation images when sampling during training is explicitly enabled. API keys, tokens, and passwords are masked. Weights and code are not uploaded, and training images are not scanned automatically. However, the full configuration includes model/dataset paths and prompts; see the privacy notes in the documentation. `mode="offline"` writes local logs only, while `enabled=false` disables W&B entirely. TOML and YAML configurations use the same schema. See `docs/wandb-config-samples.md` for details.

See `docs/prefetch-wandb.md` for the full guide. Verification covers real CPU spawn workers, actual cache reads, and numerical consistency with a tiny model. Speedups have not been measured with full-model GPU training.

## Progress display

Interactive terminals use single-line `tqdm` progress bars:

- `Cache TE/VAE`: Completed images, speed, and estimated time remaining (ETA). `encoded` counts newly encoded items; `reused` counts reused items. Image latents and text embeddings are cached together.
- `Train`: Counts optimizer steps, not gradient-accumulation microsteps. Displays `loss`, `lr`, speed, and ETA on the same line.

The trainer no longer prints a line for every image or step, or dumps the entire run.json to the terminal. Full configuration and per-step metrics are still written to `run.json` and `metrics.csv`. Initialization diagnostics, errors, and command completion summaries remain visible. Animation is disabled when output is redirected or the terminal is not a TTY, keeping control characters out of logs.

When caching resumes after an interruption, only completed files with matching sources and settings are reused; unfinished items are processed. Reuse still requires file reads for validation. `--overwrite` forces re-encoding. Progress display changes do not change the identity of existing caches.

## Output artifacts

```text
output/small/
  run.json                    # Effective config, provenance, trainable dtype/count
  metrics.csv                 # Loss, LR, and elapsed time for each optimizer update
  step-000100.safetensors
  step-000200.safetensors
  step-000300.safetensors
  final.safetensors
  completed.json              # Written only on success; includes final checkpoint hash
```

The full 32-block model attaches **224 A/B adapter pairs** by default. Training refuses to start if `output_dir` is nonempty, preventing overwrites and mixed experiment logs.

`run.json` also records the loss type and weighting, plus the optimizer class actually instantiated, its version, and its effective parameters.

### Warm start is not a full resume

Use a **new output_dir**:

```bash
./run.sh train --config configs/another-run.toml --lora /path/to/final.safetensors
```

Only DiffSynth LoRAs with matching targets/rank are accepted. The optimizer, step counter, and RNG start over; **this is not an exact training resume**. Rank, shape, or key mismatches fail rather than silently leaving MLP weights unloaded.

## Preserved DiffSynth recipe and explicit differences

The baseline preserves these defaults; wavelet and Adopt_adv are optional:

- Official split `gate_layer`/`proj` MLP and PEFT LoRA, with automatic detection of Linear targets inside blocks.
- Alpha defaults to rank but can be set independently; trainable LoRA weights use BF16.
- VAE posterior **mean**, RGBA, the original text template, and pre-final-RMSNorm embeddings.
- `FlowMatchScheduler("Qwen-Image")`, fixed training `mu=0.8`, and a 1000-timestep schedule.
- `FlowMatchSFTLoss`: Velocity target `noise - clean`, with MSE multiplied by the upstream bell-shaped weighting.
- AdamW by default; `adopt_adv` is optional. The default `diffsynth` LR schedule preserves the legacy ConstantLR behavior of LR/3 for the first 5 updates. Alternatives are a true `constant` schedule, `constant_with_warmup`, and `cosine`. Set optimizer warmup steps with `num_warmup`.

Wrapper changes:

- Uses an optimizer-step limit instead of the upstream example's epoch/repeat controls, giving short test runs a fixed length.
- Caches before training and checks model/data provenance and stale caches.
- Defaults to `attention="segmented"` to avoid the initial Flex compilation, without changing mask semantics. You can use `"flex"`; availability depends on Torch and the GPU.
- Removes the capture hook added by upstream after each TE encode to avoid accumulating hooks. The vendored upstream source is not modified.
- `model.cpu_offload=true` uses the upstream layer-offload manager **for DiT training only**. This CUDA path has not been tested and is not enabled by default.
- `training.checkpointing_offload=true` uses CPU memory through upstream activation checkpoint offloading. This is not weight quantization.

## Offline verification and diagnostics

```bash
./run.sh doctor
./run.sh verify-upstream
./run.sh smoke-test --output-dir verification/my-smoke
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 OMP_NUM_THREADS=1 \
  OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/pytest -q
```

`doctor` reports not-ready status and exits with a nonzero code when data, models, or caches are missing. It does not download models in the background.

`smoke-test` runs **a real, scaled-down upstream DiT with PEFT and the original loss** on the CPU. It checks FP32/BF16, gradient checkpointing, gradient accumulation, optimizer updates, safetensors export, and matching outputs after reloading. It also verifies the full model's 224 automatically detected targets on the meta device.

**This is not training with the full pretrained model, and it does not demonstrate generation quality or GPU throughput.** Tests do not download full model weights or automatically use private datasets. Full training and image quality still require validation through the actual workflow above.

## License and attribution

See [ATTRIBUTION.md](ATTRIBUTION.md), [LICENSE](LICENSE), and `upstream-manifest.json`. This project does not include model weights; their terms of use are governed by Qwen's official releases.
