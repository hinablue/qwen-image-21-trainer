# Qwen-Image 2.1 小資料集 LoRA 訓練器

[正體中文](README.md) | [English](README.en.md)

獨立啟動、固定版本的 **DiffSynth-Studio 訓練封裝**。不依賴 ai-toolkit、不改既有訓練器，適合先用少量圖片確認 LoRA 學習行為。

- 上游固定：`modelscope/DiffSynth-Studio@974cfa37f27ac55eba3b6d10efa21f876900572d`。
- 附帶上游 Python package 原始碼，而非只複製一支不能独立執行的 `train.py`。其他模型的共用／登錄原始碼保留，但公開 CLI **只支援 Qwen-Image 2.1 T2I LoRA**。
- 第一版：**單 GPU 訓練 process（另可加 CPU 預取 workers）、每 microstep 一張圖**，支援 gradient accumulation；不支援 editing dataset、多 GPU、LoKr、量化訓練或完整 optimizer-state resume。
- 與 ai-toolkit 的 fused `gate_up` LoRA 不可直接互換。本版輸出優先用附帶的 DiffSynth `sample` 驗證。

## 安裝

```bash
git clone https://github.com/hinablue/qwen-image-21-trainer.git
cd qwen-image-21-trainer
uv sync --locked --group dev
./run.sh --help
./run.sh verify-upstream
```

需要 Python 3.11 或 3.12 與 uv。若以複製方式移到另一台主機，請複製**整個專案（含 vendor）**，不必複製 `.venv`，再執行：

```bash
uv sync --locked --group dev
```

`uv.lock` 固定依賴。此組合在 Linux aarch64／Python 3.11 安裝驗證；其他作業系統與 GPU 平台未實測。不要只安裝單獨 wheel 然後從 PyPI 隨意補另一版 DiffSynth；CLI 會檢查已安裝上游 Python 檔案 hash。

## 公開範例與本機設定

`configs/small.toml` 提供 MSE＋AdamW baseline；`configs/wavelet-adopt.example.toml` 提供 wavelet＋Adopt_adv 與 rank=32／alpha=16 的範例，W&B 與訓練取樣預設關閉。

可複製為 `configs/my-run.toml` 後填入自己的路徑，並在每個命令加入 `--config configs/my-run.toml`。`configs/` 採公開範例白名單，其他設定與匯入 provenance 預設不進 Git。模型、資料集、訓練產物、憑證與私人實驗筆記亦不包含於 repository；詳見 [版本控管與隱私](docs/repository-hygiene.md)。選項定義見 [wavelet／Adopt_adv](docs/wavelet-adopt.md)。

## 最短使用流程

### 1. 放入图片與同名 caption

```text
data/train/
  image01.png
  image01.txt
  image02.jpg
  image02.txt
```

- `.txt` 是 UTF-8 caption，不可空白。
- 支援 PNG／JPG／JPEG／WEBP／BMP，預設遞迴讀子目錄。
- 不修改原圖或 caption；拒絕壞圖、缺 caption、同目錄同 stem 的多張圖片、symlink 等有歧義輸入。
- 可以只有一張圖做流程測試；若要看是否能學到人物或風格，請使用真正有代表性的少量圖片，固定未參與訓練的 sample prompt 比較。

### 2. 調整 `configs/small.toml`

通常只要先改：

```toml
[model]
format = "official"
root = "/absolute/path/to/Qwen-Image-2.1"

[dataset]
path = "/absolute/path/to/my_small_dataset"
```

請在原本的 section 內修改，不要新增重複 section。**設定裡的相對路徑相對於 TOML/YAML 設定檔所在目錄**，不是 shell 的 cwd。

預設小測試設定：

- 最大面積 `262144`（512×512），保留比例、依上游 resize/crop 規則對齊 32 px；不是強迫全部變正方形。
- `rank = 32`；未填 alpha 時等於 rank，可另設 `alpha = 16` 等獨立值。
- LR `1e-4`、AdamW、weight decay `0.01`。
- `max_steps = 300`：精確 **optimizer update 數**，不是 epoch 或 microstep 數。
- `gradient_accumulation_steps = 1`；增為 4 時，300 updates 會使用 1200 microsteps。
- 每 100 updates 儲存；gradient checkpointing 開啟。
- DiT／LoRA BF16；不量化。

### 3. 選擇模型格式

**已有 ai-toolkit 的 Comfy BF16 檔案：**選擇 `model.format="comfy_bf16"`，`model.root` 指向含 `diffusion_models`、`text_encoders`、`vae` 的 models 目錄，並以 `model.processor_path` 指向官方 processor 目錄；完整範例見 [模型載入](docs/model-loading.md)。載入時轉換 keys／拆分 gate_up／移除 size-1 時間軸，不改來源、不另存整份模型；目前只接受純 BF16，量化檔案會被拒絕。

**官方 split 格式：**選擇 `model.format="official"`（預設），模型目錄如下：

```text
Qwen-Image-2.1/
  transformer/diffusion_pytorch_model*.safetensors
  text_encoder/model*.safetensors
  vae/diffusion_pytorch_model*.safetensors
  processor/tokenizer_config.json
  processor/...
```

**ComfyUI 的 INT8／FP8／ConvRot 等量化檔案不受此轉接支援。** BF16 的 fused `gate_up` 則可經上述 Comfy 模式載入；完整 base weights 的拆分不等於 fused LoRA 能直接互換。

已經有完整官方目錄時，直接設定 `model.root`，不需要再下載。只有在使用 official 格式且沒有模型時，才由你明確啟動下載：

```bash
./run.sh download --confirm-large-download
```

此操作會下載數十 GB 的官方模型，**不會啟動訓練**。命令先將 Hugging Face revision 解成 commit，下載 provenance 寫在模型目錄；可另外傳入 `--revision <commit>`。其他命令沒有自動下載分支。

### 4. 整理、快取、訓練

```bash
./run.sh prepare
./run.sh cache
./run.sh train --dry-run
./run.sh train
```

另選設定檔時，每個命令都加 `--config /path/to/experiment.toml`。

- `prepare`：完整檢查圖片／caption，計算內容 hash，建立 `cache/small/dataset.json`。
- `cache`：只載入 **文字編碼器與 VAE**，使用上游圖片前處理與 conditioning pipeline；保留 RGBA、使用 posterior mean。完成後存為 safetensors。
- `train --dry-run`：檢查完整資料／模型／cache 一致性，列出有效設定；不把完整模型載進 GPU、不訓練。
- `train`：預設只載入 **DiT**，使用 cache 訓練；明確啟用 `sample.enabled` 後，取樣期間會額外載入 TE/VAE。

`cache` 階段仍需要 TE＋VAE 的記憶體；sample 階段需要完整 pipeline。快取只降低訓練階段的常駐模型需求，不表示整體可以在任意小 VRAM 上執行。

圖片、caption、解析度或模型變更後，重建：

```bash
./run.sh prepare --overwrite
./run.sh cache --overwrite
```

模型權重 provenance 採檔案路徑／大小／mtime；processor/config 採內容 hash。這不是對數十 GB 模型逐一做完整權重 hash。資料圖片／caption 與 cache 本身有內容 hash。

### 5. 驗證圖片

先用相同 prompt／seed 產生 base，再產生 LoRA 圖：

```bash
./run.sh sample --output output/base.png --prompt "你的固定驗證 prompt"
./run.sh sample --lora output/small/final.safetensors \
  --output output/lora.png --prompt "你的固定驗證 prompt"
```

圖片輸出用 PNG 保留 alpha，拒絕覆蓋既有檔案。sample 參數在 `[sample]`，預設 seed 42、30 steps、CFG 3.0、512×512。這是 sample CFG，不是訓練時套用 CFG。

## Rank / alpha 與 LR schedule

設定放在既有 `[training]` 區段，不要另建重複表格：

```toml
rank = 32
alpha = 16
lr_scheduler = "constant_with_warmup"
num_warmup = 100
```

alpha 實際作用於 PEFT forward/backward，checkpoint 保存 raw A/B 加每層 alpha；同 rank/alpha warm-start 不會重複縮放。`configs/wavelet-adopt.example.toml` 示範 rank=32／alpha=16 與獨立輸出目錄。LR 排程須按實驗明確選擇；未填時保留 diffsynth baseline。細節：`docs/lora-alpha.md`、`docs/lr-schedule.md`。

W&B 同時保留原 train/* 與新增 Krea2-style loss/current、loss/average（run 累計平均）、lr/dit（當次使用 LR）、sample_N 及數值型 metrics.csv artifact；不捏造 epoch/DPO/TQD 指標或直接上傳完整 console log。

## 選擇要訓練的 blocks／layers

在既有 `[training]` 區段設定陣列（TOML／YAML 均支援）：

```toml
# 僅訓練 MLP 與 attention Q；清單內任一 pattern 命中即可。
include_blocks = ["*mlp*", "*.attn.to_q"]
# 再排除第 0 個 block；同時命中 include/exclude 時，exclude 優先。
exclude_blocks = ["transformer_blocks.0.*"]
```

- `include_blocks` 省略或 `[]`：不限制原本可掛載的 LoRA targets；非空時**僅訓練符合項目**，不需搭配 exclude-all。
- `exclude_blocks` 省略或 `[]`：不額外排除；非空時排除任一 pattern 命中的項目。
- 使用大小寫敏感的 **glob 萬用字元**，不是 Python regex：`*` 任意長度、`?` 單一字元、`[01]` 字元集合；`.` 是普通字元，`*` 可跨越多個點分隔層級。
- 比對掛載 LoRA **之前的完整 module path**，不含 `pipe.dit.`、`.weight` 或 `.lora_A...`。只篩選既有 eligible Linear 層，不擴大到 LayerNorm、embedding 或 block 外的層。
- 篩選發生在 PEFT adapter 建立**之前**；未選中的層不建立 adapter，base weights 照常凍結，也不進入 optimizer。不會跳過這些層的 forward。
- 任一 pattern 未命中候選層會警告；最後零個可訓練 target 直接報錯，不回退成全部訓練。

這版 Qwen-Image 2.1 的實際名稱例如：

```text
transformer_blocks.0.attn.to_q
transformer_blocks.0.attn.to_k
transformer_blocks.0.attn.to_v
transformer_blocks.0.attn.to_out.0
transformer_blocks.0.img_mlp.proj
transformer_blocks.0.img_mlp.out
transformer_blocks.0.img_mlp.gate_layer
```

因此 `*.attn.q*` **不會命中**這版的 Q projection，要寫 `*.attn.to_q`；整個 block 用 `transformer_blocks.0.*`。正式模型預設 224 組 adapter；僅 MLP 為 96 組、排除 MLP 為 128 組、僅 Q 為 32 組。

也可由 train CLI 覆寫個別清單，不修改設定檔（`--include-blocks`／`--exclude-blocks` 是等價別名）：

```bash
./run.sh train --config configs/small.toml \
  --include_blocks "['*mlp*', '*.attn.to_q']" \
  --exclude_blocks "['transformer_blocks.0.*']"

# 也可傳入多個 pattern；加引號避免被 shell 展開。
./run.sh train --config configs/small.toml --exclude_blocks '*mlp*' '*.attn.to_q'

# [] 清空設定檔中的清單，未指定的另一個清單保持不變。
./run.sh train --config configs/small.toml --include_blocks '[]'
```

訓練初始化顯示實際選中數／候選數；`run.json` 的 `lora_target_filter` 記錄 patterns、優先規則、未命中項目與完整 `selected_targets`。`--dry-run` 會保留有效設定，但不載入 DiT，因此不宣稱已驗證實際命中層。變更 targets 不需要重建 TE/VAE cache；warm-start checkpoint 仍要求 **相同的實際 targets、rank、alpha**，不會靜默丟棄多餘 adapter。

`smoke-test --config ...` 也會使用 filters，但它只有兩個 tiny blocks；限定正式模型較後段的 block 時不能用這個 tiny fixture 驗證。

## CPU 預取與 W&B

可設定 `training.num_workers=2`、`prefetch_factor=2`，用 spawn CPU workers 預讀現有 safetensors，最多 4 筆 outstanding 任務。預設 GPU 只有主程序的 DiT（啟用 sample 時暫載 TE/VAE）；workers 不跑 TE/VAE、不改資料順序或 noise/timestep RNG。`num_workers=0` 為預設同步讀取；既有快取不需重建。cache 的圖片前處理沒有改成多 workers。

W&B 可用 `[wandb]` 控制，公開範例預設關閉；明確開啟 online logging 時，沿用執行環境的 `WANDB_API_KEY`。`log_config=true` 記錄完整脫敏配置與設定 artifact；`log_samples=true` 可上傳明確啟用的訓練驗證圖。API keys／tokens／密碼會遮罩，不上傳權重或程式碼，也不自動掃描訓練圖片；但完整配置會包含模型／資料路徑與 prompts，詳見文件的隱私說明。`mode="offline"` 只寫本機記錄，`enabled=false` 則完全關閉。完整設定支援相同 schema 的 TOML 和 YAML。細節見 `docs/wandb-config-samples.md`。

完整說明：`docs/prefetch-wandb.md`。目前的驗證包含真實 CPU spawn、實際 cache 讀取及 tiny 模型數值一致性；沒有以正式 GPU 訓練量測加速倍率。

## 進度顯示

互動終端使用同一行更新的 `tqdm`：
- `Cache TE/VAE`：圖片完成數、速度／ETA，`encoded` 為新編碼數、`reused` 為重用數；圖片 latent 與文字 embedding 一起快取。
- `Train`：以 optimizer steps 計數（不是 gradient-accumulation microsteps），同列顯示 `loss`、`lr`、速度／ETA。

不再每張圖片／每個 step 印一行，也不把整份 run.json 印到終端；完整設定與每步 metrics 仍寫入 `run.json`、`metrics.csv`。初始化診斷、錯誤與命令完成摘要保留。重導向檔案或非 TTY 環境會停用動畫，避免控制字元灌滿 log。

中斷重跑 cache 時，只重用完成且來源／設定相符的檔案；未完成項目會繼續處理。重用仍有讀檔驗證成本，`--overwrite` 則強制重編碼。進度顯示變更不會改變既有快取身分。

## 產物

```text
output/small/
  run.json                    # 有效設定、來源、實際 trainable dtype／數量
  metrics.csv                 # 每次 optimizer update 的 loss、LR、elapsed time
  step-000100.safetensors
  step-000200.safetensors
  step-000300.safetensors
  final.safetensors
  completed.json              # 僅成功完成時寫出，包含 final checkpoint hash
```

預設完整 32-block 模型會掛載 **224 組 A/B adapter**。`output_dir` 非空時拒絕開始，避免覆蓋或把兩個實驗的 log 混在一起。

`run.json` 亦記錄 loss 類型／weighting，以及真正建立的 optimizer class、版本和有效參數。

### Warm start 不是完整 resume

在**新的 output_dir** 使用：

```bash
./run.sh train --config configs/another-run.toml --lora /path/to/final.safetensors
```

只接受同 targets／rank 的 DiffSynth LoRA；optimizer、step 與 RNG 重新開始，**不是斷點精確續跑**。rank/shape/key 不匹配會失敗，不靜默漏載 MLP。

## 保留的 DiffSynth recipe 與明示差異

Baseline 預設保留（wavelet／Adopt_adv 可另行選用）：
- 官方 split `gate_layer`／`proj` MLP、PEFT LoRA，自動偵測 block 內 Linear targets。
- alpha 預設等於 rank，但可獨立設定；可訓練 LoRA BF16。
- VAE posterior **mean**、RGBA、原始文字模板與 pre-final-RMSNorm embedding。
- `FlowMatchScheduler("Qwen-Image")`、訓練固定 `mu=0.8`、1000-timestep schedule。
- `FlowMatchSFTLoss`：velocity target `noise - clean`、MSE×上游鐘形權重。
- 預設 AdamW；可另選 `adopt_adv`。LR 預設 `diffsynth` 保留舊 ConstantLR 的前 5 次 update LR/3；另支援真正 `constant`、`constant_with_warmup`、`cosine`，以 `num_warmup` 指定 optimizer warmup steps。

封裝改動：
- 以 optimizer step 上限取代範例的 epoch/repeat 管理，支援確定的短測試長度。
- 先快取後訓練；模型／資料來源與 stale cache 檢查。
- 預設 `attention="segmented"` 避免首次 Flex compile；mask 語義不變。可改為 `"flex"`，實際可用性依 Torch/GPU。
- 每次 TE encode 後移除該次上游新增的 capture hook，避免反覆掛 hook；不修改隨附上游 source。
- `model.cpu_offload=true` 使用上游 layer-offload manager，**僅用於 DiT 訓練**。此 CUDA 路徑未實測，不預設開啟。
- `training.checkpointing_offload=true` 可透過上游 activation checkpoint offload 使用 CPU 記憶體；不等於權重量化。

## 離線驗證與診斷

```bash
./run.sh doctor
./run.sh verify-upstream
./run.sh smoke-test --output-dir verification/my-smoke
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 OMP_NUM_THREADS=1 \
  OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/pytest -q
```

`doctor` 缺少資料／模型／cache 時會回報未 ready 並以非零 exit code 結束；這不會偷偷下載模型。

`smoke-test` 用 CPU 跑**真正的上游縮小版 DiT＋PEFT＋原始 loss**，分別驗證 FP32/BF16、gradient checkpointing、gradient accumulation、optimizer updates、safetensors 匯出、重載輸出一致。另以 meta device 驗證正式模型的 224 個自動 targets。

**這不是完整預訓練模型訓練，也不證明生成品質或 GPU 吞吐。** 測試不會下載完整權重或自動使用私人資料集；完整訓練與影像品質仍須由上述實際流程驗證。

## 授權與來源

見 [ATTRIBUTION.md](ATTRIBUTION.md)、[LICENSE](LICENSE) 與 `upstream-manifest.json`。本專案不包含模型權重；模型使用條款以 Qwen 官方發布為準。
