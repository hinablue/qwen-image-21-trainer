# Wavelet loss 與 Adopt_adv

這兩個選項都進入實際 training forward/backward 與 optimizer factory，並非只接受設定名稱。

## 已準備的設定

- `configs/small.toml`：保留 DiffSynth MSE＋AdamW baseline recipe。
- `configs/wavelet-adopt.example.toml`：可攜範例，使用 wavelet＋Adopt_adv、rank=32／alpha=16，輸出至獨立的 `output/wavelet-adopt-example`。
- 兩份範例預設使用 official 模型 layout；可按 `docs/model-loading.md` 改成既有 Comfy BF16 權重。模型與資料路徑請自行設定，沒有附帶私人資料集。

新增設定如下：

```toml
[training]
loss_type = "wavelet"        # mse | wavelet
loss_weighting = "diffsynth" # diffsynth | none
optimizer = "adopt_adv"      # adamw | adopt_adv；Adopt_adv 亦接受
learning_rate = 0.0001
weight_decay = 0.01

[training.optimizer_params]
cautious_wd = true
kourkoutas_beta = true
use_atan2 = true
```

未填選項時維持 `mse`、`diffsynth` weighting、`adamw`、空 optimizer_params。
`learning_rate`、`weight_decay` 必須放在 `[training]`，不能在 optimizer_params 重複指定。
不支援／拼錯的 optimizer、loss 或 optimizer parameter 會明確失敗，不會偷偷改成 AdamW。

## Wavelet 的確切定義

對齊本機 ai-toolkit `toolkit/util/losses.py` 的 `wavelet_loss`：

1. `model_pred`、clean latents、sampled noise 都轉成 FP32。
2. 使用 `DWTForward(J=1, mode="zero", wave="haar")`。
3. target 是 clean latents 的 LL/LH/HL/HH 四個頻帶。
4. prediction 是 `noise - model_pred` 的相同四個頻帶。
5. 串接四頻帶後，以等權 MSE 取平均。
6. `loss_weighting="diffsynth"` 再乘上 DiffSynth 原本的 timestep weighting；`"none"` 才不乘。

這不是從 noisy latent 與 sigma 另外推導的 x0 loss，也不是自訂多層、高頻加權 wavelet。模型仍預測 flow velocity，採原本 `noise - clean` 方向。

**重要：Haar 是正交轉換。當 latent 的高寬為偶數，四頻帶等權、完整保留時，這個 scalar loss 在精確算術下等價於 velocity MSE（Parseval）。** 不要把選項名稱解讀為「一定更強調細節」。浮點运算順序會有差異；尤其上游 BF16 的 `noise-clean` 可先發生捨入，而此 wavelet 路徑先轉 FP32 再相減，因此實際數值不保證逐 bit 相同。

Wavelet filters 每個 training module 各自持有，運算維持 FP32，不使用 ai-toolkit 的 process-global DWT cache，避免裝置切換時拿到錯誤裝置的 filters。

## Adopt_adv

使用 PyPI `adv_optm==2.5.13` 真實類別：

```python
from adv_optm import Adopt_adv
```

依賴已寫入 `pyproject.toml`／`uv.lock`，安裝在本專案 `.venv`，未修改 ai-toolkit 或其他環境。wavelet 依賴為 `pytorch-wavelets==1.3.0` 及相容的 PyWavelets。

常用參數：
- `betas`（兩個介於 0 含至 1 不含的值）、`eps`。
- `cautious_wd`、`kourkoutas_beta`、`use_atan2`、`stochastic_rounding`。
- `state_precision`，採套件支援的選項；省略時保留 `auto`。

其他已允許參數與型別驗證在 `training_options.py`；callable 類型的 `clip_lambda`／`layer_key_fn` 不從 TOML 載入執行。並非套件的每個 GPU／壓縮 state 模式都經過實測；本次主要驗證上面設定檔中的三個開關、預設 state policy 與 FP32/BF16 參數。

`run.json` 記錄實際建立的 optimizer class、package version、有效 betas/eps，以及是否啟用三個開關。沒有 silent fallback。

其餘仍沿用 DiffSynth 固定 mu=0.8 的 schedule、BF16 LoRA、原 ConstantLR（前 5 個 optimizer steps LR/3），**不是把 ai-toolkit 的 linear timestep、warmup、EMA、caption dropout 等整套重現**。

ADOPT 的起始 step 會建立統計狀態；optimizer step 數不表示每一次所有參數都必定改變。短 smoke 會跨越初始化步驟並檢查實際 LoRA 值確實有更新。

## 驗證

```bash
./run.sh smoke-test --config configs/wavelet-adopt.example.toml \
  --output-dir verification/my-wavelet-adopt-smoke

CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 OMP_NUM_THREADS=1 \
  OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/pytest -q
```

已驗證：
- wavelet loss 值與 gradient 對齊 ai-toolkit 定義，涵蓋 FP32/BF16、偶數與奇數高寬。
- default MSE 分支在相同 RNG 狀態下直接呼叫上游 loss，結果完全相同。
- 真實縮小版上游 DiT 的 MSE/AdamW、MSE/Adopt_adv、wavelet/AdamW、wavelet/Adopt_adv 組合。
- gradient checkpointing、gradient accumulation、實際 optimizer 更新、safetensors 存讀與重載輸出一致。

**上述不是完整模型 GPU 訓練或影像品質驗證。** 測試新 loss/optimizer 不會自動使用私人資料集進行 cache 或完整模型訓練。
