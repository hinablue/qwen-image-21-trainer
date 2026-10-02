# Learning-rate schedule 與 Krea2-style W&B 記錄

TOML 既有 `[training]`：

```toml
learning_rate = 1e-4
lr_scheduler = "constant_with_warmup"
num_warmup = 100
```

YAML 同欄位放 `training:`。`num_warmup` 是 optimizer updates，不是 accumulation microsteps、圖片數或百分比。

## 排程

- `constant`：全程使用設定的 base LR。num_warmup 必須為 0。
- `constant_with_warmup`：線性 warmup 後固定 base LR。num_warmup=0 時退化為 constant。
- `cosine`：可選線性 warmup，再用單次 half-cosine 降至 0；不包含 restarts，num_cycles 固定 0.5。
- `diffsynth`：相容預設，完整保留舊 PyTorch ConstantLR 的 first-five-updates base_lr/3，然後跳回 base_lr。這不是「全程 constant」也不是線性 warmup。num_warmup 必須為 0。

為避免偷偷改動原 recipe，未填 lr_scheduler 仍使用 diffsynth；沒有把它冒名當作 constant。

## step 邊界

新排程採 Hugging Face zero-index convention：初始化在 schedule step 0，每次 optimizer.step 後才 scheduler.step。

- num_warmup>0 時，第一個 optimizer update 用 LR=0；之後線性升高。
- N 次更新使用 schedule indices 0 到 N-1。cosine 的 0 LR 在最後 scheduler.step 到 index N 才抵達，因此最後一筆「實際更新用 LR」通常仍大於 0。
- CSV/W&B 的 learning_rate 是 optimizer.step **之前**取得的當次實際 LR，不是 step 之後已排定的下一次 LR。
- warmup 不能超過 max_steps；cosine 要求 warmup 嚴格小於 max_steps，保留衰減區間。
- `--lora` warm-start 不恢復 scheduler state，會從新 schedule step 0 開始。

smoke-test 固定執行 3 次 tiny optimizer updates。提供 production config 時會檢驗該完整排程的前三步，而不是為了 warmup 把測試偷偷延長到完整訓練；報告分別列 schedule horizon 與實際 updates。

## W&B 對照 Krea2

保留既有 `train/*` 和 `samples/images`，另加：

- `loss/current`：本次 optimizer update 的 loss（包含既定 loss weighting 與 accumulation 平均）。
- `loss/average`：本 run 至今的累計平均。Krea2 原碼是以 epoch index 更新的 moving-average recorder；本 trainer 固定 update budget，不假造 epoch 邊界或冒稱同一平均定義。
- `lr/dit`：DiT 本次實際使用 LR。不照搬 Krea2 的 unet 名称或 post-scheduler 下一步 LR。
- `sample_0`、`sample_1` 等：依設定 prompt index 穩定的圖片欄位，與該步 metrics 同次提交。
- alpha、rank、lr_scheduler、num_warmup 與完整 config 快照保留既有紀錄。

`wandb.log_training_log` 預設 true：在 run 結束前，將已 flush/關閉的數值型 metrics.csv 重建為受限欄位 snapshot，再以 training-log artifact 保存。不是整份 Python/console log；不擷取 stdout、stderr、第三方 log 或任意 output 檔案，以免帶出憑證。設 false 可關閉這份額外 artifact，不影響正常 scalar logs 或本機 CSV。

不虛構 DPO/CPO/TQD/epoch/max-norm/Prodigy 專用指標。取得離線 SDK artifact/journal 只驗證本機記錄，不等於已驗證 online 權限或雲端接收。
