# 獨立 LoRA rank / alpha

在 TOML 的既有 `[training]` 中設定：

```toml
rank = 32
alpha = 16
```

YAML 使用同一結構：

```yaml
training:
  rank: 32
  alpha: 16
```

## 真正作用於訓練

LoRA 分支為 `B(A(x)) × alpha/rank`，不是只改 metadata。32/16 的分支倍率為 0.5；PEFT 注入時即收到 alpha，forward/backward 和訓練中的 sample 使用同一設定。這不等於把 optimizer learning rate 減半。

省略 alpha（或 YAML null）時使用 rank，維持舊行為。明確填 1 就是 alpha=1，不是自動 sentinel。接受正有限 int/float；零、負數、bool、非有限值與無法有效縮放的極端值會拒絕。

## 儲存

- 保存未預乘 alpha/rank 的原始 A/B。
- 每層附 `<module>.alpha` scalar，以 FP64 保留 alpha 數值；A/B 維持自身 FP32/BF16 dtype。
- metadata 的 rank/alpha/scale 及 raw-storage 格式來自真正的 adapter 狀態；與呼叫端宣稱不同時拒絕。
- 週期 checkpoint 和 final 使用同一格式。不改 base weights，不重寫舊檔案。
- run.json、dry-run、W&B 都記錄 alpha 與倍率；資料快取不需重建。

## 重載

Warm-start 直接載 raw A/B 到同 rank/alpha 的 PEFT adapters，不經 inference converter，避免 alpha/rank 重複套用。不同 rank/alpha/targets/shape 會拒絕，不暗中 rescale。

單獨 `sample --lora` 按檔案內 alpha 轉換一次，推論 strength 保持 1，與 training.alpha 不混用。不能只用 safetensors metadata 判斷 alpha：每層 scalar 也必須一致。

舊 A/B-only checkpoint 視為 alpha=rank；若 metadata 宣稱非 rank alpha 而缺每層 alpha，或新格式被剝掉 alpha，會拒絕猜測。DiffSynth split MLP 不會因此變成 ai-toolkit fused MLP。

同設定訓練重載可逐 tensor/輸出比對；融合後推論因 BF16 base weight rounding／計算順序可能有誤差，不承諾 fused base 與 PEFT bitwise 相同。外部通用 PEFT loader 仍可能需要 adapter config，不宣稱所有載入器都自動理解此格式。

`--lora` 仍是權重 warm-start，不是完整 optimizer/步數/RNG/LR schedule resume；新 run 的 optimizer 與 LR schedule 重新開始。

## 分開的 preset

`configs/wavelet-adopt.example.toml` 示範 rank=32／alpha=16；變更實驗時請另設 output_dir/W&B name，避免混用舊產物。未指定 LR scheduler 時仍為 diffsynth 相容排程。完整 GPU 訓練不由功能測試自動啟動。
