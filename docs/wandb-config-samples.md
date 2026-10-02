# W&B 設定快照與訓練驗證圖片

## 完整 config 記錄

W&B 啟用時，`wandb.log_config` 預設為 true：

- W&B Config 保留原有 scalar 超參數，另有完整、脫敏的已解析設定。
- training-config artifact 包含 `source-config.yaml`（有從設定檔載入時）與 `effective-config.json`。
- source-config 是在 load_config 當下捕捉的設定內容；不是訓練開始後再去重讀可能已修改的檔案。
- 來源可為 TOML 或 YAML，統一重序列化為 YAML 副本並記錄原始格式。**不保留原始註解，也不是 byte-for-byte 原檔**，避免註解帶出憑證。
- effective-config 包含補齊的預設值與解析後路徑，便於重現實验。
- 憑證欄位、已知憑證環境值與 URL 登入資訊等會脫敏；不要把 API key 寫進設定或聊天。
- 本機副本置於該 run 的 `wandb-config/`，不改原始設定檔。
- 不上傳模型、程式碼、整個 output 目錄或隱含的資料集圖片。

設定中列出的模型／資料路徑、sample prompt 會隨完整配置上傳，這是此功能的範圍；需要退回原先純 scalar 超參數可設 `log_config=false`。

## 訓練時生成並上傳 sample

修改既有 TOML 區段（不要重複加相同表格標頭）：

```toml
[wandb]
enabled = true
log_config = true
log_samples = true

[sample]
enabled = true
every = 100
prompts = ["a photo of a ceramic cup on a wooden table", "a portrait in soft window light"]
negative_prompt = ""
width = 512
height = 512
steps = 30
cfg_scale = 3.0
seed = 42
```

- `sample.enabled` 預設 false，不會因原本已有 prompt 設定而偷偷增加 GPU 工作。
- 每 `every` 個 optimizer updates 與最後一步產生圖片；同一步只執行一次，不在 step 0 取樣。
- `prompts` 非空時取代單一 `prompt`；最多 16 項，依序生成。空陣列則沿用既有 `sample.prompt`。
- 每次使用固定 `seed`，方便比較不同訓練步的變化。
- PNG 與步數紀錄放在該 run 的 `samples/`，拒絕覆蓋已有圖片。
- W&B 的圖片與該 step metrics 一起提交，避免先 commit scalar 後才 log 同 step 圖片而被丟棄。
- sample 即使不在 log_every 的 scalar 間隔，仍會記錄該步圖片與指標。
- `wandb.log_samples=false` 只關閉上傳；sample.enabled=true 仍會在本機生成。
- 不會自動上傳任意外部照片或訓練資料集。sample 路徑限本次 output/samples，檢查路徑與符號連結；圖片 metadata 不作為上傳內容。

### GPU 與訓練狀態

取樣重用正在訓練、已注入 LoRA 的 DiT，不再載第二份 DiT。**但會額外載入 TE／VAE**，有額外記憶體與時間成本，不能再把啟用 sample 的情況稱作「全程只載 DiT」。VAE decode 使用 tiling；每輪取樣後釋放額外模型。

取樣使用獨立 inference scheduler，保存／恢復 RNG 與 train/eval 狀態，不更改 optimizer 或 LoRA requires_grad。暫不支援 `model.cpu_offload=true` 與訓練中 sampling 同時使用，設定驗證會先拒絕，不默默嘗試不相容的 offload hooks。

原本的單張 `sample` 子命令仍可使用；它不會自動接上已在跑的訓練 W&B run。

## YAML

`.yaml`／`.yml` 與 TOML 使用相同的 native trainer schema。例如完整模型／dataset／training 設定之外，可加入：

```yaml
wandb:
  enabled: true
  project: qwen-image-21-trainer
  log_config: true
  log_samples: true
sample:
  enabled: true
  every: 100
  prompts:
    - a photo of a ceramic cup on a wooden table
  seed: 42
  width: 512
  height: 512
  steps: 30
  cfg_scale: 3.0
```

```bash
./run.sh train --config path/to/config.yaml
```

相對路徑相對於設定檔所在目錄。YAML 使用 safe loader，拒絕 unsafe tags、重複欄位、循環／過深結構與超過 1 MiB 的檔案。這不是 ai-toolkit 的 job/process schema；ai-toolkit 設定仍需使用原有匯入流程，不能直接當作本訓練器的 config。

## 驗證界線

開發測試只用 CPU tiny/fixture 與真實 W&B offline SDK，不啟動完整模型的 GPU 取樣、不連 online、不動正在跑的訓練。Fixture PNG 只用於驗證保存／上傳資料流，不代表完整 Qwen 模型的生成品質。新功能僅在下次啟動載入，不能熱加到已在執行的 run。
