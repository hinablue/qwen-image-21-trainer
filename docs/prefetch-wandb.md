# Train CPU 預取與 W&B logging

## CPU 預取

```toml
[training]
num_workers = 2
prefetch_factor = 2
```

- 一個主 GPU 訓練程序，兩個 spawn CPU workers；每個 worker 最多預取兩筆工作，合計最多四筆 outstanding 任務，另有主程序目前正在使用的樣本。
- workers 只用 CPU 讀既有 image latent／text embedding／mask，不載入 DiT、TE 或 VAE，也不呼叫 CUDA。
- `batch_size=None` 與 identity collator 保留每筆原始 tensor shape/dtype，不額外添加 batch 維度或 padding。
- 使用與原程式完全相同的 private Python Random(seed) shuffle/pop 次序；repeats 仍只展開取樣位置。DataLoader 使用獨立的 CPU torch.Generator，worker setup 不消耗主訓練的 noise/timestep RNG。
- 資料流涵蓋完整 `max_steps × gradient_accumulation_steps` 預算，不會每次資料循環都重啟 workers；正常结束、例外及提早離開均關閉 workers。
- `num_workers=0` 保留同步讀取。`prefetch_factor` 需為正整數，workers=0 時不啟動預取。
- 目前不開 pin_memory／CUDA-stream 預取；GB10 統一記憶體上的實際收益須量測，不能保證更多 workers 一定更快。
- CSV 新增 `data_wait_seconds`：該 optimizer step 等候 CPU sample iterator 的累計時間，不包含後續 CPU→GPU 搬移，不是 GPU kernel latency。

這次僅改 train 的快取資料讀取；cache 的圖片解碼／VAE／TE 編碼流程不變。也不會略過原本的 startup manifest/cache 完整性驗證。

## W&B

```toml
[wandb]
enabled = true
project = "qwen-image-21-trainer"
entity = ""
name = "my-experiment"
mode = "online"
log_every = 1
```

W&B 的 API key 只沿用執行環境，不提供 TOML key 欄位、不互動登入、不把 key 寫進 log。公開範例均預設 enabled=false；啟用 online 前請明確設定執行環境的 WANDB_API_KEY。entity 空白時沿用 SDK 的既有帳號／環境；name 空白時採 output_dir 的 basename。

- 記錄 step、loss、learning rate、elapsed seconds、images seen、資料等待時間與有有效 elapsed 時的累計 steps/s。
- 依 log_every 記錄，第一步及最後一步也保留。完整 CSV 仍逐 step 寫入，不因 W&B 的 logging 間隔減少。
- 預設 log_config=true 時，除原有超參數外也記錄完整脫敏配置與設定檔 artifact（包括路徑與配置中的 prompts）；明確啟用 sample 後可送生成的驗證圖片。log_config=false 可退回純超參數模式；不使用 wandb.watch、不上傳權重或程式碼，也不掃描訓練圖片。詳細欄位見 docs/wandb-config-samples.md。
- 停用 console/code/git/自動 metadata 與 stats 收集，避免破壞 tqdm 或意外帶出執行路徑；GPU 系統監控圖不在此預設範圍。
- run ID／URL 等追蹤資訊留在本機 run.json；聊天回覆不自動貼連結。
- logging 由主程序處理，CPU workers 不建立 W&B run。tracking 在設定訓練 seed 前初始化。
- 成功／異常退出都結束 run；初始化或 logging 問題不偷偷改成 offline 或忽略。CSV 是獨立保留的本機紀錄。

只做離線記錄可設 `mode="offline"`，SDK 檔案位於該 output_dir 內。完全不使用 W&B 則設 `enabled=false`。缺 API key 的 online 模式會明確失敗，不詢問互動登入。

## 使用

原有指令不變；快取已完成後：

```bash
./run.sh train --config configs/my-run.toml --dry-run
./run.sh train --config configs/my-run.toml
```

目前提供的驗證是 CPU workers、真實 cache 子集、同 seed 的 tiny FP32/BF16 模型一致性與 W&B offline/mocked paths。沒有啟動正式完整模型訓練，也不將 offline 成功當成 online 憑證／網路驗證。
