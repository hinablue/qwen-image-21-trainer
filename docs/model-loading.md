# 本機 BF16 權重轉接

## 設定

```toml
[model]
format = "comfy_bf16"
root = "/absolute/path/to/comfy/models"
processor_path = "/absolute/path/to/Qwen-Image-2.1/processor"
device = "cuda"
attention = "segmented"
cpu_offload = false
```

會讀取以下既有檔案：
- `diffusion_models/qwen_image_2.1_bf16.safetensors`
- `text_encoders/qwen3vl_8b_bf16.safetensors`
- `vae/qwen_image_2.1_vae_bf16.safetensors`

僅支援完整、未量化、全 BF16 的 Qwen-Image 2.1 Comfy layout。混合 fused/split、量化 companion、非 BF16、缺 keys、額外 keys、不符 shape 都明確失敗。

保留原官方模式 `format="official"`，以 split-layout 目錄經原 `from_pretrained` 載入；`processor_path` 空白時使用該 root 的 processor 子目錄。預設 baseline 為 official；使用 Comfy BF16 時請將範例中的占位路徑換成自己的位置。

## 轉接是否改變模型？

不計算新權重，也不重新量化。
- DiT：`gate_up` 上半部是 gate，下半部是 up；以 rows view 拆成 `gate_layer`／`proj`。
- TE：只轉換 Comfy／官方／DiffSynth wrapper 的命名層級，wrapper prefix 只套用一次。
- VAE：轉換 Wan-style 模組名稱，只移除長度恰好為 1 的 temporal axis。

對完整 base model，這是數學等價的參數布局轉換，BF16 tensor payload 不改動。CPU 載入可保留原 mmap storage；GPU 使用時仍需要一般的 CPU→GPU 搬移，不能把「零拷貝轉接」誤解為 GPU 不占記憶體。

不同 GEMM kernel／累加順序可能影響 BF16 rounding，**不保證完整生成結果逐 bit 一致**。之後的 LoRA 採 DiffSynth 分開的 gate/up adapters，與 ai-toolkit fused LoRA 共用 A 的參數化不同；這不表示 base weights 已改變。

## 載入流程

只把 converted state_dict 放進 ModelConfig 並不夠：原 `auto_load_model` 仍會 hash 原始檔案。此 adapter 在完整 header/key/shape 驗證後，使用固定 registry 的原 `ModelPool.load_model_file` 進行 strict assignment，不 monkeypatch upstream registry 或修改 vendor。

- `cache`：只真正載入 TE＋VAE。
- `train`：預設只真正載入 DiT；`sample.enabled=true` 會在取樣期間額外載 TE/VAE，並共用當前 DiT。
- `sample`：三者都載入。

header 預檢可能讀取所有 role 的描述資訊，但不等於將所有 tensor payload 載入 GPU。外部 processor 的 JSON／tokenizer 內容與映射版本均納入 cache signature；位於 `~/.cache` 不會被錯誤排除。

## 已完成的實際驗證

開發期間曾使用本機完整 Comfy BF16 權重、將 device 設為 CPU，實際執行 DiT-only 和 TE+VAE-only pipeline 載入；來源權重與私人路徑不隨 repository 發布。

- DiT：265 個來源 tensors → 297 個嚴格載入參數。
- TE：750 → 750。
- VAE：238 → 238。
- 每一個載入參數均與獨立重新開啟的來源 tensor view 做 int16 bit-view 比對，全部相同。
- CPU BF16 dtype、storage pointer、非指定 role 為 None、processor 類別、TE FP32 rotary buffers 均驗證；沒有殘留 meta tensor。
- 三個來源的 header SHA-256、size、mtime、ctime、inode 前後不變；未寫出完整轉換 checkpoint。
- CUDA 在此驗證中未初始化。

上述完整權重載入報告為開發本機驗證，不隨 repository 發布，也不是每次測試都重跑；公開測試使用合成 fixture／meta 模型。這不等於完整模型的 GPU forward、資料編碼或生成品質驗證。tensor payload 的全量比對與 digest 也不是發布者真偽認證。
