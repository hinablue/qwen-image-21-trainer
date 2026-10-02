# 版本控管與隱私

Repository 只包含可重現訓練器所需的程式、測試、文件、依賴 lockfile、可攜設定範例，以及固定版本的 DiffSynth vendor source／授權／hash manifests。模型權重與訓練資料不隨程式發布。

## 預設不提交

`.gitignore` 排除：

- Python 虛擬環境、bytecode、pytest／Ruff／其他工具快取。
- 任意層級的 `build/`、`dist/`、egg-info，包含 vendor 安裝時產生的副本。
- `.env`／`.env.*`、私鑰與常見登入設定；只有命名為 `.env.example`／`.env.*.example` 的占位範例例外。
- 根目錄的模型、資料集、cache、output、verification、checkpoints、logs、runs、samples，以及 W&B 本機紀錄。
- safetensors、checkpoint、GGUF、ONNX、numpy arrays 等大型權重／中間產物與 log。
- 本機實驗設定、ai-toolkit 匯入 provenance 與私人資料清單。
- IDE、OS、暫存及備份檔案。

`configs/` 採白名單：只追蹤 `small.toml` 與經審查的 `*.example.toml`／`*.example.yaml`／`*.example.yml`。範例檔名不是自動脫敏機制，仍須檢查內容。

```bash
cp configs/small.toml configs/my-run.toml
# 修改 my-run.toml 的模型／資料路徑，再執行各階段：
./run.sh prepare --config configs/my-run.toml
./run.sh cache --config configs/my-run.toml
./run.sh train --config configs/my-run.toml --dry-run
./run.sh train --config configs/my-run.toml
```

`my-run.toml` 預設被忽略。若資料或輸出放在其他自訂目錄，請將其加入本機 `.git/info/exclude`，或放在 repository 外。私人筆記可放在 `docs/local/`。不要使用 `git add -f` 強制加入私密檔案。

## 發布前檢查

```bash
git status --short --untracked-files=all
git diff --cached --stat
git diff --cached --check
git ls-files
# 例：確認本機設定與 vendor build 命中 ignore 規則
git check-ignore -v configs/my-run.toml vendor/DiffSynth-Studio/build/example.py
```

`.gitignore` 不會停止追蹤已提交檔案，也不會清除歷史。發布前仍須檢查 staged **內容**是否含私人路徑、prompts、憑證、實驗產物或未授權素材；一旦憑證曾公開，移除檔案不等於撤銷憑證。

W&B 是另一條上傳路徑，不受 Git ignore 控制。公開範例預設停用 W&B；啟用後的配置與圖片上傳範圍見 [W&B 設定](wandb-config-samples.md)。
